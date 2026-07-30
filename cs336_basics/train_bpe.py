import os
from typing import BinaryIO
import regex as re
from collections import Counter
from collections import defaultdict


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # initilize outputs
    vocab = {i:bytes([i]) for i in range(256)}
    merges = []

    # add special_tokens
    for i, v in enumerate(special_tokens):
        vocab[len(vocab)] = v.encode("utf_8")

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size must be at least {len(vocab)}, "
            f"but got {vocab_size}"
        )

    # 将special_tokens从长到短排列
    if special_tokens:
        sorted_special_tokens = sorted(
            special_tokens,
            key=len,
            reverse=True,
        )
    
    # 构造special_pattern，便于匹配
        special_pattern = "|".join(
            re.escape(token)
            for token in sorted_special_tokens
        )
    else:
        special_pattern = None

    word_freq: Counter[tuple[bytes, ...]] = Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 将文本切分成不同chunk
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # 去掉special tokens，避免被算入后续的merge
            if special_pattern is not None:
                text_parts = re.split(special_pattern, chunk)
            else:
                text_parts = [chunk]  

            for text_part in text_parts:
                for match in re.finditer(PAT, text_part):
                    pre_token = match.group()
                    token_bytes = pre_token.encode("utf-8")                    
                    byte_seq = tuple(bytes([byte_value]) for byte_value in token_bytes)

                    word_freq[byte_seq] += 1

        # 优化：再每一轮merge时只修改含有该pair的word
        words = list(word_freq.keys())
        freqs = [word_freq[word] for word in words]

        pair_counts = Counter()
        pair_to_word_ids = defaultdict(set)

        for i in range(len(words)):
            word = words[i]
            for pair in zip(word[:-1],word[1:]):
                pair_counts[pair] += freqs[i]
                pair_to_word_ids[pair].add(i)

        while len(vocab) < vocab_size and pair_counts:

            max_pair = max(pair_counts, key=lambda pair:(pair_counts[pair], pair))
            
            token_1, token_2 = max_pair
            merged_token = token_1+token_2

            merges.append(max_pair)
            vocab[len(vocab)] = merged_token

            include_pair_ids = list(pair_to_word_ids[max_pair])

            for i in include_pair_ids:
                word = words[i]
                frequency = freqs[i]

                old_pair_counts = Counter(zip(word[:-1], word[1:]))

                for pair, occurrence_count in old_pair_counts.items():
                    pair_counts[pair] -= occurrence_count * frequency
                    pair_to_word_ids[pair].discard(i)

                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                        pair_to_word_ids.pop(pair, None)

                index = 0
                merged_word = []

                while index < len(word):
                    if (
                        index + 1 < len(word)
                        and word[index] == token_1
                        and word[index + 1] == token_2
                    ):
                        merged_word.append(merged_token)
                        index += 2

                    else:
                        merged_word.append(word[index])
                        index += 1

                new_word = tuple(merged_word)
                words[i] = new_word 

                new_pair_counts = Counter(zip(new_word[:-1], new_word[1:]))       
                for pair, occurrence_count in new_pair_counts.items():
                    pair_counts[pair] += occurrence_count * frequency
                    pair_to_word_ids[pair].add(i)

       # # merge(naive)
        # while len(vocab) < vocab_size:
        #     pairs = Counter()
        #     for word, num in word_freq.items():
        #         for token_1, token_2 in zip(word[:-1], word[1:]):
        #             pairs[(token_1, token_2)] += num

        #     if not pairs: break

        #     max_pair = max(pairs, key=lambda pair:(pairs[pair], pair))

        #     t1, t2 = max_pair
        #     merged_token = t1+t2
        #     merges.append(max_pair)
        #     vocab[len(vocab)] = merged_token

        #     # form new pairs
        #     new_word_freq = Counter()

        #     for word_tokens, frequency in word_freq.items():
        #         merged_word_tokens = []
        #         index = 0

        #         while index < len(word_tokens):
        #             if (
        #                 index + 1 < len(word_tokens)
        #                 and word_tokens[index] == t1
        #                 and word_tokens[index + 1] == t2
        #             ):
        #                 merged_word_tokens.append(merged_token)
        #                 index += 2
        #             else:
        #                 merged_word_tokens.append(word_tokens[index])
        #                 index += 1

        #         new_word_freq[tuple(merged_word_tokens)] += frequency

        #     word_freq = new_word_freq                                   
            
    return vocab, merges
