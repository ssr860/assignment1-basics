from __future__ import annotations

import pickle
from collections.abc import Iterable, Iterator
from typing import Any

import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,    
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens

        self.token_to_id = {token_bytes: token_id
                             for token_id, token_bytes in self.vocab.items()}

        self.merge_ranks = {merges[i]: i for i in range(len(merges))}

        self.special_token_to_id = {}

        if special_tokens is not None:
            for special_token in special_tokens:
                special_token_byte = special_token.encode("utf_8")
                if special_token_byte not in self.token_to_id:
                    index = max(self.vocab.keys(), default = -1) +1
                    self.vocab[index] = special_token_byte
                    self.special_token_to_id[special_token] = index
                    self.token_to_id[special_token_byte] = index
                else:
                    self.special_token_to_id[special_token] = self.token_to_id[special_token_byte]

        if special_tokens:
            sorted_special_tokens = sorted(
                special_tokens,
                key=len,
                reverse=True,
            )

            # escape译为转译
            escaped_tokens = [
                re.escape(token)
                for token in sorted_special_tokens
            ]
            
            self.special_pattern = re.compile(
                "(" + "|".join(escaped_tokens) + ")"
            )
            
        else:
            self.special_pattern = None



    @classmethod
    def from_files(
        cls,
        vocab_filepath:str,
        merges_filepath:str,
        special_tokens:list[str] | None = None, 
        ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab: dict[int, bytes] = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges: list = pickle.load(f)

        return cls(
            vocab, merges, special_tokens
        )


    def _split_special_tokens(
            self,
            text:str,
        )->list[str]:

        if self.special_pattern is not None:
            text_parts = re.split(self.special_pattern, text)

        else:
            text_parts = [text]  

        return text_parts


    # 对每个词调用该函数为iter做准备
    def _apply_bpe(
            self,
            pre_token: str,
    ) ->list[bytes]:
        tokens = [bytes([value]) 
                           for value in pre_token.encode("utf_8")]

        while len(tokens) >= 2:
            merged_tokens = []

            candidate_pairs = [pair 
                              for pair in zip(tokens,tokens[1:]) 
                              if pair in self.merge_ranks]

            if not candidate_pairs: break

            cur_merge = min(candidate_pairs, 
                            key=lambda pair: self.merge_ranks[pair])

            index = 0

            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and tokens[index] == cur_merge[0]
                    and tokens[index + 1] == cur_merge[1]
                ):
                    merged_tokens.append(
                        tokens[index] + tokens[index + 1]
                    )
                    index += 2
                else:
                    merged_tokens.append(tokens[index])
                    index += 1

            tokens = merged_tokens      

        return tokens


    def encode(
        self,
        text: str,
    ) -> list[int]:

        output_ids: list[int] = []
        texts = self._split_special_tokens(text)
        for text in texts:
            if text == "":
                continue

            if text in self.special_token_to_id:
                output_ids.append(self.special_token_to_id[text])

            else:
                for match in PAT.finditer(text):
                    pre_token = match.group(0)
                    bpe_tokens = self._apply_bpe(pre_token)
                    output_ids.extend(self.token_to_id[token]  for token in bpe_tokens)
        return output_ids


    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)


    def decode(
        self,
        ids: list[int],
    ) -> str:    
        return b"".join([self.vocab[i] for i in ids]).decode("utf_8", errors="replace")