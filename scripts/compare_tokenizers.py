import gc
import json
import pickle
import random
import statistics
import time
from pathlib import Path

import numpy as np

from cs336_basics.get_tokenizer import Tokenizer


SPECIAL_TOKEN = "<|endoftext|>"

COMPARE_TEXT = (
    # "The little girl opened the door and found a beautiful garden."
    "Three senior administration officials told Reuters that the president "
    "is considering putting an import tariff on Chinese steel."
)

# The Pile 的 825 GB 按十进制 GB 计算。
PILE_SIZE_BYTES = 825 * 10**9

# 每次读取前 16 MiB 文本进行吞吐率测试。
BENCHMARK_MAX_BYTES = 16 * 1024**2
BENCHMARK_REPEATS = 3

# 是否真正编码四个完整数据集。
# 设为 True 后，运行脚本会生成四个 .npy 文件。
RUN_DATASET_ENCODING = True

# 每编码这么多个 token，打印一次进度。
PROGRESS_EVERY_TOKENS = 10_000_000


def load_tokenizer(
    vocab_path: str | Path,
    merges_path: str | Path,
) -> Tokenizer:
    """从 pickle 文件加载 vocab 和 merges。"""
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    with open(merges_path, "rb") as f:
        merges = pickle.load(f)

    return Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=[SPECIAL_TOKEN],
    )


def sample_documents(
    path: str | Path,
    num_documents: int = 10,
    seed: int = 42,
) -> list[str]:
    """
    使用 reservoir sampling 流式随机抽取文档。

    TinyStories 和 OpenWebText 使用 <|endoftext|>
    分隔不同文档。
    """
    rng = random.Random(seed)

    samples: list[str] = []
    current_parts: list[str] = []
    document_count = 0

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            parts = line.split(SPECIAL_TOKEN)

            for index, part in enumerate(parts):
                current_parts.append(part)

                if index < len(parts) - 1:
                    document = "".join(current_parts).strip()
                    current_parts = []

                    if not document:
                        continue

                    document_count += 1

                    if len(samples) < num_documents:
                        samples.append(document)
                    else:
                        replacement_index = rng.randrange(
                            document_count
                        )

                        if replacement_index < num_documents:
                            samples[replacement_index] = document

    # 处理文件末尾没有 <|endoftext|> 的情况。
    final_document = "".join(current_parts).strip()

    if final_document:
        document_count += 1

        if len(samples) < num_documents:
            samples.append(final_document)
        else:
            replacement_index = rng.randrange(document_count)

            if replacement_index < num_documents:
                samples[replacement_index] = final_document

    return samples


def evaluate(
    tokenizer: Tokenizer,
    documents: list[str],
) -> dict[str, int | float]:
    """计算一组文档的 bytes/token。"""
    total_bytes = 0
    total_tokens = 0

    for document in documents:
        token_ids = tokenizer.encode(document)

        total_bytes += len(document.encode("utf-8"))
        total_tokens += len(token_ids)

    bytes_per_token = (
        total_bytes / total_tokens
        if total_tokens > 0
        else 0.0
    )

    return {
        "documents": len(documents),
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": bytes_per_token,
    }


def print_result(
    tokenizer_name: str,
    dataset_name: str,
    result: dict[str, int | float],
) -> None:
    """打印 tokenizer 压缩率结果。"""
    print(
        f"{tokenizer_name:24s} on {dataset_name:12s}: "
        f"{int(result['bytes']):8d} bytes, "
        f"{int(result['tokens']):8d} tokens, "
        f"{float(result['bytes_per_token']):.4f} bytes/token"
    )


def print_documents(
    dataset_name: str,
    documents: list[str],
    num_to_show: int = 3,
    max_characters: int = 2000,
) -> None:
    """打印随机抽取的若干篇文档。"""
    print("\n")
    print("#" * 80)
    print(f"{dataset_name}: showing {num_to_show} sampled documents")
    print("#" * 80)

    for index, document in enumerate(
        documents[:num_to_show],
        start=1,
    ):
        print(
            f"\n{'=' * 24} "
            f"{dataset_name} document {index} "
            f"{'=' * 24}"
        )

        print(document[:max_characters])

        if len(document) > max_characters:
            print(
                f"\n... truncated "
                f"({len(document)} total characters)"
            )


def get_token_bytes(
    tokenizer: Tokenizer,
    token_id: int,
) -> bytes:
    """根据 token ID 查询对应的词表 bytes。"""
    token_bytes = tokenizer.vocab[token_id]

    if not isinstance(token_bytes, bytes):
        raise TypeError(
            f"Expected vocab[{token_id}] to be bytes, "
            f"but got {type(token_bytes).__name__}"
        )

    return token_bytes


def print_tokenization(
    tokenizer_name: str,
    tokenizer: Tokenizer,
    text: str,
) -> None:
    """逐个打印一句话使用了哪些 vocab token。"""
    token_ids = tokenizer.encode(text)

    print("\n")
    print("#" * 80)
    print(f"{tokenizer_name} tokenization")
    print("#" * 80)

    print(f"Original text : {text!r}")
    print(f"UTF-8 bytes   : {len(text.encode('utf-8'))}")
    print(f"Token count   : {len(token_ids)}")
    print(f"Token IDs     : {token_ids}")

    print("\nDetailed tokens:")

    reconstructed_bytes = bytearray()

    for position, token_id in enumerate(token_ids):
        token_bytes = get_token_bytes(
            tokenizer,
            token_id,
        )

        reconstructed_bytes.extend(token_bytes)

        decoded_piece = token_bytes.decode(
            "utf-8",
            errors="replace",
        )

        print(
            f"[{position:02d}] "
            f"id={token_id:5d} | "
            f"length={len(token_bytes):2d} bytes | "
            f"bytes={token_bytes!r} | "
            f"text={decoded_piece!r}"
        )

    reconstructed_text = bytes(
        reconstructed_bytes
    ).decode(
        "utf-8",
        errors="replace",
    )

    print(f"\nReconstructed : {reconstructed_text!r}")
    print(
        "Matches input :",
        reconstructed_text == text,
    )


def print_side_by_side_summary(
    tiny_tokenizer: Tokenizer,
    owt_tokenizer: Tokenizer,
    text: str,
) -> None:
    """并列显示两个 tokenizer 的 token bytes。"""
    tiny_ids = tiny_tokenizer.encode(text)
    owt_ids = owt_tokenizer.encode(text)

    tiny_pieces = [
        get_token_bytes(tiny_tokenizer, token_id)
        for token_id in tiny_ids
    ]

    owt_pieces = [
        get_token_bytes(owt_tokenizer, token_id)
        for token_id in owt_ids
    ]

    print("\n")
    print("#" * 80)
    print("Side-by-side tokenization summary")
    print("#" * 80)

    print(f"Text: {text!r}")

    print(
        "\nTinyStories tokenizer "
        f"({len(tiny_ids)} tokens):"
    )
    print(tiny_pieces)

    print(
        "\nOpenWebText tokenizer "
        f"({len(owt_ids)} tokens):"
    )
    print(owt_pieces)


def read_benchmark_text(
    path: str | Path,
    max_bytes: int,
) -> tuple[str, int]:
    """
    读取固定大小的字节，并解码成 tokenizer 所需的 str。

    返回：
        text：实际送入 tokenizer 的字符串
        encoded_bytes：该字符串重新编码后的 UTF-8 字节数
    """
    with open(path, "rb") as f:
        raw_bytes = f.read(max_bytes)

    text = raw_bytes.decode(
        "utf-8",
        errors="replace",
    )

    encoded_bytes = len(text.encode("utf-8"))

    return text, encoded_bytes


def benchmark_tokenizer(
    tokenizer_name: str,
    tokenizer: Tokenizer,
    dataset_path: str | Path,
    max_bytes: int = BENCHMARK_MAX_BYTES,
    repeats: int = BENCHMARK_REPEATS,
) -> dict[str, float | int]:
    """
    测量 tokenizer.encode() 的纯编码吞吐率。

    数据读取发生在计时之前，因此结果基本不包含磁盘 I/O。
    """
    text, input_bytes = read_benchmark_text(
        dataset_path,
        max_bytes,
    )

    if not text:
        raise ValueError(
            f"Benchmark file is empty: {dataset_path}"
        )

    # 小规模预热，降低第一次调用带来的额外影响。
    warmup_text = text[: min(len(text), 100_000)]
    tokenizer.encode(warmup_text)

    elapsed_times: list[float] = []
    token_counts: list[int] = []

    for repeat in range(1, repeats + 1):
        gc.collect()

        start_time = time.perf_counter()
        token_ids = tokenizer.encode(text)
        elapsed = time.perf_counter() - start_time

        elapsed_times.append(elapsed)
        token_counts.append(len(token_ids))

        print(
            f"{tokenizer_name} benchmark "
            f"run {repeat}/{repeats}: "
            f"{elapsed:.4f} seconds, "
            f"{len(token_ids):,} tokens"
        )

        del token_ids

    median_seconds = statistics.median(
        elapsed_times
    )

    median_token_count = int(
        statistics.median(token_counts)
    )

    bytes_per_second = input_bytes / median_seconds
    mib_per_second = bytes_per_second / 1024**2

    pile_seconds = PILE_SIZE_BYTES / bytes_per_second
    pile_hours = pile_seconds / 3600
    pile_days = pile_hours / 24

    print("\n" + "-" * 80)
    print(f"Tokenizer       : {tokenizer_name}")
    print(f"Benchmark file  : {dataset_path}")
    print(f"Input bytes     : {input_bytes:,}")
    print(f"Output tokens   : {median_token_count:,}")
    print(f"Median time     : {median_seconds:.4f} seconds")
    print(
        f"Throughput      : "
        f"{bytes_per_second:,.2f} bytes/second"
    )
    print(
        f"Throughput      : "
        f"{mib_per_second:.2f} MiB/second"
    )
    print(
        f"Estimated Pile  : "
        f"{pile_hours:.2f} hours "
        f"({pile_days:.2f} days)"
    )
    print("-" * 80)

    return {
        "input_bytes": input_bytes,
        "tokens": median_token_count,
        "seconds": median_seconds,
        "bytes_per_second": bytes_per_second,
        "mib_per_second": mib_per_second,
        "pile_hours": pile_hours,
        "pile_days": pile_days,
    }


def encode_dataset_to_uint16(
    tokenizer_name: str,
    tokenizer: Tokenizer,
    input_path: str | Path,
    output_path: str | Path,
    buffer_size: int = 1_000_000,
) -> dict[str, int | float | str]:
    """
    将完整文本数据集编码为一维 uint16 NumPy 数组。

    为避免把全部 token ID 放进内存：
    1. 先分块写入临时 raw uint16 文件；
    2. 使用 np.memmap 映射临时文件；
    3. 保存为标准 .npy 文件；
    4. 删除临时文件。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: {input_path}"
        )

    if not hasattr(tokenizer, "encode_iterable"):
        raise AttributeError(
            "Tokenizer must implement encode_iterable() "
            "to encode large datasets without loading "
            "the entire file into memory."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    total_tokens = 0
    max_token_id = -1
    buffer: list[int] = []

    source_bytes = input_path.stat().st_size

    print("\n")
    print("#" * 80)
    print(f"Encoding dataset with {tokenizer_name}")
    print("#" * 80)
    print(f"Input       : {input_path}")
    print(f"Input bytes : {source_bytes:,}")
    print(f"Output      : {output_path}")
    print(f"Datatype    : uint16")

    start_time = time.perf_counter()

    with (
        input_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as input_file,
        temporary_path.open("wb") as output_file,
    ):
        token_iterator = tokenizer.encode_iterable(
            input_file
        )

        next_progress_point = PROGRESS_EVERY_TOKENS

        for token_id in token_iterator:
            token_id = int(token_id)

            if not 0 <= token_id <= np.iinfo(np.uint16).max:
                raise ValueError(
                    f"Token ID {token_id} cannot be "
                    "represented by uint16."
                )

            buffer.append(token_id)

            total_tokens += 1
            max_token_id = max(max_token_id, token_id)

            if len(buffer) >= buffer_size:
                np.asarray(
                    buffer,
                    dtype=np.uint16,
                ).tofile(output_file)

                buffer.clear()

            if total_tokens >= next_progress_point:
                elapsed = time.perf_counter() - start_time

                print(
                    f"Encoded {total_tokens:,} tokens "
                    f"in {elapsed:.2f} seconds"
                )

                next_progress_point += (
                    PROGRESS_EVERY_TOKENS
                )

        if buffer:
            np.asarray(
                buffer,
                dtype=np.uint16,
            ).tofile(output_file)

            buffer.clear()

    if total_tokens == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"No tokens were produced from {input_path}"
        )

    # 将临时 raw 文件映射为 NumPy 数组，不将其整体读入 RAM。
    token_memmap = np.memmap(
        temporary_path,
        dtype=np.uint16,
        mode="r",
        shape=(total_tokens,),
    )

    np.save(
        output_path,
        token_memmap,
        allow_pickle=False,
    )

    del token_memmap
    temporary_path.unlink()

    elapsed = time.perf_counter() - start_time
    encoding_throughput = source_bytes / elapsed

    output_bytes = output_path.stat().st_size

    metadata = {
        "tokenizer": tokenizer_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dtype": "uint16",
        "token_count": total_tokens,
        "max_token_id": max_token_id,
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "elapsed_seconds": elapsed,
        "end_to_end_bytes_per_second": encoding_throughput,
    }

    metadata_path = output_path.with_suffix(
        ".json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(f"Token count : {total_tokens:,}")
    print(f"Maximum ID  : {max_token_id:,}")
    print(f"Output size : {output_bytes:,} bytes")
    print(f"Time        : {elapsed:.2f} seconds")
    print(
        f"End-to-end  : "
        f"{encoding_throughput:,.2f} bytes/second"
    )
    print(f"Metadata    : {metadata_path}")
    print(
        "\nLoad later with:\n"
        f'np.load("{output_path}", mmap_mode="r")'
    )

    return metadata


def main() -> None:
    tiny_tokenizer = load_tokenizer(
        "artifacts/tinystories_bpe/vocab.pkl",
        "artifacts/tinystories_bpe/merges.pkl",
    )

    owt_tokenizer = load_tokenizer(
        "artifacts/owt_bpe/vocab.pkl",
        "artifacts/owt_bpe/merges.pkl",
    )

    # ------------------------------------------------------------------
    # 1. 随机抽取测试文档
    # ------------------------------------------------------------------

    tiny_documents = sample_documents(
        "data/TinyStoriesV2-GPT4-valid.txt",
        num_documents=10,
        seed=42,
    )

    owt_documents = sample_documents(
        "data/owt_valid.txt",
        num_documents=10,
        seed=42,
    )

    # ------------------------------------------------------------------
    # 2. 比较两个 tokenizer 的压缩率
    # ------------------------------------------------------------------

    experiments = [
        (
            "TinyStories tokenizer",
            "TinyStories",
            tiny_tokenizer,
            tiny_documents,
        ),
        (
            "OWT tokenizer",
            "TinyStories",
            owt_tokenizer,
            tiny_documents,
        ),
        (
            "TinyStories tokenizer",
            "OpenWebText",
            tiny_tokenizer,
            owt_documents,
        ),
        (
            "OWT tokenizer",
            "OpenWebText",
            owt_tokenizer,
            owt_documents,
        ),
    ]

    print("#" * 80)
    print("Compression comparison")
    print("#" * 80)

    for (
        tokenizer_name,
        dataset_name,
        tokenizer,
        documents,
    ) in experiments:
        result = evaluate(
            tokenizer,
            documents,
        )

        print_result(
            tokenizer_name,
            dataset_name,
            result,
        )

    # ------------------------------------------------------------------
    # 3. 打印三篇 TinyStories 和三篇 OpenWebText
    #
    # 目前根据你的要求注释掉。需要查看时取消注释。
    # ------------------------------------------------------------------

    # print_documents(
    #     dataset_name="TinyStories",
    #     documents=tiny_documents,
    #     num_to_show=3,
    #     max_characters=2000,
    # )

    # print_documents(
    #     dataset_name="OpenWebText",
    #     documents=owt_documents,
    #     num_to_show=3,
    #     max_characters=2000,
    # )

    # ------------------------------------------------------------------
    # 4. 比较两个 tokenizer 对同一句话的具体切分
    # ------------------------------------------------------------------

    print_side_by_side_summary(
        tiny_tokenizer=tiny_tokenizer,
        owt_tokenizer=owt_tokenizer,
        text=COMPARE_TEXT,
    )

    print_tokenization(
        tokenizer_name="TinyStories tokenizer",
        tokenizer=tiny_tokenizer,
        text=COMPARE_TEXT,
    )

    print_tokenization(
        tokenizer_name="OpenWebText tokenizer",
        tokenizer=owt_tokenizer,
        text=COMPARE_TEXT,
    )

    # ------------------------------------------------------------------
    # 5. 问题 (c)：测量 tokenizer throughput
    # ------------------------------------------------------------------

    print("\n")
    print("#" * 80)
    print("Tokenizer throughput benchmarks")
    print("#" * 80)

    benchmark_tokenizer(
        tokenizer_name="TinyStories tokenizer",
        tokenizer=tiny_tokenizer,
        dataset_path="data/TinyStoriesV2-GPT4-valid.txt",
    )

    benchmark_tokenizer(
        tokenizer_name="OpenWebText tokenizer",
        tokenizer=owt_tokenizer,
        dataset_path="data/owt_valid.txt",
    )

    # ------------------------------------------------------------------
    # 6. 问题 (d)：将四个数据集编码为 uint16 NumPy 数组
    # ------------------------------------------------------------------

    if RUN_DATASET_ENCODING:
        datasets_to_encode = [
            (
                "TinyStories tokenizer",
                tiny_tokenizer,
                "data/TinyStoriesV2-GPT4-train.txt",
                "artifacts/tokenized/"
                "tinystories_train_uint16.npy",
            ),
            (
                "TinyStories tokenizer",
                tiny_tokenizer,
                "data/TinyStoriesV2-GPT4-valid.txt",
                "artifacts/tokenized/"
                "tinystories_valid_uint16.npy",
            ),
            (
                "OpenWebText tokenizer",
                owt_tokenizer,
                "data/owt_train.txt",
                "artifacts/tokenized/"
                "owt_train_uint16.npy",
            ),
            (
                "OpenWebText tokenizer",
                owt_tokenizer,
                "data/owt_valid.txt",
                "artifacts/tokenized/"
                "owt_valid_uint16.npy",
            ),
        ]

        for (
            tokenizer_name,
            tokenizer,
            input_path,
            output_path,
        ) in datasets_to_encode:
            encode_dataset_to_uint16(
                tokenizer_name=tokenizer_name,
                tokenizer=tokenizer,
                input_path=input_path,
                output_path=output_path,
            )


if __name__ == "__main__":
    main()