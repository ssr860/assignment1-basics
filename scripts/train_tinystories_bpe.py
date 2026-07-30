import pickle
import time
from pathlib import Path

from tests.adapters import run_train_bpe


DATA_PATH = Path("data/TinyStoriesV2-GPT4-train.txt")
OUTPUT_DIR = Path("artifacts/tinystories_bpe")


def main():
    start_time = time.perf_counter()

    vocab, merges = run_train_bpe(
        input_path=DATA_PATH,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )

    elapsed_time = time.perf_counter() - start_time

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(OUTPUT_DIR / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    longest_id, longest_token = max(
        vocab.items(),
        key=lambda item: len(item[1]),
    )

    print(f"Training time: {elapsed_time:.2f} seconds")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Longest token id: {longest_id}")
    print(f"Longest token length: {len(longest_token)} bytes")
    print(f"Longest token bytes: {longest_token!r}")
    print(
        "Longest token decoded:",
        longest_token.decode("utf-8", errors="replace"),
    )


if __name__ == "__main__":
    main()
