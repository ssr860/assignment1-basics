from __future__ import annotations

import pickle
import time
from pathlib import Path

from tests.adapters import run_train_bpe


DATA_PATH = Path("data/owt_train.txt")
OUTPUT_DIR = Path("artifacts/owt_bpe")

VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<|endoftext|>"]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find training data: {DATA_PATH.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training data: {DATA_PATH.resolve()}", flush=True)
    print(f"Vocabulary size: {VOCAB_SIZE}", flush=True)
    print(f"Special tokens: {SPECIAL_TOKENS}", flush=True)
    print("Starting OpenWebText BPE training...", flush=True)

    start_time = time.perf_counter()

    vocab, merges = run_train_bpe(
        input_path=DATA_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    training_time = time.perf_counter() - start_time

    assert len(vocab) == VOCAB_SIZE
    assert b"<|endoftext|>" in vocab.values()

    vocab_path = OUTPUT_DIR / "vocab.pkl"
    merges_path = OUTPUT_DIR / "merges.pkl"

    with vocab_path.open("wb") as file:
        pickle.dump(vocab, file, protocol=pickle.HIGHEST_PROTOCOL)

    with merges_path.open("wb") as file:
        pickle.dump(merges, file, protocol=pickle.HIGHEST_PROTOCOL)

    longest_token_id, longest_token = max(
        vocab.items(),
        key=lambda item: len(item[1]),
    )

    print("\nTraining finished")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Longest token id: {longest_token_id}")
    print(f"Longest token length: {len(longest_token)} bytes")
    print(f"Longest token bytes: {longest_token!r}")
    print(
        "Longest token decoded:",
        longest_token.decode("utf-8", errors="replace"),
    )
    print(f"Vocabulary saved to: {vocab_path}")
    print(f"Merges saved to: {merges_path}")


if __name__ == "__main__":
    main()
