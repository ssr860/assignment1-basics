from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import torch

from cs336_basics.blocks import Transformer
from cs336_basics.get_tokenizer import Tokenizer
from cs336_basics.training_utilities import load_checkpoint


DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with a saved CS336 language model."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to the checkpoint with the largest step number.",
    )
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus-sampling probability threshold in (0, 1].",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="May be supplied multiple times; defaults to <|endoftext|>.",
    )
    parser.add_argument(
        "--stop-token-id",
        type=int,
        default=None,
        help="Stop after generating this token ID.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, or a specific device such as cuda:0.",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a JSON object in {config_path}")
    return config


def resolve_device(device_argument: str) -> torch.device:
    if device_argument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_argument)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("step_"))
    except ValueError:
        return -1


def resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    checkpoint_dir = run_dir / "checkpoints"
    candidates = list(checkpoint_dir.glob("step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No step_*.pt checkpoints found in {checkpoint_dir}")
    return max(candidates, key=checkpoint_step)


def build_model(
    config: dict[str, Any],
    device: torch.device,
) -> Transformer:
    dtype_name = str(config.get("dtype", "float32"))
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype in config.json: {dtype_name}")

    return Transformer(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=float(config["rope_theta"]),
        device=device,
        dtype=DTYPE_MAP[dtype_name],
    )


def load_serialized_object(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")

    if path.suffix.lower() in {".pkl", ".pickle"}:
        # Only load pickle files that you created or otherwise trust.
        with path.open("rb") as file:
            return pickle.load(file)

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    raise ValueError(
        f"Unsupported tokenizer file extension {path.suffix!r}; "
        "use .pkl, .pickle, or .json"
    )


def to_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return bytes(value)
    if isinstance(value, str):
        # JSON cannot represent bytes. This convention treats JSON strings as
        # UTF-8 text; pickle is preferable when arbitrary byte tokens are used.
        return value.encode("utf-8")
    raise TypeError(f"Cannot convert tokenizer value to bytes: {value!r}")


def load_tokenizer(
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
) -> Tokenizer:
    raw_vocab = load_serialized_object(vocab_path)
    raw_merges = load_serialized_object(merges_path)

    if not isinstance(raw_vocab, dict):
        raise TypeError("Vocabulary must be a dict mapping token IDs to bytes")
    vocab = {int(token_id): to_bytes(token) for token_id, token in raw_vocab.items()}

    if not isinstance(raw_merges, list):
        raise TypeError("Merges must be a list of byte-pair tuples")
    merges: list[tuple[bytes, bytes]] = []
    for pair in raw_merges:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise TypeError(f"Invalid merge pair: {pair!r}")
        merges.append((to_bytes(pair[0]), to_bytes(pair[1])))

    return Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
    )


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample one token from each row of logits using top-p sampling."""
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1]")

    # temperature == 0 conventionally means deterministic greedy decoding.
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    # Perform softmax and cumulative sums in float32 for numerical stability.
    probabilities = torch.softmax(
        logits.to(torch.float32) / temperature,
        dim=-1,
    )
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        dim=-1,
        descending=True,
    )
    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    # Keep every token before the threshold and also the token that crosses it.
    previous_cumulative_probabilities = (
        cumulative_probabilities - sorted_probabilities
    )
    tokens_to_keep = previous_cumulative_probabilities < top_p
    filtered_probabilities = sorted_probabilities.masked_fill(
        ~tokens_to_keep,
        0.0,
    )
    filtered_probabilities = filtered_probabilities / filtered_probabilities.sum(
        dim=-1,
        keepdim=True,
    )

    sampled_sorted_index = torch.multinomial(
        filtered_probabilities,
        num_samples=1,
    )
    return torch.gather(
        sorted_indices,
        dim=-1,
        index=sampled_sorted_index,
    )


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    prompt_token_ids: list[int],
    eos_token_id: int | None,
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_p: float,
    device: str | torch.device,
) -> list[int]:
    if not prompt_token_ids:
        raise ValueError("The prompt must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1]")

    device = torch.device(device)
    was_training = model.training
    model.eval()
    generated = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long,
        device=device,
    )

    try:
        for _ in range(max_new_tokens):
            model_input = generated[:, -context_length:]
            logits = model(model_input)
            next_token_logits = logits[:, -1, :]

            next_token = sample_next_token(
                logits=next_token_logits,
                temperature=temperature,
                top_p=top_p,
            )
            generated = torch.cat(
                (generated, next_token),
                dim=1,
            )

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    finally:
        model.train(was_training)

    return generated.squeeze(0).tolist()


def main() -> None:
    args = parse_args()
    config = load_config(args.run_dir)
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint(args.run_dir, args.checkpoint)
    special_tokens = args.special_token or ["<|endoftext|>"]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = load_tokenizer(
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        special_tokens=special_tokens,
    )
    model = build_model(config, device)
    iteration = load_checkpoint(
        src=checkpoint_path,
        model=model,
        optimizer=None,
    )

    prompt_token_ids = tokenizer.encode(args.prompt)
    generated_token_ids = generate(
        model=model,
        prompt_token_ids=prompt_token_ids,
        eos_token_id=args.stop_token_id,
        max_new_tokens=args.max_new_tokens,
        context_length=int(config["context_length"]),
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    generated_text = tokenizer.decode(generated_token_ids)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Step:       {iteration}")
    print("\nGenerated text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()