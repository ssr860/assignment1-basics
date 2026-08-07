from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping


def _json_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_serializable(item) for item in value]

    return value


class ExperimentLogger:
    def __init__(
        self,
        run_dir: Path,
        config: Mapping[str, Any],
        resume: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self.config_path = run_dir / "config.json"
        self.metrics_path = run_dir / "metrics.jsonl"
        self.checkpoint_dir = run_dir / "checkpoints"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.config_path.exists() and not resume:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. "
                "Choose a new run name or use --resume-from."
            )

        if not self.config_path.exists():
            with self.config_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    _json_serializable(dict(config)),
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

        self.wall_time_offset = self._last_wall_time()
        self.start_time = time.perf_counter()

    def _last_wall_time(self) -> float:
        if not self.metrics_path.exists():
            return 0.0

        last_wall_time = 0.0

        with self.metrics_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)
                last_wall_time = max(
                    last_wall_time,
                    float(record.get("wall_time_seconds", 0.0)),
                )

        return last_wall_time

    @property
    def elapsed_seconds(self) -> float:
        return (
            self.wall_time_offset
            + time.perf_counter()
            - self.start_time
        )

    def log(
        self,
        step: int,
        split: str,
        loss: float,
        **metrics: Any,
    ) -> None:
        record = {
            "step": step,
            "wall_time_seconds": self.elapsed_seconds,
            "split": split,
            "loss": loss,
            **metrics,
        }

        with self.metrics_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    _json_serializable(record),
                    ensure_ascii=False,
                )
                + "\n"
            )