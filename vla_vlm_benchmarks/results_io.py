from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def result_path(directory: Path, model_key: str, benchmark_key: str) -> Path:
    return directory / f"{model_key}__{benchmark_key}.json"


def partial_path(directory: Path, model_key: str, benchmark_key: str) -> Path:
    return directory / f"{model_key}__{benchmark_key}.partial.json"


def error_path(directory: Path, model_key: str, benchmark_key: str) -> Path:
    return directory / f"{model_key}__{benchmark_key}.error.json"


def supersede(path: Path, results_dir: Path) -> Path | None:
    """Move a finished result out of the summarized set instead of overwriting it."""

    if not path.exists():
        return None
    try:
        finished = read_json(path).get("finished_at") or "unknown"
    except ValueError:
        finished = "unreadable"
    destination = results_dir / "superseded" / finished.replace(":", "-") / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.replace(destination)
    return destination
