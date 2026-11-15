from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def rescore(path: Path) -> str | None:
    """Add ``vqa_accuracy_contains`` to a finished VQA result from its stored predictions.

    Scores already in the file are left as they are. Returns what was done, or
    None when the file needs nothing.
    """

    from .data_load import load_examples
    from .registry import BENCHMARKS
    from .results_io import read_json, write_json
    from .scoring import _mean, vqa_accuracy

    payload = read_json(path)
    spec = BENCHMARKS.get(payload.get("benchmark_key"))
    metrics = payload.get("metrics")
    if spec is None or spec.scoring != "vqa" or metrics is None or "vqa_accuracy_contains" in metrics:
        return None
    examples, _ = load_examples(spec, payload["limit"], payload["seed"])
    answers = {example.id: list(example.answers) for example in examples}
    records = payload["examples"]
    missing = [record["id"] for record in records if record["id"] not in answers]
    if missing:
        return f"skipped: {len(missing)} example ids not in the reloaded subset"
    for record in records:
        prediction = record.get("prediction")
        record["score_contains"] = vqa_accuracy(prediction, answers[record["id"]], contains=True) if prediction else 0.0
    metrics["vqa_accuracy_contains"] = _mean([record["score_contains"] for record in records if record.get("score") is not None])
    payload.setdefault("rescored", {})["vqa_accuracy_contains"] = datetime.now(timezone.utc).isoformat()
    if not path.exists() or read_json(path).get("finished_at") != payload.get("finished_at"):
        return "skipped: rewritten by a run while rescoring"
    write_json(path, payload)
    return f"vqa_accuracy_contains={metrics['vqa_accuracy_contains']:.3f}"


def main(argv: list[str] | None = None) -> int:
    from .runner import RESULTS_DIR

    parser = argparse.ArgumentParser(description="Add scores introduced after a result was written.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    for path in sorted(args.results_dir.glob("*.json")):
        if path.name == "summary.json" or path.name.endswith(".partial.json"):
            continue
        outcome = rescore(path)
        if outcome:
            print(f"{path.name}: {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
