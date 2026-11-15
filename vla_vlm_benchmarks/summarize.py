from __future__ import annotations

from pathlib import Path


def _ids(result: dict) -> list[str]:
    return [str(example["id"]) for example in result.get("examples", [])]


def primary_metric(result: dict) -> str | None:
    """The benchmark's current primary metric when the result stores it, else the one it recorded.

    Every metric is stored, so changing a benchmark's primary metric
    re-reads finished results instead of rerunning them.
    """

    from .registry import BENCHMARKS

    spec = BENCHMARKS.get(result.get("benchmark_key"))
    if spec is not None and spec.primary_metric in (result.get("metrics") or {}):
        return spec.primary_metric
    return result.get("primary_metric")


def comparable(model: dict, baseline: dict, ignore: tuple[str, ...] = ()) -> str | None:
    """None when a delta is meaningful. Otherwise a short reason. ``ignore`` skips named field checks."""

    checks = (
        "protocol_version",
        "benchmark_key",
        "prompt_style",
        "seed",
        "limit",
        "text_layers",
        "dataset",
        "split",
    )
    for field in checks:
        if field in ignore:
            continue
        if model.get(field) != baseline.get(field):
            return f"{field} differs ({model.get(field)!r} vs {baseline.get(field)!r})"
    if primary_metric(model) != primary_metric(baseline):
        return f"primary_metric differs ({primary_metric(model)!r} vs {primary_metric(baseline)!r})"
    revisions = (model.get("prompt_revision", 1), baseline.get("prompt_revision", 1))
    if revisions[0] != revisions[1]:
        return f"prompt_revision differs ({revisions[0]!r} vs {revisions[1]!r})"
    if _ids(model) != _ids(baseline):
        return "example ids differ"
    return None


def _unparsed_rate(result: dict, metric: str | None) -> float | None:
    examples = result.get("examples") or []
    if metric == "exact_match_flexible" and examples:
        return sum(1 for example in examples if example.get("parsed_flexible") is None) / len(examples)
    return (result.get("metrics") or {}).get("unparsed_rate")


def build_summary(results: list[dict]) -> dict:
    from .registry import BENCHMARKS

    by_key: dict[tuple[str, str], dict] = {}
    failures = []
    for result in results:
        if result.get("error") and "metrics" not in result:
            failures.append(
                {
                    "model_key": result.get("model_key"),
                    "benchmark_key": result.get("benchmark_key"),
                    "error": result.get("error"),
                }
            )
            continue
        by_key[(result["model_key"], result["benchmark_key"])] = result

    rows = []
    for (model_key, benchmark_key), result in sorted(by_key.items()):
        if result.get("role") in {"baseline", "baseline_probe"}:
            continue
        baseline_name = result.get("matched_baseline_key")
        baseline = by_key.get((baseline_name, benchmark_key)) if baseline_name else None
        metric = primary_metric(result)
        metrics = result.get("metrics") or {}
        model_score = metrics.get(metric)
        row = {
            "benchmark_key": benchmark_key,
            "model_key": model_key,
            "family": result.get("family"),
            "matched_baseline_key": baseline_name,
            "primary_metric": metric,
            "model_score": model_score,
            "baseline_score": None,
            "delta": None,
            "chance": metrics.get("chance") if result.get("multiple_choice") else None,
            "unparsed_rate": _unparsed_rate(result, metric),
            "n": metrics.get("n"),
            "text_layers": result.get("text_layers"),
            "comparable": False,
            "reason": "baseline result missing" if baseline is None else None,
        }
        if baseline is not None:
            reason = comparable(result, baseline)
            baseline_score = (baseline.get("metrics") or {}).get(metric)
            row["baseline_score"] = baseline_score
            row["reason"] = reason
            row["comparable"] = reason is None and model_score is not None and baseline_score is not None
            if row["comparable"]:
                row["delta"] = model_score - baseline_score
        spec = BENCHMARKS.get(benchmark_key)
        secondary = spec.secondary_metric if spec is not None else None
        if secondary is not None and secondary in metrics:
            secondary_baseline = ((baseline or {}).get("metrics") or {}).get(secondary)
            row["secondary"] = {
                "metric": secondary,
                "model_score": metrics[secondary],
                "baseline_score": secondary_baseline,
                "delta": metrics[secondary] - secondary_baseline
                if row["comparable"] and secondary_baseline is not None
                else None,
            }
        rows.append(row)
    return {"rows": rows, "failures": failures, "n_results": len(by_key)}


def _cell(value, signed: bool = False) -> str:
    if value is None:
        return ""
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def format_summary(summary: dict) -> str:
    header = (
        f"{'benchmark':<14} {'model':<28} {'baseline':<28} "
        f"{'metric':<20} {'model':>8} {'base':>8} {'delta':>8} {'chance':>8} {'unparsed':>8} note"
    )
    lines = [header]
    for row in summary["rows"]:
        note = "" if row["comparable"] else (row["reason"] or "")
        lines.append(
            f"{row['benchmark_key']:<14} {row['model_key']:<28} "
            f"{(row['matched_baseline_key'] or ''):<28} {(row['primary_metric'] or ''):<20} "
            f"{_cell(row['model_score']):>8} {_cell(row['baseline_score']):>8} "
            f"{_cell(row['delta'], signed=True):>8} {_cell(row.get('chance')):>8} "
            f"{_cell(row.get('unparsed_rate')):>8} {note}"
        )
    secondary_rows = [row for row in summary["rows"] if row.get("secondary")]
    if secondary_rows:
        lines.append("")
        lines.append("secondary scores (VQA with the answer contained in the reply; GSM8K strict-match)")
        for row in secondary_rows:
            secondary = row["secondary"]
            lines.append(
                f"{row['benchmark_key']:<14} {row['model_key']:<28} "
                f"{(row['matched_baseline_key'] or ''):<28} {secondary['metric']:<20} "
                f"{_cell(secondary['model_score']):>8} {_cell(secondary['baseline_score']):>8} "
                f"{_cell(secondary['delta'], signed=True):>8}"
            )
    if summary["failures"]:
        lines.append("failures")
        for failure in summary["failures"]:
            lines.append(
                f"  {failure.get('model_key')} {failure.get('benchmark_key')}: {failure.get('error')}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .results_io import read_json, write_json
    from .runner import RESULTS_DIR

    parser = argparse.ArgumentParser(description="Compare saved VLA and baseline benchmark JSON.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    results = []
    for path in sorted(args.results_dir.glob("*.json")):
        if path.name in {"summary.json"} or path.name.endswith(".partial.json"):
            continue
        payload = read_json(path)
        if "model_key" not in payload or "benchmark_key" not in payload:
            continue
        results.append(payload)
    summary = build_summary(results)
    write_json(args.results_dir / "summary.json", summary)
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
