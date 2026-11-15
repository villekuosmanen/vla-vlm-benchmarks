from __future__ import annotations

import argparse
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import PROTOCOL_VERSION
from .errors import EvalSetupError
from .prompts import prompt_revision
from .registry import BENCHMARKS, MODELS, SUITES, THINK_BUDGET, env_mismatch, format_catalog, matched_baseline_key
from .results_io import error_path, partial_path, read_json, result_path, supersede, write_json
from .scoring import aggregate_metrics, score_generation, score_loglikelihood, split_reasoning, truncate_at

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score VLM backbones inside VLAs, and the original VLMs they were trained from."
    )
    parser.add_argument("--model", action="append", default=[], help="Model key. Repeatable, or comma-separated.")
    parser.add_argument("--benchmark", action="append", default=[], help="Benchmark key. Repeatable, or comma-separated.")
    parser.add_argument("--suite", choices=sorted(SUITES), help="baselines, vlas, or all.")
    parser.add_argument("--limit", type=int, default=None, help="Examples per benchmark. 0 keeps the full filtered split. Default: the benchmark's own limit.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default=None, help="Replace the checkpoint of the single --model.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun even when a finished JSON for this seed and limit exists.")
    args = parser.parse_args(argv)
    if args.list:
        print(format_catalog())
        return 0
    try:
        model_keys, benchmark_keys = _selection(args)
    except EvalSetupError as exc:
        print(exc)
        return 2
    if args.dry_run:
        print(_plan_text(model_keys, benchmark_keys, args.limit, args.seed, args.checkpoint))
        return 0
    import transformers

    from .loading import load_adapter

    logging.getLogger("transformers").setLevel(logging.ERROR)
    device = _device(args.device)
    if device == "cpu":
        print("running on cpu; a CUDA GPU is the practical way to finish this suite")
    failures = 0
    for model_key in model_keys:
        spec = MODELS[model_key]
        mismatch = env_mismatch(spec, transformers.__version__)
        if mismatch:
            print(f"skip {spec.key}: {mismatch}")
            continue
        if args.checkpoint:
            from dataclasses import replace

            spec = replace(spec, checkpoint=args.checkpoint)
        pending = []
        for benchmark_key in benchmark_keys:
            benchmark = BENCHMARKS[benchmark_key]
            limit = benchmark.default_limit if args.limit is None else args.limit
            destination = result_path(args.results_dir, spec.key, benchmark_key)
            if not args.force and _reusable(
                destination, spec.key, benchmark_key, spec.checkpoint, args.seed, limit,
                prompt_revision(spec.prompt_style),
            ):
                print(f"skip {spec.key} {benchmark_key} ({destination.name})")
                continue
            pending.append((benchmark, limit, destination))
        if not pending:
            continue
        try:
            print(f"loading {spec.key} from {spec.checkpoint} on {device}")
            adapter = load_adapter(spec, device)
        except Exception as exc:
            failures += len(pending)
            detail = "".join(traceback.format_exception(exc))
            print(f"failed to load {spec.key}: {exc}")
            for benchmark, limit, _destination in pending:
                _write_error(args.results_dir, spec, benchmark.key, args.seed, limit, detail)
            continue
        try:
            for benchmark, limit, destination in pending:
                print(f"running {spec.key} on {benchmark.key} (limit={limit or 'all'}, seed={args.seed})")
                try:
                    payload = _run_benchmark(adapter, benchmark, limit, args.seed, args.results_dir)
                    previous = supersede(destination, args.results_dir)
                    if previous:
                        print(f"  kept the previous result at {previous}")
                    write_json(destination, payload)
                    partial = partial_path(args.results_dir, spec.key, benchmark.key)
                    partial.unlink(missing_ok=True)
                    error_path(args.results_dir, spec.key, benchmark.key).unlink(missing_ok=True)
                    primary = payload["metrics"].get(payload["primary_metric"])
                    rendered = "n/a" if primary is None else f"{primary:.3f}"
                    print(f"  {payload['primary_metric']}={rendered} n={payload['metrics']['n']} -> {destination}")
                except Exception as exc:
                    failures += 1
                    detail = "".join(traceback.format_exception(exc))
                    print(f"  failed {spec.key} {benchmark.key}: {exc}")
                    _write_error(args.results_dir, spec, benchmark.key, args.seed, limit, detail)
        finally:
            adapter.close()
    return 1 if failures else 0


def _selection(args) -> tuple[list[str], list[str]]:
    model_keys = _expand(args.model)
    benchmark_keys = _expand(args.benchmark)
    if not model_keys:
        if not args.suite:
            raise EvalSetupError("pass --model or --suite")
        model_keys = list(SUITES[args.suite])
    if not benchmark_keys:
        benchmark_keys = list(BENCHMARKS)
    unknown_models = [key for key in model_keys if key not in MODELS]
    unknown_benchmarks = [key for key in benchmark_keys if key not in BENCHMARKS]
    if unknown_models or unknown_benchmarks:
        raise EvalSetupError(
            f"unknown model {unknown_models} or benchmark {unknown_benchmarks}; try --list"
        )
    if args.checkpoint and len(model_keys) != 1:
        raise EvalSetupError("--checkpoint replaces the checkpoint of exactly one --model")
    return model_keys, benchmark_keys


def _expand(values: list[str]) -> list[str]:
    keys = []
    for value in values:
        keys.extend(part.strip() for part in value.split(",") if part.strip())
    return keys


def _plan_text(model_keys, benchmark_keys, limit, seed, checkpoint) -> str:
    lines = [f"seed={seed}"]
    for model_key in model_keys:
        spec = MODELS[model_key]
        loaded = checkpoint or spec.checkpoint
        lines.append(f"{spec.key} [{spec.role}] {loaded} prompt={spec.prompt_style}")
    for benchmark_key in benchmark_keys:
        spec = BENCHMARKS[benchmark_key]
        chosen = spec.default_limit if limit is None else limit
        lines.append(
            f"{spec.key} {spec.dataset} limit={chosen or 'all'} primary={spec.primary_metric}"
        )
    return "\n".join(lines)


def _device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _reusable(
    path: Path, model_key: str, benchmark_key: str, checkpoint: str, seed: int, limit: int, revision: int
) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    return bool(
        payload.get("protocol_version") == PROTOCOL_VERSION
        and payload.get("model_key") == model_key
        and payload.get("benchmark_key") == benchmark_key
        and payload.get("checkpoint") == checkpoint
        and payload.get("seed") == seed
        and payload.get("limit") == limit
        and payload.get("prompt_revision", 1) == revision
        and payload.get("finished_at")
        and payload.get("metrics")
        and not (payload.get("metrics") or {}).get("primary_missing")
        and (payload.get("metrics") or {}).get(payload.get("primary_metric")) is not None
    )


def _run_benchmark(adapter, benchmark, limit: int, seed: int, results_dir: Path) -> dict:
    from .data_load import load_examples

    spec = adapter.spec
    started = _now()
    examples, meta = load_examples(benchmark, limit, seed)
    records = []
    for index, example in enumerate(examples, start=1):
        records.append(_score_example(adapter, benchmark, example))
        if index % 10 == 0 or index == len(examples):
            write_json(
                partial_path(results_dir, spec.key, benchmark.key),
                {"model_key": spec.key, "benchmark_key": benchmark.key, "n_finished": index, "examples": records},
            )
            print(f"  {index}/{len(examples)}", flush=True)
    metrics = aggregate_metrics(records, benchmark)
    if metrics["n_errors"] == len(records):
        raise EvalSetupError(f"every example failed for {spec.key} on {benchmark.key}: {records[0]['error']}")
    if benchmark.has_images:
        image_policy = "benchmark"
    elif spec.requires_image:
        image_policy = "blank_white"
    else:
        image_policy = "none"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": spec.key,
        "benchmark_key": benchmark.key,
        "role": spec.role,
        "family": spec.family,
        "name": spec.name,
        "checkpoint": spec.checkpoint,
        "revisions": adapter.revisions,
        "baseline_key": spec.baseline_key,
        "matched_baseline_key": matched_baseline_key(spec),
        "prompt_style": spec.prompt_style,
        "prompt_revision": prompt_revision(spec.prompt_style),
        "image_policy": image_policy,
        "license_note": spec.license_note,
        "notes": spec.notes,
        "ancestor_checkpoint": spec.ancestor_checkpoint,
        "seed": seed,
        "limit": limit,
        "text_layers": adapter.text_layers,
        "dataset": meta["dataset"],
        "dataset_config": meta["dataset_config"],
        "split": meta["split"],
        "dataset_revision": meta["dataset_revision"],
        "dataset_meta": meta,
        "primary_metric": benchmark.primary_metric,
        "scoring": benchmark.scoring,
        "convention": benchmark.convention,
        "multiple_choice": benchmark.multiple_choice,
        "max_new_tokens": benchmark.max_new_tokens,
        "think_budget": THINK_BUDGET if benchmark.scoring != "loglikelihood" else 0,
        "stop": list(benchmark.stop),
        "weight_coverage": adapter.coverage,
        "warnings": list(adapter.warnings) + list(meta["warnings"]),
        "metrics": metrics,
        "started_at": started,
        "finished_at": _now(),
        "examples": records,
    }


def _score_example(adapter, benchmark, example) -> dict:
    """One example under the benchmark's protocol. Failures score 0 and are counted, never dropped."""

    record = {
        "id": example.id,
        "gold": example.gold,
        "group": example.group,
        "chance": example.chance,
        "prediction": None,
        "parsed": None,
        "parsed_ok": False,
        "score": 0.0,
        "error": None,
    }
    if example.scoring == "loglikelihood":
        record.update(correct=False, correct_norm=False)
    if example.scoring == "gsm8k":
        record["score_flexible"] = 0.0
    if example.scoring == "vqa":
        record["score_contains"] = 0.0
    try:
        if example.scoring == "loglikelihood":
            logprobs = adapter.continuation_logprobs(
                example.images, example.text, [" " + choice for choice in example.choices]
            )
            record.update(score_loglikelihood(example, logprobs))
        else:
            reply = adapter.generate_text(example.images, example.text, benchmark.max_new_tokens, stop=benchmark.stop)
            reasoning, prediction = split_reasoning(reply)
            if benchmark.stop:
                prediction = truncate_at(prediction, benchmark.stop)
            if reasoning is not None:
                record["reasoning"] = reasoning[:4000]
            record["prediction"] = prediction[:4000]
            record.update(score_generation(example, prediction))
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _write_error(directory: Path, spec, benchmark_key: str, seed: int, limit: int, detail: str) -> None:
    write_json(
        error_path(directory, spec.key, benchmark_key),
        {
            "protocol_version": PROTOCOL_VERSION,
            "model_key": spec.key,
            "benchmark_key": benchmark_key,
            "checkpoint": spec.checkpoint,
            "seed": seed,
            "limit": limit,
            "error": detail[-8000:],
            "finished_at": _now(),
        },
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
