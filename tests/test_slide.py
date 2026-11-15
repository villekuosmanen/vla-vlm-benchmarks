from vla_vlm_benchmarks import PROTOCOL_VERSION
from vla_vlm_benchmarks.prompts import prompt_revision
from vla_vlm_benchmarks.registry import BENCHMARKS, MODELS
from vla_vlm_benchmarks.results_io import write_json
from vla_vlm_benchmarks.slide import ENTRIES, load_all, score_row


def _result(model, checkpoint, benchmark, score, finished_at="2026-11-15T10:00:00"):
    spec = BENCHMARKS[benchmark]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_revision": prompt_revision(MODELS[model].prompt_style),
        "model_key": model,
        "checkpoint": checkpoint,
        "benchmark_key": benchmark,
        "prompt_style": MODELS[model].prompt_style,
        "seed": 0,
        "limit": 20,
        "text_layers": None,
        "dataset": spec.dataset,
        "split": "test",
        "primary_metric": spec.primary_metric,
        "finished_at": finished_at,
        "metrics": {spec.primary_metric: score, "n": 20},
        "examples": [{"id": "1"}],
    }


def test_slide_keeps_a_replaced_checkpoint_and_waits_for_an_unfinished_one(tmp_path):
    flow, current = (next(entry for entry in ENTRIES if entry.label == label) for label in ("WALL-OSS flow", "WALL-OSS 0.5"))
    baseline = MODELS["qwen2_5_vl_3b_instruct"]
    archive = tmp_path / "superseded" / "2026-11-15T21-00-00"
    for benchmark in BENCHMARKS:
        write_json(tmp_path / f"qwen2_5_vl_3b_instruct__{benchmark}.json", _result(baseline.key, baseline.checkpoint, benchmark, 0.8))
        write_json(archive / f"wall_oss__{benchmark}.json", _result("wall_oss", flow.checkpoint, benchmark, 0.5))
    write_json(tmp_path / "wall_oss__pope.json", _result("wall_oss", current.checkpoint, "pope", 0.7, "2026-11-15T22:00:00"))

    found = load_all(tmp_path)
    flow_row, current_row = score_row(found, flow), score_row(found, current)
    assert abs(flow_row.vla - 0.5) < 1e-9 and abs(flow_row.vlm - 0.8) < 1e-9
    assert current_row.vla is None
    assert len(current_row.problems) == len(BENCHMARKS) - 1
