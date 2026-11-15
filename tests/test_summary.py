from vla_vlm_benchmarks.results_io import read_json, supersede, write_json
from vla_vlm_benchmarks.runner import main
from vla_vlm_benchmarks.summarize import build_summary


def _result(model, role, score, layers=18, baseline="paligemma_3b_pt224"):
    return {
        "protocol_version": 1,
        "model_key": model,
        "benchmark_key": "pope",
        "role": role,
        "family": "paligemma",
        "prompt_style": "paligemma_pt",
        "seed": 0,
        "limit": 20,
        "text_layers": layers,
        "dataset": "lmms-lab/POPE",
        "split": "test",
        "primary_metric": "f1",
        "multiple_choice": True,
        "matched_baseline_key": None if role == "baseline" else baseline,
        "metrics": {"f1": score, "chance": 0.5, "n": 20},
        "examples": [{"id": "1"}, {"id": "2"}],
    }


def test_summary_deltas_only_when_the_protocol_matches():
    summary = build_summary(
        [
            _result("paligemma_3b_pt224", "baseline", 0.8),
            _result("pi05", "vla", 0.5),
            _result("groot_n17", "vla", 0.4, layers=16, baseline="cosmos_reason2_2b_layer16"),
        ]
    )
    rows = {row["model_key"]: row for row in summary["rows"]}
    assert rows["pi05"]["comparable"] is True
    assert abs(rows["pi05"]["delta"] - (0.5 - 0.8)) < 1e-9
    assert rows["pi05"]["chance"] == 0.5
    assert rows["groot_n17"]["comparable"] is False
    assert rows["groot_n17"]["delta"] is None
    assert "paligemma_3b_pt224" not in rows


def test_summary_reports_the_secondary_score_beside_the_primary():
    def vqa(model, role, official, contains):
        result = _result(model, role, official)
        result.update(benchmark_key="textvqa", primary_metric="vqa_accuracy", multiple_choice=False)
        result["metrics"] = {"vqa_accuracy": official, "vqa_accuracy_contains": contains, "n": 20}
        return result

    summary = build_summary([vqa("paligemma_3b_pt224", "baseline", 0.8, 0.82), vqa("pi05", "vla", 0.5, 0.6)])
    row = summary["rows"][0]
    assert abs(row["delta"] - (0.5 - 0.8)) < 1e-9
    assert row["secondary"]["metric"] == "vqa_accuracy_contains"
    assert abs(row["secondary"]["delta"] - (0.6 - 0.82)) < 1e-9


def test_summary_refuses_a_delta_across_prompt_revisions():
    revised = _result("pi0", "vla", 0.5)
    revised["prompt_revision"] = 2
    summary = build_summary([_result("paligemma_3b_pt224", "baseline", 0.8), revised])
    row = summary["rows"][0]
    assert row["comparable"] is False
    assert "prompt_revision" in row["reason"]


def test_a_rerun_moves_the_old_result_aside(tmp_path):
    old = tmp_path / "pi0__pope.json"
    write_json(old, {**_result("pi0", "vla", 0.5), "finished_at": "2026-11-15T10:00:00+00:00"})
    moved = supersede(old, tmp_path)
    assert not old.exists()
    assert moved == tmp_path / "superseded" / "2026-11-15T10-00-00+00-00" / "pi0__pope.json"
    assert read_json(moved)["metrics"]["f1"] == 0.5
    assert supersede(old, tmp_path) is None


def test_cli_lists_and_plans_without_loading_weights(capsys):
    assert main(["--list"]) == 0
    listed = capsys.readouterr().out
    assert "google/paligemma-3b-pt-224" in listed
    assert "lerobot/pi0fast-base" in listed
    assert "Qwen/Qwen3.5-2B-Base" in listed

    assert main(["--dry-run", "--model", "pi0_fast", "--benchmark", "pope"]) == 0
    plan = capsys.readouterr().out
    assert "lerobot/pi0fast-base" in plan
    assert "lmms-lab/POPE" in plan

    assert main(["--model", "not-a-model"]) == 2
    assert main(["--checkpoint", "local", "--suite", "vlas"]) == 2
