from __future__ import annotations

import argparse
from pathlib import Path

from . import PROTOCOL_VERSION
from .registry import BENCHMARKS, MODELS
from .summarize import primary_metric

_SHORT = {
    "paligemma_3b_pt224": "PaliGemma pt",
    "pi0": "π₀",
    "pi05": "π₀.₅",
    "pi0_fast": "π₀-FAST",
    "qwen3_5_2b_base": "Qwen3.5-2B",
    "g05": "G0.5",
    "cosmos_reason2_2b": "Cosmos-R2",
    "cosmos_reason2_2b_layer16": "Cosmos-R2 L16",
    "groot_n17": "GR00T N1.7",
    "qwen2_5_vl_3b_instruct": "Qwen2.5-VL-3B",
    "wall_oss": "WALL-OSS",
    "molmo2_er": "Molmo2-ER",
    "molmoact2": "MolmoAct2",
    "prism_dinosiglip_224px_7b": "Prismatic 7B",
    "openvla": "OpenVLA",
}
_FAMILY_COLORS = {
    "paligemma": "#1f77b4",
    "qwen3_5": "#ff7f0e",
    "cosmos": "#2ca02c",
    "qwen2_5_vl": "#d62728",
    "molmo": "#9467bd",
    "prismatic": "#8c564b",
}
_METRIC_LABELS = {
    "f1": "F1 (mean of 3 splits)",
    "vqa_accuracy": "VQA accuracy",
    "accuracy": "accuracy",
    "acc": "accuracy (0-shot cloze)",
    "exact_match_strict": "exact match (8-shot CoT, strict)",
    "exact_match_flexible": "exact match (8-shot CoT, last number)",
}


def load_results(results_dir: Path) -> dict[tuple[str, str], dict]:
    from .results_io import read_json

    found = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "summary.json" or path.name.endswith((".partial.json", ".error.json")):
            continue
        payload = read_json(path)
        if payload.get("protocol_version") != PROTOCOL_VERSION or "metrics" not in payload:
            continue
        found[(payload["model_key"], payload["benchmark_key"])] = payload
    return found


def model_order() -> list[str]:
    """Models grouped by family, baselines first, in registry order."""

    families: dict[str, list[str]] = {}
    for key, spec in MODELS.items():
        families.setdefault(spec.family, []).append(key)
    ordered = []
    for keys in families.values():
        ordered.extend(sorted(keys, key=lambda key: MODELS[key].role == "vla"))
    return ordered


def draw(results: dict[tuple[str, str], dict], output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    benchmarks = [key for key in BENCHMARKS if any(bench == key for _, bench in results)]
    models = [key for key in model_order() if any(model == key for model, _ in results)]
    columns = 4
    rows = (len(benchmarks) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.4 * rows), squeeze=False)
    positions = []
    x = 0.0
    previous_family = None
    for key in models:
        family = MODELS[key].family
        if previous_family is not None and family != previous_family:
            x += 0.6
        positions.append(x)
        x += 1.0
        previous_family = family

    for axis, benchmark_key in zip(axes.flat, benchmarks):
        spec = BENCHMARKS[benchmark_key]
        chances = []
        always_yes = []
        for position, model_key in zip(positions, models):
            result = results.get((model_key, benchmark_key))
            model = MODELS[model_key]
            vla = model.role == "vla"
            color = _FAMILY_COLORS.get(model.family, "#7f7f7f")
            if result is None:
                axis.text(position, 1, "n/a", ha="center", va="bottom", fontsize=7, rotation=90, color="#999999")
                continue
            metrics = result["metrics"]
            score = metrics.get(primary_metric(result))
            if metrics.get("chance") is not None:
                chances.append(metrics["chance"])
            if metrics.get("always_yes_f1") is not None:
                always_yes.append(metrics["always_yes_f1"])
            if score is None:
                continue
            axis.bar(
                position,
                100 * score,
                width=0.8,
                color=color if vla else "white",
                edgecolor=color,
                hatch=None if vla else "///",
                linewidth=1.2,
            )
            axis.text(position, 100 * score + 1, f"{100 * score:.0f}", ha="center", va="bottom", fontsize=7)
        label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1}
        right = positions[-1] + 0.6
        if spec.multiple_choice and chances:
            chance = sum(chances) / len(chances)
            axis.axhline(100 * chance, linestyle="--", color="black", linewidth=1, zorder=3)
            axis.text(right, 100 * chance, f"chance {100 * chance:.0f}%", ha="left", va="center", fontsize=8, bbox=label_box, clip_on=False)
        if always_yes:
            level = sum(always_yes) / len(always_yes)
            axis.axhline(100 * level, linestyle=":", color="#555555", linewidth=1, zorder=3)
            axis.text(right, 100 * level, f"always yes {100 * level:.0f}%", ha="left", va="center", fontsize=8, color="#555555", bbox=label_box, clip_on=False)
        n_values = {result["metrics"]["n"] for (model, bench), result in results.items() if bench == benchmark_key}
        n_label = "/".join(str(n) for n in sorted(n_values))
        axis.set_title(f"{spec.title} (n={n_label})\n{_METRIC_LABELS.get(spec.primary_metric, spec.primary_metric)}", fontsize=10)
        axis.set_ylim(0, 105)
        axis.set_xticks(positions)
        axis.set_xticklabels([_SHORT.get(key, key) for key in models], rotation=60, ha="right", fontsize=8)
        axis.set_ylabel("%")
        axis.spines[["top", "right"]].set_visible(False)
    for axis in list(axes.flat)[len(benchmarks):]:
        axis.axis("off")

    legend = [
        Patch(facecolor="white", edgecolor="#555555", hatch="///", label="original VLM (baseline)"),
        Patch(facecolor="#555555", edgecolor="#555555", label="VLM inside the VLA"),
    ]
    figure.tight_layout(rect=(0, 0, 1, 0.92), w_pad=4.0)
    figure.suptitle(f"VLA backbones vs their original VLMs (protocol v{PROTOCOL_VERSION})", y=0.995, fontsize=13)
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False, fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv: list[str] | None = None) -> int:
    from .runner import RESULTS_DIR

    parser = argparse.ArgumentParser(description="Bar charts of the saved results, with chance lines on multiple-choice benchmarks.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    results = load_results(args.results_dir)
    if not results:
        print(f"no protocol v{PROTOCOL_VERSION} results in {args.results_dir}")
        return 1
    output = draw(results, args.output or args.results_dir / "chart.png")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
