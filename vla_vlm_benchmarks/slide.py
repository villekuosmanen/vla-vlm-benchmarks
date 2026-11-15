from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import PROTOCOL_VERSION
from .plot import _FAMILY_COLORS
from .prompts import prompt_revision
from .registry import BENCHMARKS, MODELS
from .summarize import comparable, primary_metric

LOGO_DIR = Path(__file__).resolve().parent.parent / ".cache" / "logos"
# Logos taken from a maker's own site rather than its Hub avatar. pi.website sits
# behind a bot check, so its icon is stored instead of downloaded.
STORED_LOGO_DIR = Path(__file__).resolve().parent.parent / "assets" / "logos"
CREDIT = "@villekuosmanen"
# The X mark from Simple Icons (CC0): outline, then the cut-out, in its 24-unit box with y pointing down.
_X_MARK = (
    ((14.234, 10.162), (22.977, 0), (20.905, 0), (13.314, 8.824), (7.251, 0), (0.258, 0), (9.426, 13.343),
     (0.258, 24), (2.33, 24), (10.346, 14.682), (16.749, 24), (23.742, 24)),
    ((11.397, 13.461), (10.468, 12.132), (3.076, 1.56), (6.258, 1.56), (12.223, 10.092), (13.152, 11.421),
     (20.906, 22.511), (17.724, 22.511)),
)


@dataclass(frozen=True)
class SlideEntry:
    label: str
    model_key: str
    released: str  # first public paper or weights, YYYY-MM-DD
    logo_org: str  # Hub organization of the maker; its avatar is the logo unless one is stored
    baseline_label: str
    checkpoint: str | None = None  # the registry checkpoint when None
    baseline_key: str | None = None  # the registry's matched baseline when None; may differ in layer count


ENTRIES = (
    SlideEntry("OpenVLA", "openvla", "2024-06-10", "openvla", "Prismatic 7B"),
    SlideEntry("π₀", "pi0", "2024-10-31", "physical-intelligence", "PaliGemma 3B pt"),
    SlideEntry("π₀.₅", "pi05", "2025-04-22", "physical-intelligence", "PaliGemma 3B pt"),
    SlideEntry("WALL-OSS flow", "wall_oss", "2025-09-06", "x-square-robot", "Qwen2.5-VL-3B", "x-square-robot/wall-oss-flow"),
    SlideEntry("GR00T N1.7", "groot_n17", "2026-02-25", "nvidia", "Cosmos-Reason2-2B*", baseline_key="cosmos_reason2_2b"),
    SlideEntry("MolmoAct2", "molmoact2", "2026-05-04", "allenai", "Molmo2-ER"),
    SlideEntry("WALL-OSS 0.5", "wall_oss", "2026-05-28", "x-square-robot", "Qwen2.5-VL-3B", "x-square-robot/wall-oss-0.5"),
    SlideEntry("G0.5", "g05", "2026-06-16", "OpenGalaxea", "Qwen3.5-2B-Base"),
)


@dataclass
class Row:
    entry: SlideEntry
    vla: float | None
    vlm: float | None
    problems: list[str]


def load_all(results_dir: Path) -> dict[tuple[str, str, str], dict]:
    """The newest current-protocol result per (model, checkpoint, benchmark), superseded runs included.

    A replaced checkpoint (WALL-OSS flow after 0.5) stays reachable this way.
    """

    from .results_io import read_json

    found: dict[tuple[str, str, str], dict] = {}
    for path in [*results_dir.glob("*.json"), *results_dir.glob("superseded/*/*.json")]:
        if path.name == "summary.json" or path.name.endswith((".partial.json", ".error.json")):
            continue
        payload = read_json(path)
        key = payload.get("model_key")
        if (
            key not in MODELS
            or payload.get("protocol_version") != PROTOCOL_VERSION
            or not payload.get("metrics")
            or payload.get("prompt_revision", 1) != prompt_revision(MODELS[key].prompt_style)
        ):
            continue
        slot = (key, payload.get("checkpoint"), payload.get("benchmark_key"))
        if slot not in found or payload.get("finished_at", "") > found[slot].get("finished_at", ""):
            found[slot] = payload
    return found


def _score(result: dict) -> float | None:
    return result["metrics"].get(primary_metric(result))


def score_row(found: dict, entry: SlideEntry) -> Row:
    """Mean primary score of the VLA and its matched VLM over every benchmark, or what is missing."""

    spec = MODELS[entry.model_key]
    checkpoint = entry.checkpoint or spec.checkpoint
    baseline = MODELS[entry.baseline_key or spec.matched_baseline_key]
    ignore = ("text_layers",) if entry.baseline_key else ()
    vla_scores, vlm_scores, problems = [], [], []
    for benchmark in BENCHMARKS:
        vla = found.get((spec.key, checkpoint, benchmark))
        vlm = found.get((baseline.key, baseline.checkpoint, benchmark))
        if vla is None or vlm is None:
            problems.append(f"{benchmark}: no {'VLA' if vla is None else 'VLM'} result")
            continue
        reason = comparable(vla, vlm, ignore)
        if reason is None and (_score(vla) is None or _score(vlm) is None):
            reason = "score missing"
        if reason:
            problems.append(f"{benchmark}: {reason}")
            continue
        vla_scores.append(_score(vla))
        vlm_scores.append(_score(vlm))
    if problems:
        return Row(entry, None, None, problems)
    return Row(entry, sum(vla_scores) / len(vla_scores), sum(vlm_scores) / len(vlm_scores), [])


def model_average(found: dict, model_key: str) -> float | None:
    spec = MODELS[model_key]
    scores = [found.get((model_key, spec.checkpoint, benchmark)) for benchmark in BENCHMARKS]
    if any(result is None or _score(result) is None for result in scores):
        return None
    return sum(_score(result) for result in scores) / len(scores)


def guess_level(found: dict) -> float | None:
    """Average of a uniform random guess: chance on multiple choice, zero on open answers."""

    levels = []
    for benchmark, spec in BENCHMARKS.items():
        if not spec.multiple_choice:
            levels.append(0.0)
            continue
        chances = [result["metrics"]["chance"] for (_, _, key), result in found.items()
                   if key == benchmark and result["metrics"].get("chance") is not None]
        if not chances:
            return None
        levels.append(sum(chances) / len(chances))
    return sum(levels) / len(levels)


def logo(org: str):
    """The maker's logo as a rounded RGBA image, None when unavailable.

    A file in ``assets/logos/`` wins; otherwise the Hugging Face avatar, cached.
    """

    from PIL import Image, ImageDraw

    path = STORED_LOGO_DIR / f"{org}.png"
    if not path.exists():
        path = LOGO_DIR / f"{org}.png"
    if not path.exists():
        try:
            for kind in ("organizations", "users"):
                try:
                    with urllib.request.urlopen(f"https://huggingface.co/api/{kind}/{org}/avatar", timeout=20) as reply:
                        url = json.load(reply)["avatarUrl"]
                    break
                except Exception:
                    url = None
            if url is None or url.endswith(".svg"):
                return None
            LOGO_DIR.mkdir(parents=True, exist_ok=True)
            download = path.with_suffix(".download")
            urllib.request.urlretrieve(url, download)
            Image.open(download).convert("RGBA").save(path)
            download.unlink()
        except Exception as exc:
            print(f"no logo for {org}: {exc}")
            return None
    image = Image.open(path).convert("RGBA").resize((256, 256), Image.LANCZOS)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 255, 255), radius=48, fill=255)
    image.putalpha(Image.composite(image.getchannel("A"), mask, mask))
    return image


def _tint(color: str, amount: float) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    red, green, blue = to_rgb(color)
    return (red + (1 - red) * amount, green + (1 - green) * amount, blue + (1 - blue) * amount)


def _points(delta: float) -> str:
    sign = "+" if delta > 0 else "−" if delta < 0 else "±"
    return f"{sign}{abs(delta):.1f} pts"


def draw(rows: list[Row], output: Path, guess: float | None, cosmos_cut: float | None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import Patch

    plt.rcParams["font.family"] = ["Noto Sans", "DejaVu Sans"]
    figure = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    figure.text(0.045, 0.925, "Most VLAs lose the vision-language skills of their VLM", fontsize=25, weight="bold", color="#1a1a1a")
    figure.text(
        0.045, 0.875,
        "Average score over 8 vision-language benchmarks, each VLA against its original VLM, in release order",
        fontsize=13, color="#555555",
    )
    axis = figure.add_axes((0.065, 0.315, 0.91, 0.5))
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#bbbbbb")
    axis.set_ylim(0, 100)
    axis.set_xlim(-0.6, len(rows) + 0.25)
    axis.set_yticks(range(0, 101, 20))
    axis.set_yticklabels([f"{tick}%" for tick in range(0, 101, 20)], fontsize=11, color="#666666")
    axis.tick_params(axis="y", length=0)
    axis.set_xticks([])
    axis.grid(axis="y", color="#eeeeee", linewidth=1)
    axis.set_axisbelow(True)

    text_height = 4.6
    for position, row in enumerate(rows):
        entry = row.entry
        color = _FAMILY_COLORS.get(MODELS[entry.model_key].family, "#7f7f7f")
        if row.vla is None:
            axis.text(position, 4, "results\npending", ha="center", va="bottom", fontsize=11, color="#999999")
        else:
            vla, vlm = 100 * row.vla, 100 * row.vlm
            axis.bar(position, vlm, width=0.8, color=_tint(color, 0.72), edgecolor=_tint(color, 0.35), linewidth=1, zorder=2)
            axis.bar(position, vla, width=0.44, color=color, zorder=3)
            if vla >= 9:
                axis.text(position, vla - 1.2, f"{vla:.1f}%", ha="center", va="top", fontsize=12, weight="bold", color="white", zorder=4)
                top = vla
            else:
                axis.text(position, vla + 0.8, f"{vla:.1f}%", ha="center", va="bottom", fontsize=12, weight="bold", color="#1a1a1a", zorder=4)
                top = vla + 0.8 + text_height
            label_y = max(vlm, top) + 1.0
            axis.text(position, label_y, f"VLM {vlm:.1f}%", ha="center", va="bottom", fontsize=10.5, color="#666666", zorder=4)
            delta = vla - vlm
            tone = "#1e8449" if delta > 1 else "#c0392b" if delta < -1 else "#666666"
            axis.text(
                position, label_y + text_height + 1.6, _points(delta), ha="center", va="bottom", fontsize=11.5,
                weight="bold", color=tone, zorder=4,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": _tint(tone, 0.85), "edgecolor": "none"},
            )
        image = logo(entry.logo_org)
        if image is not None:
            axis.add_artist(AnnotationBbox(
                OffsetImage(image, zoom=0.5 * 72 / 256), (position, 0), xycoords=("data", "axes fraction"),
                xybox=(0, -7), boxcoords="offset points", box_alignment=(0.5, 1.0), frameon=False,
            ))
        released = f"{['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][int(entry.released[5:7]) - 1]} {entry.released[:4]}"
        for text, offset, style in (
            (entry.label, -48, {"fontsize": 14, "weight": "bold", "color": "#1a1a1a"}),
            (released, -68, {"fontsize": 11, "color": "#666666"}),
            (f"on {entry.baseline_label}", -84, {"fontsize": 9.5, "color": "#888888", "style": "italic", "linespacing": 1.15}),
        ):
            axis.annotate(text, (position, 0), xycoords=("data", "axes fraction"), xytext=(0, offset),
                          textcoords="offset points", ha="center", va="top", annotation_clip=False, **style)

    if guess is not None:
        axis.axhline(100 * guess, color="#999999", linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
        axis.text(len(rows) + 0.22, 100 * guess + 0.8, f"random\nguessing\n{100 * guess:.0f}%", ha="right", va="bottom",
                  fontsize=9.5, color="#888888", linespacing=1.1)
    figure.legend(
        handles=[
            Patch(facecolor="#d9d9d9", edgecolor="#a6a6a6", label="original VLM it was built from"),
            Patch(facecolor="#595959", label="the same VLM inside the VLA"),
        ],
        loc="upper right", bbox_to_anchor=(0.97, 0.965), frameon=False, fontsize=11.5, handlelength=1.6,
    )
    cosmos_note = f"; the same 16-layer cut of the original Cosmos weights averages {100 * cosmos_cut:.1f}%" if cosmos_cut is not None else ""
    notes = [
        "Mean of each benchmark's primary score: POPE F1; VQAv2 and TextVQA VQA accuracy; RealWorldQA, AI2D and ERQA accuracy; "
        "ARC-Easy 0-shot accuracy; GSM8K 8-shot exact match.",
        f"Same prompts and greedy decoding for each VLA and its VLM (protocol v{PROTOCOL_VERSION}). Random guessing: chance on "
        "multiple choice, zero on open answers. Release dates: first public paper or weights.",
        f"*GR00T N1.7 keeps only the first 16 of Cosmos-Reason2-2B's 28 layers and no text head, so it cannot answer in text{cosmos_note}.",
    ]
    figure.text(0.045, 0.03, "\n".join(notes), fontsize=8.6, color="#777777", va="bottom", linespacing=1.45)
    _credit(figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    return output


def _credit(figure, color: str = "#9a9a9a") -> None:
    """The X mark and handle, small and grey, in the bottom-right corner."""

    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MarkPath
    from matplotlib.transforms import Affine2D

    handle = figure.text(0.972, 0.036, CREDIT, ha="right", va="baseline", fontsize=10, color=color)
    box = handle.get_window_extent(figure.canvas.get_renderer())
    size = 0.105  # inches, about the cap height of the handle
    scale = size / 24
    left = box.x0 / figure.dpi - 0.05 - size
    baseline = 0.036 * figure.get_figheight()
    vertices, codes = [], []
    for outline in _X_MARK:
        vertices += [*outline, outline[0]]
        codes += [MarkPath.MOVETO] + [MarkPath.LINETO] * (len(outline) - 1) + [MarkPath.CLOSEPOLY]
    place = Affine2D().scale(scale, -scale).translate(left, baseline + size) + figure.dpi_scale_trans
    figure.add_artist(PathPatch(MarkPath(vertices, codes), transform=place, facecolor=color, edgecolor="none"))


def main(argv: list[str] | None = None) -> int:
    from .runner import RESULTS_DIR

    parser = argparse.ArgumentParser(description="One slide: each VLA's average score over its original VLM's, in release order.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    found = load_all(args.results_dir)
    rows = [score_row(found, entry) for entry in sorted(ENTRIES, key=lambda entry: entry.released)]
    for row in rows:
        if row.vla is None:
            print(f"{row.entry.label}: pending ({'; '.join(row.problems)})")
        else:
            print(f"{row.entry.label}: VLA {100 * row.vla:.1f}%  VLM {100 * row.vlm:.1f}%  {_points(100 * (row.vla - row.vlm))}")
    output = draw(rows, args.output or args.results_dir / "slide_vla_average.png", guess_level(found),
                  model_average(found, "cosmos_reason2_2b_layer16"))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
