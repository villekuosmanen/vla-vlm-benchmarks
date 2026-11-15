from __future__ import annotations

import argparse
import textwrap
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .registry import BENCHMARKS, MODELS
from .scoring import _STRICT_GSM8K
from .slide import ENTRIES, SlideEntry, _credit, load_all, logo

WIDTH, HEIGHT = 13.333, 7.5
LEFT, RIGHT = 0.6, 12.73
# OpenVLA's ActionTokenizer maps its 256 bins onto the last 256 Llama-2 tokens.
OPENVLA_ACTION_BINS = 256
# OpenVLA's action tokens decode to letters from a dozen scripts.
FONTS = ["Noto Sans", "Noto Sans CJK JP", "Noto Sans Devanagari", "Noto Sans Bengali", "Noto Sans Gurmukhi",
         "Noto Sans Tamil", "Noto Sans Telugu", "Noto Sans Kannada", "Noto Sans Malayalam", "Noto Sans Sinhala",
         "Noto Sans Thai", "Noto Sans Myanmar", "Noto Sans Khmer", "Noto Sans Syriac", "Noto Serif Tibetan",
         "Noto Sans Math", "DejaVu Sans"]
NUMBERS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
# A reply that opens with a right-to-left letter would otherwise lay out its whole line right to left.
LTR = "\u200e"


@dataclass
class Reply:
    entry: SlideEntry
    record: dict | None
    action_tokens: int = 0


@dataclass(frozen=True)
class CardText:
    reasoning: str | None
    answer: str | None  # None when the reply states no final answer
    quoted: bool = True
    option: str | None = None


def collect(found: dict, benchmark: str, example_id: str) -> tuple[list[Reply], dict | None]:
    """Each slide entry's stored record for one example, and the payload that holds it."""

    replies, source = [], None
    for entry in sorted(ENTRIES, key=lambda entry: entry.released):
        checkpoint = entry.checkpoint or MODELS[entry.model_key].checkpoint
        payload = found.get((entry.model_key, checkpoint, benchmark))
        record = None
        if payload is not None:
            record = next((example for example in payload["examples"] if example["id"] == example_id), None)
            source = source or payload
        replies.append(Reply(entry, record))
    return replies, source


def action_token_count(prediction: str) -> int:
    """How many action-bin tokens an OpenVLA reply is made of; zero unless it is nothing else."""

    from transformers import LlamaTokenizerFast

    tokenizer = LlamaTokenizerFast.from_pretrained(MODELS["openvla"].checkpoint)
    ids = tokenizer.encode(prediction, add_special_tokens=False)
    ids = [token for token in ids if tokenizer.convert_ids_to_tokens(token) != "▁"]
    first_bin = tokenizer.vocab_size - OPENVLA_ACTION_BINS
    return len(ids) if ids and all(token >= first_bin for token in ids) else 0


def _isolate_marks(text: str) -> str:
    # Each action token decodes to one character, so a combining mark among them has no base; left
    # bare, the renderer shapes it onto the neighbouring token and finds no font holding both.
    return "".join("◌" + char if unicodedata.category(char) == "Mn" else char for char in text)


def card_text(reply: Reply, example) -> CardText:
    """What a card prints: reasoning in grey, the answer, and the option a letter stands for."""

    record = reply.record
    prediction = record.get("prediction") or ""
    if reply.action_tokens:
        prediction = _isolate_marks(prediction)
    if example.scoring == "loglikelihood":
        return CardText(None, example.choices[int(record["parsed"])], quoted=False)
    if example.scoring == "gsm8k":
        stated = _STRICT_GSM8K.search(prediction)
        working = (prediction[: stated.start()] if stated else prediction).strip()
        return CardText(working or None, record.get("parsed"), quoted=False)
    parsed = record.get("parsed")
    option = None
    if example.letters and example.choices and parsed in example.letters:
        option = example.choices[example.letters.index(parsed)]
    return CardText(record.get("reasoning"), prediction, option=option)


def _clip(text: str, width: int, lines: int) -> str:
    wrapped = textwrap.wrap(" ".join(text.split()), width=width, break_long_words=True) or [""]
    if len(wrapped) > lines:
        wrapped = wrapped[:lines]
        wrapped[-1] = wrapped[-1][: width - 1].rstrip() + "…"
    return "\n".join(wrapped)


def _wrap(text: str, width_in: float, fontsize: float, mono: bool, lines: int) -> str:
    """Wrap each paragraph to a width in inches, from an average glyph width per font size."""

    width = max(int(width_in / ((0.6 if mono else 0.5) * fontsize / 72)), 10)
    return "\n".join(_clip(part, width, lines) for part in text.splitlines() if part.strip())


def _number(count: int) -> str:
    return NUMBERS[count] if count < len(NUMBERS) else str(count)


def _verdict(score: float | None) -> tuple[str, str]:
    if score is None:
        return "no result", "#888888"
    if score >= 0.999:
        return "✓ correct", "#1e8449"
    if score > 0:
        return "partly right", "#b9770e"
    return "✗ wrong", "#c0392b"


def _prompt_and_label(example) -> tuple[str, str]:
    if example.scoring == "gsm8k":
        return "Q: " + example.text.rsplit("\n\nQ: ", 1)[-1], "PROMPT  (after eight worked examples, lm-eval's chain-of-thought prompt)"
    if example.scoring == "loglikelihood":
        return example.text, "PROMPT  (each option is scored as its continuation)"
    return example.text, "PROMPT"


def _answer_key(example) -> str:
    if example.scoring == "vqa":
        ranked = Counter(example.answers).most_common()
        (top_answer, votes), others = ranked[0], [answer for answer, _ in ranked[1:]]
        answers = f"“{top_answer}” from {votes} of {len(example.answers)} annotators"
        if others:
            answers += "; also " + ", ".join(f"“{answer}”" for answer in others)
        return f"Human answers: {answers}"
    if example.scoring == "loglikelihood":
        options = ", ".join(f"“{choice}”" for choice in example.choices)
        return f"Options: {options}. Correct: “{example.choices[int(example.gold)]}”"
    if example.letters and example.choices:
        return f"Correct answer: {example.gold}. {example.choices[example.letters.index(example.gold)]}"
    return f"Correct answer: {example.gold}"


def _method_note(spec, example) -> str:
    if example.scoring == "vqa":
        method = f"Scored with {spec.title}'s official metric against the human answers."
    elif example.scoring == "pope":
        method = "Scored on the yes or no in the reply."
    elif example.scoring == "loglikelihood":
        method = "Scored by likelihood as in lm-eval: nothing is generated, and each card shows the option the model rates most likely."
    elif example.scoring == "gsm8k":
        method = "Scored as in lm-eval: only a number after “The answer is” counts as the final answer."
    elif example.letters:
        method = "Scored on the option letter in the reply."
    else:
        method = "Scored on the reply's wording, normalized."
    decoding = "" if example.scoring == "loglikelihood" else "; greedy decoding"
    return f"{method} Each VLA's backbone runs on its original VLM's architecture with that VLM's prompt{decoding}."


def draw(replies: list[Reply], example, benchmark: str, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import FancyBboxPatch

    from .slide import _tint

    plt.rcParams["font.family"] = FONTS
    mono = ["Noto Sans Mono", "Noto Sans Mono CJK JP", "DejaVu Sans Mono"]
    figure = plt.figure(figsize=(WIDTH, HEIGHT), facecolor="white")
    inches = figure.dpi_scale_trans
    renderer = figure.canvas.get_renderer()

    def text(x, y, value, **style):
        return figure.text(x / WIDTH, y / HEIGHT, value, **style)

    def place(artist, x, y):
        artist.set_position((x / WIDTH, y / HEIGHT))

    def tall(artist) -> float:
        return artist.get_window_extent(renderer).height / figure.dpi

    def box(x, y, w, h, **style):
        figure.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", transform=inches, **style))

    spec = BENCHMARKS[benchmark]
    subject = "One photo, one question" if example.images else "One question"
    text(LEFT, 6.93, f"{subject}, {_number(len(replies))} VLAs", fontsize=25, weight="bold", color="#1a1a1a")
    text(LEFT, 6.6, f"{spec.title} question {example.id}{'' if example.images else ', text only'}. The VLM inside each VLA, "
         "given the same prompt as its original VLM.", fontsize=13, color="#555555")

    top, left, header_bottom = 6.32, LEFT, 6.32
    if example.images:
        image = example.images[0]
        width = min(2.1 * image.width / image.height, 4.2)
        height = width * image.height / image.width
        axis = figure.add_axes((left / WIDTH, (top - height) / HEIGHT, width / WIDTH, height / HEIGHT))
        axis.imshow(image)
        axis.axis("off")
        header_bottom = top - height
        left += width + 0.35
    prompt, label = _prompt_and_label(example)
    text(left, top - 0.05, label, fontsize=9.5, weight="bold", color="#888888")
    box_w = RIGHT - left
    for size, most in ((13, 3), (11.5, 5), (10, 8)):
        shown = _wrap(prompt, box_w - 0.3, size, mono=True, lines=8)
        if shown.count("\n") < most:
            break
    prompt_artist = text(left + 0.15, top - 0.32, shown, fontsize=size, family=mono, color="#1a1a1a", va="top",
                         linespacing=1.35)
    box_h = tall(prompt_artist) + 0.2
    box(left, top - 0.22 - box_h, box_w, box_h, facecolor="#f2f2f2", edgecolor="none")
    y = top - 0.22 - box_h - 0.14
    cards = [card_text(reply, example) if reply.record else None for reply in replies]
    legend = None
    if example.scoring == "gsm8k":
        legend = "In each card: the model's working in grey, then the final answer it states in black."
    elif any(card and card.reasoning for card in cards):
        legend = "In each card: the model's reasoning in grey, then its answer in black."
    for line, style in ((_answer_key(example), {"fontsize": 11, "color": "#555555"}),
                        (legend, {"fontsize": 10, "color": "#888888"})):
        if line:
            artist = text(left, y, _wrap(line, box_w, style["fontsize"], mono=False, lines=2), va="top", **style)
            y -= tall(artist) + 0.07
    header_bottom = min(header_bottom, y)

    notes = [
        _method_note(spec, example),
        "WALL-OSS flow was trained to reason before answering, when it chooses to. G0.5 is trained to reason only about "
        "robot tasks, and WALL-OSS 0.5 to answer directly.",
    ]
    footer = figure.text(LEFT / WIDTH, 0.03, "\n".join(textwrap.fill(note, 195) for note in notes), fontsize=8.6,
                         color="#777777", va="bottom", linespacing=1.45)
    footer_top = 0.03 * HEIGHT + tall(footer)

    columns, gap = 4, 0.15
    card_w = (RIGHT - LEFT - (columns - 1) * gap) / columns
    grid_top = header_bottom - 0.12
    card_h = min(max((grid_top - footer_top - 0.12 - gap) / 2, 1.3), 2.0)
    for index, (reply, card) in enumerate(zip(replies, cards)):
        row, column = divmod(index, columns)
        x = LEFT + column * (card_w + gap)
        top = grid_top - row * (card_h + gap)
        box(x, top - card_h, card_w, card_h, facecolor="#fafafa", edgecolor="#e2e2e2", linewidth=1)
        image = logo(reply.entry.logo_org)
        if image is not None:
            figure.add_artist(AnnotationBbox(
                OffsetImage(image, zoom=0.36 * 72 / 256), (x + 0.12, top - 0.12), xycoords=inches,
                box_alignment=(0, 1), frameon=False,
            ))
        text(x + 0.58, top - 0.13, reply.entry.label, fontsize=12.5, weight="bold", color="#1a1a1a", va="top")
        released = reply.entry.released
        text(x + 0.58, top - 0.36, f"{MONTHS[int(released[5:7]) - 1]} {released[:4]}", fontsize=9.5, color="#888888",
             va="top")
        label, tone = _verdict(None if reply.record is None else reply.record.get("score"))
        text(x + card_w - 0.12, top - 0.15, label, fontsize=9.5, weight="bold", color=tone, ha="right", va="top",
             bbox={"boxstyle": "round,pad=0.3", "facecolor": _tint(tone, 0.85), "edgecolor": "none"})
        if card is None:
            continue
        inner_x = x + 0.15
        floor = top - card_h + (0.3 if reply.action_tokens else 0.1)
        if card.answer is None:
            answer = text(inner_x, 0, "no final answer", fontsize=13, color="#888888", style="italic", va="top")
        else:
            value = f"“{card.answer}”" if card.quoted else card.answer
            answer = text(inner_x, 0, LTR + _clip(value, 25, 2 if card.reasoning or card.option else 3), fontsize=13,
                          color="#1a1a1a", va="top", linespacing=1.15)
        option = None
        if card.option:
            option = text(inner_x, 0, _clip(card.option, 44, 2), fontsize=9.5, color="#555555", va="top", linespacing=1.2)
        below = tall(answer) + (tall(option) + 0.05 if option else 0)
        y = top - 0.6
        if card.reasoning:
            reasoning = text(inner_x, y, "", fontsize=7, color="#8a8a8a", va="top", linespacing=1.2)
            for lines in range(6, 0, -1):
                reasoning.set_text(LTR + _clip(card.reasoning, 56, lines))
                if tall(reasoning) <= y - floor - below - 0.06:
                    break
            y -= tall(reasoning) + 0.06
        place(answer, inner_x, y)
        if option:
            place(option, inner_x, y - tall(answer) - 0.05)
        if reply.action_tokens:
            text(inner_x, top - card_h + 0.12, f"{reply.action_tokens} action tokens, decoded as text", fontsize=8.5,
                 color="#888888", style="italic")

    _credit(figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    return output


def main(argv: list[str] | None = None) -> int:
    from .data_load import load_examples
    from .runner import RESULTS_DIR

    parser = argparse.ArgumentParser(description="One slide: every VLA's reply to a single benchmark question.")
    parser.add_argument("--benchmark", default="textvqa", choices=sorted(BENCHMARKS))
    parser.add_argument("--id", default="37153", help="example id as stored in the results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    found = load_all(args.results_dir)
    replies, source = collect(found, args.benchmark, args.id)
    if source is None:
        print(f"no {args.benchmark} results for these VLAs in {args.results_dir}")
        return 1
    examples, _ = load_examples(BENCHMARKS[args.benchmark], source["limit"], source["seed"])
    example = next((example for example in examples if example.id == args.id), None)
    if example is None:
        print(f"example {args.id} is not in the scored {args.benchmark} subset")
        return 1
    for reply in replies:
        if reply.entry.model_key == "openvla" and reply.record and reply.record.get("prediction"):
            reply.action_tokens = action_token_count(reply.record["prediction"])
        shown = "no result" if reply.record is None else repr(reply.record.get("prediction") or reply.record.get("parsed"))
        print(f"{reply.entry.label}: {shown}" + (f"  ({reply.action_tokens} action tokens)" if reply.action_tokens else ""))
    output = args.output or args.results_dir / f"slide_example_{args.benchmark}_{args.id}.png"
    print(draw(replies, example, args.benchmark, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
