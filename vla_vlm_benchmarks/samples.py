from __future__ import annotations

import re
from dataclasses import dataclass

# Post-prompts from lmms-eval (vqav2, textvqa, pope via LLaVA's question files, ai2d).
SHORT_ANSWER = "Answer the question using a single word or phrase."
OPTION_LETTER = "Answer with the option's letter from the given choices directly."


@dataclass(frozen=True)
class ParsedExample:
    """One benchmark item before its images are decoded.

    ``scoring`` picks the protocol: ``vqa`` (VQA accuracy over annotator
    answers), ``pope`` (yes/no), ``choice`` (generated letter, or exact match
    when ``letters`` is empty), ``loglikelihood`` (lm-eval cloze over
    ``choices``), or ``gsm8k`` (generated chain of thought, final number).
    """

    id: str
    row_index: int
    text: str
    gold: str
    scoring: str
    answers: tuple[str, ...] = ()
    letters: tuple[str, ...] = ()
    choices: tuple[str, ...] = ()
    group: str | None = None
    keep: bool = True
    drop_reason: str | None = None

    @property
    def chance(self) -> float | None:
        """Expected score of a uniform random guess among the stated options."""

        if self.scoring == "loglikelihood":
            return 1.0 / len(self.choices)
        if self.scoring == "choice":
            if self.letters:
                return 1.0 / len(self.letters)
            return 0.5 if self.gold.strip().lower() in {"yes", "no"} else 0.0
        return None


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    gold: str
    scoring: str
    answers: tuple[str, ...]
    letters: tuple[str, ...]
    choices: tuple[str, ...]
    group: str | None
    chance: float | None
    images: tuple = ()


def _id(row: dict, index: int) -> str:
    for key in ("question_id", "questionId", "id", "qid"):
        value = row.get(key)
        if value is not None and value != "":
            return str(value)
    return str(index)


def _answers(row: dict) -> list[str]:
    raw = row.get("answers")
    collected: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                collected.append(item)
            elif isinstance(item, dict) and item.get("answer") is not None:
                collected.append(str(item["answer"]))
    elif isinstance(raw, dict):
        nested = raw.get("answer")
        if isinstance(nested, list):
            collected.extend(str(item) for item in nested)
        elif isinstance(nested, str):
            collected.append(nested)
    if not collected and row.get("multiple_choice_answer") is not None:
        collected.append(str(row["multiple_choice_answer"]))
    if not collected and row.get("answer") is not None:
        collected.append(str(row["answer"]))
    return collected


def parse_vqa(row: dict, index: int) -> ParsedExample:
    answers = _answers(row)
    # Display only; scoring reads all the annotators' answers. TextVQA has no
    # consensus field, so show the most common answer as VQAv2's does.
    gold = str(row.get("multiple_choice_answer") or (max(answers, key=answers.count) if answers else ""))
    return ParsedExample(
        id=_id(row, index),
        row_index=index,
        text=f"{str(row['question']).strip()}\n{SHORT_ANSWER}",
        gold=gold,
        scoring="vqa",
        answers=tuple(answers or [gold]),
    )


def parse_pope(row: dict, index: int) -> ParsedExample:
    gold = str(row["answer"]).strip().lower()
    group = row.get("category", row.get("pope_split"))
    return ParsedExample(
        id=_id(row, index),
        row_index=index,
        text=f"{str(row['question']).strip()}\n{SHORT_ANSWER}",
        gold=gold,
        scoring="pope",
        answers=(gold,),
        group=None if group is None else str(group).lower(),
    )


def _option_text(letter: str, option: str) -> str:
    return re.sub(rf"^{letter}[\).:\s]+", "", str(option).strip(), flags=re.IGNORECASE)


def parse_ai2d(row: dict, index: int) -> ParsedExample:
    raw_options = list(row.get("options") or [])
    letters = tuple(chr(ord("A") + position) for position in range(len(raw_options)))
    choices = tuple(_option_text(letter, option) for letter, option in zip(letters, raw_options))
    answer = str(row.get("answer")).strip()
    if answer.isdigit() and int(answer) < len(letters):
        gold = letters[int(answer)]
    elif answer.upper() in letters:
        gold = answer.upper()
    else:
        gold = next((letter for letter, text in zip(letters, choices) if text.lower() == answer.lower()), "")
    question = str(row.get("question") or "").strip()
    lines = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
    return ParsedExample(
        id=_id(row, index),
        row_index=index,
        text=f"{question}\n{lines}\n{OPTION_LETTER}",
        gold=gold,
        scoring="choice",
        answers=(gold,),
        letters=letters,
        choices=choices,
        keep=bool(gold and letters),
        drop_reason=None if gold and letters else "unresolved_answer",
    )


def inline_options(question: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Option letters and texts embedded in a question (``A. x`` on lines or after ``Choices:``).

    Returns empty tuples unless the letters run A, B, C, ... without gaps.
    """

    body = question.split("Choices:", 1)[1] if "Choices:" in question else question
    body = re.split(r"Please answer", body)[0]
    parts = re.split(r"(?:^|(?<=\s))\(?([A-F])[\.\)](?=\s)", body)
    letters: list[str] = []
    texts: list[str] = []
    for position in range(1, len(parts) - 1, 2):
        letter = parts[position]
        expected = chr(ord("A") + len(letters))
        if letter != expected:
            if letter in letters:
                texts[-1] = f"{texts[-1]} {letter}. {parts[position + 1].strip()}".strip()
                continue
            return (), ()
        letters.append(letter)
        texts.append(parts[position + 1].strip())
    return tuple(letters), tuple(texts)


def parse_realworldqa(row: dict, index: int) -> ParsedExample:
    """RealWorldQA questions carry their own options and answer instruction; used verbatim."""

    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    if re.fullmatch(r"[A-Fa-f]", answer):
        gold = answer.upper()
        letters, choices = inline_options(question)
        if gold not in letters:
            letters, choices = ("A", "B", "C", "D"), ()
        return ParsedExample(
            id=_id(row, index),
            row_index=index,
            text=question,
            gold=gold,
            scoring="choice",
            answers=(gold,),
            letters=letters,
            choices=choices,
        )
    return ParsedExample(
        id=_id(row, index),
        row_index=index,
        text=question,
        gold=answer,
        scoring="choice",
        answers=(answer,),
    )


def parse_erqa(row: dict, index: int) -> ParsedExample:
    """ERQA questions carry ``Choices: A. ...`` and the answer instruction; used verbatim."""

    images = row.get("images", row.get("image"))
    if isinstance(images, list):
        count = len(images)
    elif isinstance(row.get("visual_indices"), list) and row["visual_indices"]:
        count = len(row["visual_indices"])
    else:
        count = 1
    question = str(row["question"]).strip()
    gold = str(row["answer"]).strip().upper()[:1]
    letters, choices = inline_options(question)
    if gold not in letters:
        letters, choices = ("A", "B", "C", "D"), ()
    return ParsedExample(
        id=_id(row, index),
        row_index=index,
        text=question,
        gold=gold,
        scoring="choice",
        answers=(gold,),
        letters=letters,
        choices=choices,
        keep=count == 1,
        drop_reason=None if count == 1 else "multi_image",
    )


def parse_arc(row: dict, index: int) -> ParsedExample:
    """lm-eval ``arc_easy``: ``Question: q\\nAnswer:`` then each choice text as a continuation."""

    labels = [str(label) for label in row["choices"]["label"]]
    texts = tuple(str(text) for text in row["choices"]["text"])
    gold_index = labels.index(str(row["answerKey"]).strip())
    return ParsedExample(
        id=_id(row, index),
        row_index=index,
        text=f"Question: {str(row['question']).strip()}\nAnswer:",
        gold=str(gold_index),
        scoring="loglikelihood",
        answers=(texts[gold_index],),
        choices=texts,
    )


# The eight chain-of-thought exemplars of Wei et al. (2022), verbatim from
# lm-evaluation-harness lm_eval/tasks/gsm8k/gsm8k-cot.yaml (task version 3.0).
GSM8K_COT_EXEMPLARS = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
        "5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
)


def gsm8k_cot_prompt(question: str) -> str:
    shots = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in GSM8K_COT_EXEMPLARS)
    return f"{shots}\n\nQ: {question.strip()}\nA:"


def parse_gsm8k(row: dict, index: int) -> ParsedExample:
    final = str(row["answer"]).split("####")[-1].strip()
    return ParsedExample(
        id=_id(row, index),
        row_index=index,
        text=gsm8k_cot_prompt(str(row["question"])),
        gold=final,
        scoring="gsm8k",
        answers=(final,),
    )
