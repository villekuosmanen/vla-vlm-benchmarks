from __future__ import annotations

import re

# Answer normalization from the official VQA evaluator
# (GT-Vision-Lab/VQA, PythonEvaluationTools/vqaEvaluation/vqaEval.py), the
# same tables lmms-eval's EvalAIAnswerProcessor uses for VQAv2 and TextVQA.
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
_MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def _process_punctuation(text: str) -> str:
    out = text
    for mark in _PUNCT:
        if (mark + " " in text or " " + mark in text) or _COMMA_STRIP.search(text) is not None:
            out = out.replace(mark, "")
        else:
            out = out.replace(mark, " ")
    return _PERIOD_STRIP.sub("", out)


def _process_digit_article(text: str) -> str:
    words = [_MANUAL_MAP.get(word, word) for word in text.lower().split()]
    words = [word for word in words if word not in _ARTICLES]
    return " ".join(_CONTRACTIONS.get(word, word) for word in words)


def normalize_vqa(text: str) -> str:
    """EvalAI answer processing: whitespace, punctuation, number words, articles, contractions."""

    cleaned = text.replace("\n", " ").replace("\t", " ").strip().lower()
    return _process_digit_article(_process_punctuation(cleaned))


def vqa_accuracy(prediction: str, answers: list[str], contains: bool = False) -> float:
    """Official VQA accuracy in ``[0, 1]``: mean over the leave-one-out sets of ``min(1, matches / 3)``.

    As in the official evaluator and lmms-eval, the prediction is always
    normalized and the references only when the annotators disagree. With
    ``contains`` (a secondary score, not the official one) a human answer also
    matches when it appears as whole words inside the normalized reply, so a
    right answer given as a sentence still counts.
    """

    if not answers:
        return 0.0
    if contains:
        reply = f" {normalize_vqa(prediction)} "
        normalized = [normalize_vqa(answer) for answer in answers]
        matched = [bool(answer) and f" {answer} " in reply for answer in normalized]
    else:
        references = [answer.replace("\n", " ").replace("\t", " ").strip() for answer in answers]
        predicted = normalize_vqa(prediction)
        if len(set(references)) > 1:
            references = [_process_digit_article(_process_punctuation(answer)) for answer in references]
        matched = [answer == predicted for answer in references]
    if len(matched) == 1:
        return 1.0 if matched[0] else 0.0
    hits = sum(matched)
    return sum(min(1.0, (hits - own) / 3.0) for own in matched) / len(matched)


def extract_yes_no(text: str) -> str | None:
    """VLMEvalKit's ``YOrN_Extraction``: ``yes`` or ``no`` when exactly one of them is a word.

    Anything else is unparsed. It scores wrong and counts as not-yes for F1,
    as in VLMEvalKit and lmms-eval. (The 2023 POPE script read every answer
    without "no"/"not" as yes, which gives unreadable output F1 0.667.)
    """

    words = _process_punctuation(text.lower()).split()
    if "yes" in words and "no" not in words:
        return "yes"
    if "no" in words and "yes" not in words:
        return "no"
    return None


def extract_choice(text: str, letters: list[str], choices: list[str] | None = None) -> str | None:
    """Option letter from a free-form answer: VLMEvalKit's ``can_infer`` exact matching.

    Exactly one option letter as a standalone token wins; otherwise exactly
    one option text contained in the answer. Anything else is unparsed and
    scored wrong (VLMEvalKit would fall back to an LLM judge; we do not). A
    bare letter in either case also counts, as lmms-eval's case-insensitive
    letter filter would read it.
    """

    bare = re.fullmatch(r"[\W_]*([A-Za-z])[\W_]*", text.strip())
    if bare and bare.group(1).upper() in letters:
        return bare.group(1).upper()
    cleaned = text
    for char in ".()[],:;!*#{}":
        cleaned = cleaned.replace(char, " ")
    tokens = set(cleaned.split())
    present = [letter for letter in letters if letter in tokens]
    if len(present) == 1:
        return present[0]
    if choices:
        lowered = text.lower()
        hits = [letter for letter, choice in zip(letters, choices) if str(choice).lower() in lowered]
        if len(hits) == 1:
            return hits[0]
    return None


def normalize_exact(text: str) -> str:
    """Exact-match normalization for short open answers (RealWorldQA): case, punctuation, number words."""

    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(_MANUAL_MAP.get(word, word) for word in cleaned.split())


# lm-eval gsm8k_cot filters and exact_match settings, regexes verbatim
# (the strict pattern's trailing "." is an unescaped wildcard there too).
_STRICT_GSM8K = re.compile(r"The answer is (\-?[0-9\.\,]+).")
_FLEXIBLE_GSM8K = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_GSM8K_IGNORE = (",", r"\$", r"(?s).*#### ", r"\.$")


def _regex_filter(pattern: re.Pattern, text: str, group_select: int) -> str | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    match = matches[group_select]
    if isinstance(match, tuple):
        match = next((group for group in match if group), None)
    return match.strip() if match else None


def gsm8k_strict(text: str) -> str | None:
    """lm-eval ``strict-match``: the first ``The answer is N``."""

    return _regex_filter(_STRICT_GSM8K, text, 0)


def gsm8k_flexible(text: str) -> str | None:
    """lm-eval ``flexible-extract``: the last number-like span."""

    return _regex_filter(_FLEXIBLE_GSM8K, text, -1)


def gsm8k_exact_match(extracted: str | None, gold: str) -> bool:
    """lm-eval ``exact_match`` with ignore_case and the gsm8k_cot ``regexes_to_ignore``."""

    if extracted is None:
        return False
    left, right = extracted, gold
    for pattern in _GSM8K_IGNORE:
        left = re.sub(pattern, "", left)
        right = re.sub(pattern, "", right)
    return left.lower() == right.lower()


def score_generation(example, prediction: str) -> dict:
    """Parse one generated answer the way the benchmark's reference harness does."""

    if example.scoring == "pope":
        parsed = extract_yes_no(prediction)
        return {"parsed": parsed, "parsed_ok": parsed is not None, "score": 1.0 if parsed == example.gold else 0.0}
    if example.scoring == "vqa":
        return {"parsed": normalize_vqa(prediction), "parsed_ok": bool(prediction.strip()),
                "score": vqa_accuracy(prediction, list(example.answers)),
                "score_contains": vqa_accuracy(prediction, list(example.answers), contains=True)}
    if example.scoring == "choice" and example.letters:
        parsed = extract_choice(prediction, list(example.letters), list(example.choices) or None)
        return {"parsed": parsed, "parsed_ok": parsed is not None, "score": 1.0 if parsed == example.gold else 0.0}
    if example.scoring == "choice":
        parsed = normalize_exact(prediction)
        return {"parsed": parsed, "parsed_ok": bool(parsed),
                "score": 1.0 if parsed == normalize_exact(example.gold) else 0.0}
    if example.scoring == "gsm8k":
        strict = gsm8k_strict(prediction)
        flexible = gsm8k_flexible(prediction)
        return {
            "parsed": strict,
            "parsed_flexible": flexible,
            "parsed_ok": flexible is not None,
            "score": 1.0 if gsm8k_exact_match(strict, example.gold) else 0.0,
            "score_flexible": 1.0 if gsm8k_exact_match(flexible, example.gold) else 0.0,
        }
    raise ValueError(f"no generative scoring for {example.scoring!r}")


def score_loglikelihood(example, logprobs: list[float]) -> dict:
    """lm-eval multiple_choice: argmax of summed log-prob (acc) and of log-prob per character (acc_norm)."""

    gold = int(example.gold)
    lengths = [float(len(choice)) for choice in example.choices]
    choice = max(range(len(logprobs)), key=lambda position: logprobs[position])
    normalized = [logprob / length for logprob, length in zip(logprobs, lengths)]
    choice_norm = max(range(len(normalized)), key=lambda position: normalized[position])
    return {
        "logprobs": [float(value) for value in logprobs],
        "parsed": str(choice),
        "parsed_ok": True,
        "choice_norm": choice_norm,
        "correct": choice == gold,
        "correct_norm": choice_norm == gold,
        "score": 1.0 if choice == gold else 0.0,
    }


def split_reasoning(text: str) -> tuple[str | None, str]:
    """Separate a ``<think>`` block from the answer after the last ``</think>``.

    Only the answer is scored, as in VLMEvalKit and lmms-eval. A block that
    never closes leaves no answer, so letters inside the reasoning are not
    read as a choice.
    """

    text = text.strip()
    if "</think>" in text:
        reasoning, answer = text.rsplit("</think>", 1)
        return reasoning.strip().removeprefix("<think>").strip(), answer.strip()
    if text.startswith("<think>"):
        return text.removeprefix("<think>").strip(), ""
    return None, text


def truncate_at(text: str, stops: tuple[str, ...]) -> str:
    """Cut a generation at the first stop sequence, as lm-eval's ``until`` does."""

    cut = len(text)
    for stop in stops:
        position = text.find(stop)
        if position != -1:
            cut = min(cut, position)
    return text[:cut]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def yes_no_report(pairs: list[tuple[str, str | None]]) -> dict[str, float]:
    """POPE metrics with ``yes`` as the positive class. Unparsed answers count as not-yes."""

    tp = fp = fn = yes = correct = 0
    for gold, predicted in pairs:
        yes += predicted == "yes"
        correct += predicted == gold
        if predicted == "yes" and gold == "yes":
            tp += 1
        elif predicted == "yes" and gold == "no":
            fp += 1
        elif predicted != "yes" and gold == "yes":
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = len(pairs)
    positives = sum(1 for gold, _ in pairs if gold == "yes")
    base_rate = positives / total if total else 0.0
    # Expected F1 of a coin flip: precision = base rate, recall = 0.5.
    chance_f1 = 2 * base_rate * 0.5 / (base_rate + 0.5) if total else 0.0
    return {
        "accuracy": correct / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_rate": yes / total if total else 0.0,
        "chance_f1": chance_f1,
        "always_yes_f1": 2 * base_rate / (1 + base_rate) if total else 0.0,
        "n": total,
    }


def aggregate_metrics(records: list[dict], benchmark) -> dict:
    """Benchmark-level metrics, the random-guess ``chance`` level, and the parse-failure rate."""

    n_errors = sum(1 for record in records if record.get("error"))
    metrics: dict = {"n": len(records), "n_errors": n_errors}
    scored = [record for record in records if record.get("score") is not None]
    if benchmark.scoring == "pope":
        groups: dict[str, list] = {}
        for record in records:
            groups.setdefault(record.get("group") or "all", []).append((record["gold"], record.get("parsed")))
        per_split = {name: yes_no_report(pairs) for name, pairs in sorted(groups.items())}
        metrics["per_split"] = per_split
        metrics["f1"] = _mean([split["f1"] for split in per_split.values()])
        metrics["accuracy"] = _mean([split["accuracy"] for split in per_split.values()])
        metrics["yes_rate"] = _mean([split["yes_rate"] for split in per_split.values()])
        metrics["chance"] = _mean([split["chance_f1"] for split in per_split.values()])
        metrics["always_yes_f1"] = _mean([split["always_yes_f1"] for split in per_split.values()])
    elif benchmark.scoring == "loglikelihood":
        metrics["acc"] = _mean([1.0 if record["correct"] else 0.0 for record in scored])
        metrics["acc_norm"] = _mean([1.0 if record["correct_norm"] else 0.0 for record in scored])
        metrics["chance"] = _mean([record["chance"] for record in records])
    elif benchmark.scoring == "gsm8k":
        metrics["exact_match_strict"] = _mean([record["score"] for record in scored])
        metrics["exact_match_flexible"] = _mean([record["score_flexible"] for record in scored])
    else:
        metrics[benchmark.primary_metric] = _mean([record["score"] for record in scored])
        if benchmark.scoring == "vqa":
            metrics["vqa_accuracy_contains"] = _mean([record.get("score_contains", 0.0) for record in scored])
        chances = [record["chance"] for record in records if record.get("chance") is not None]
        if chances and len(chances) == len(records):
            metrics["chance"] = _mean(chances)
    unparsed = [record for record in records if record.get("parsed_ok") is False]
    metrics["unparsed_rate"] = len(unparsed) / len(records) if records else 0.0
    if metrics.get(benchmark.primary_metric) is None:
        metrics["primary_missing"] = True
    return metrics
