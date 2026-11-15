from types import SimpleNamespace

from vla_vlm_benchmarks.registry import BENCHMARKS
from vla_vlm_benchmarks.samples import parse_ai2d, parse_arc, parse_erqa, parse_gsm8k, parse_pope, parse_realworldqa
from vla_vlm_benchmarks.scoring import (
    aggregate_metrics,
    extract_choice,
    gsm8k_exact_match,
    gsm8k_flexible,
    gsm8k_strict,
    extract_yes_no,
    normalize_vqa,
    score_generation,
    score_loglikelihood,
    split_reasoning,
    truncate_at,
    vqa_accuracy,
    yes_no_report,
)


def test_only_the_answer_after_reasoning_is_scored():
    assert split_reasoning("<think>I see a dog.</think> \n\nYes") == ("I see a dog.", "Yes")
    assert split_reasoning("No\n</think>\n\nNo") == ("No", "No")
    assert split_reasoning("<think>A diagram shows") == ("A diagram shows", "")
    assert split_reasoning(" Yes ") == (None, "Yes")


def test_a_contained_vqa_answer_is_only_a_secondary_score():
    answers = ["10:10"] * 9 + ["9:10"]
    reply = "The time shown on the watch face is 10:10."
    assert vqa_accuracy(reply, answers) == 0.0
    assert vqa_accuracy(reply, answers, contains=True) == 1.0
    assert vqa_accuracy("10:10", answers, contains=True) == vqa_accuracy("10:10", answers)
    assert vqa_accuracy("two dogs", ["2"] * 10, contains=True) == 1.0
    assert vqa_accuracy("it is 110", ["10"] * 10, contains=True) == 0.0


def test_vqa_normalization_matches_the_official_evaluator():
    assert normalize_vqa("The dog.") == "dog"
    assert normalize_vqa("a cat") == "cat"
    assert normalize_vqa("two") == "2"
    assert normalize_vqa("1,000") == "1000"
    assert normalize_vqa("3.14") == "3.14"
    assert normalize_vqa("dont") == "don't"
    assert normalize_vqa("Yes!") == "yes"


def test_vqa_accuracy_leave_one_out():
    answers = ["dog"] * 3 + ["cat"] * 7
    # Leaving out a "dog" leaves 2 dogs (score 2/3); leaving out a "cat" leaves 3 (score 1).
    expected = (3 * (2 / 3) + 7 * 1) / 10
    assert abs(vqa_accuracy("The dog.", answers) - expected) < 1e-9
    assert vqa_accuracy("bird", answers) == 0
    assert vqa_accuracy("cat", ["cat"]) == 1
    assert vqa_accuracy("dog", ["cat"]) == 0
    assert vqa_accuracy("Two", ["2"] * 10) == 1


def test_yes_no_extraction_is_vlmevalkit_yorn():
    assert extract_yes_no("Yes, there is a cat.") == "yes"
    assert extract_yes_no("No") == "no"
    assert extract_yes_no("No\nThere is no dog") == "no"
    assert extract_yes_no("Yes or no?") is None
    assert extract_yes_no("There is not a dog.") is None
    assert extract_yes_no("飛타反ϊ望") is None
    assert extract_yes_no("") is None


def test_choice_extraction_is_vlmevalkit_can_infer():
    letters = ["A", "B", "C", "D"]
    assert extract_choice("B", letters) == "B"
    assert extract_choice("b", letters) == "B"
    assert extract_choice("(C)", letters) == "C"
    assert extract_choice("e", letters) is None
    assert extract_choice("The answer is D.", letters) == "D"
    assert extract_choice("A or B", letters) is None
    assert extract_choice("box", letters) is None
    assert extract_choice("the left one", letters, ["left", "right", "up", "down"]) == "A"
    assert extract_choice("", letters) is None


def test_gsm8k_filters_and_exact_match_follow_lm_eval():
    text = "She had 5 + 3 = 8 apples. The answer is 8.\n\n"
    assert gsm8k_strict(text) == "8"
    assert gsm8k_flexible(text) == "8."
    assert gsm8k_exact_match(gsm8k_flexible(text), "8")
    assert gsm8k_strict("so 1,000 in total") is None
    assert gsm8k_flexible("so $1,000 in total") == "$1,000"
    assert gsm8k_exact_match("$1,000", "1000")
    assert not gsm8k_exact_match(None, "4")
    assert truncate_at("The answer is 4.\n\nQ: next", ("Q:",)) == "The answer is 4.\n\n"


def test_examples_score_through_their_protocols():
    pope = parse_pope({"question": "Is there a cat?", "answer": "no", "category": "random"}, 0)
    assert score_generation(pope, "No.")["score"] == 1.0
    garbled = score_generation(pope, "飛타反ϊ望")
    assert garbled["parsed"] is None and garbled["score"] == 0.0

    diagram = parse_ai2d({"question": "Which way?", "options": ["left", "right"], "answer": "1"}, 0)
    assert score_generation(diagram, "B")["score"] == 1.0
    assert score_generation(diagram, "I am not sure")["parsed_ok"] is False

    count = parse_realworldqa({"question": "How many?\nPlease answer directly with a single word or number.", "answer": "2"}, 0)
    assert score_generation(count, "Two.")["score"] == 1.0

    math = parse_gsm8k({"question": "What is 2+2?", "answer": "Add them.\n#### 4"}, 0)
    scored = score_generation(math, "2 + 2 = 4. The answer is 4.")
    assert scored["score"] == 1.0 and scored["score_flexible"] == 1.0

    arc = parse_arc(
        {"id": "q1", "question": "Why?", "choices": {"label": ["A", "B"], "text": ["heat", "a long cold"]}, "answerKey": "B"},
        0,
    )
    # Summed log-prob prefers the short choice; per-character normalization flips it.
    scored = score_loglikelihood(arc, [-2.0, -3.0])
    assert scored["correct"] is False and scored["correct_norm"] is True


def test_chance_levels():
    assert parse_ai2d({"question": "q", "options": ["a", "b", "c", "d"], "answer": "0"}, 0).chance == 0.25
    erqa = parse_erqa(
        {"question": "Where? Choices: A. up. B. down. C. left. Please answer directly with only the letter.", "answer": "B", "visual_indices": [0]},
        0,
    )
    assert erqa.letters == ("A", "B", "C") and erqa.chance == 1 / 3
    assert parse_realworldqa({"question": "Is it wet?", "answer": "Yes"}, 0).chance == 0.5
    assert parse_realworldqa({"question": "How many?", "answer": "3"}, 0).chance == 0.0

    report = yes_no_report([("yes", "yes"), ("no", "yes"), ("yes", "yes"), ("no", "yes")])
    assert report["yes_rate"] == 1 and abs(report["f1"] - 2 / 3) < 1e-9
    assert report["chance_f1"] == 0.5 and abs(report["always_yes_f1"] - 2 / 3) < 1e-9


def test_pope_metrics_average_the_three_splits():
    records = [
        {"gold": "yes", "parsed": "yes", "group": "random", "score": 1.0, "parsed_ok": True},
        {"gold": "no", "parsed": "no", "group": "random", "score": 1.0, "parsed_ok": True},
        {"gold": "yes", "parsed": "yes", "group": "adversarial", "score": 1.0, "parsed_ok": True},
        {"gold": "no", "parsed": "yes", "group": "adversarial", "score": 0.0, "parsed_ok": True},
    ]
    metrics = aggregate_metrics(records, BENCHMARKS["pope"])
    assert metrics["per_split"]["random"]["f1"] == 1
    assert abs(metrics["per_split"]["adversarial"]["f1"] - 2 / 3) < 1e-9
    assert abs(metrics["f1"] - (1 + 2 / 3) / 2) < 1e-9
    assert metrics["chance"] == 0.5


def test_errors_count_as_wrong_and_unparsed_is_reported():
    benchmark = SimpleNamespace(scoring="choice", primary_metric="accuracy")
    records = [
        {"score": 1.0, "parsed_ok": True, "chance": 0.25},
        {"score": 0.0, "parsed_ok": False, "chance": 0.25},
        {"score": 0.0, "parsed_ok": False, "chance": 0.25, "error": "RuntimeError: boom"},
    ]
    metrics = aggregate_metrics(records, benchmark)
    assert abs(metrics["accuracy"] - 1 / 3) < 1e-9
    assert metrics["chance"] == 0.25
    assert metrics["n_errors"] == 1
    assert abs(metrics["unparsed_rate"] - 2 / 3) < 1e-9
