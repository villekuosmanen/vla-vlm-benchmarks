from vla_vlm_benchmarks.prompts import format_prompt
from vla_vlm_benchmarks.samples import (
    GSM8K_COT_EXEMPLARS,
    inline_options,
    parse_ai2d,
    parse_arc,
    parse_erqa,
    parse_gsm8k,
    parse_pope,
    parse_realworldqa,
    parse_vqa,
)
from vla_vlm_benchmarks.subset import choose_indices


def test_prompts_match_the_original_vlm_formats():
    assert format_prompt("paligemma_pt", "What color?") == "answer en What color?"
    assert format_prompt("prismatic_pure", "Pick up\nthe cup") == "In: Pick up the cup\nOut:"
    assert format_prompt("qwen_chat", "What color?") == "What color?"
    assert format_prompt("qwen_base", "What color?") == "What color?\n"
    assert format_prompt("qwen_base", "Q: 2+2?\nA:") == "Q: 2+2?\nA:"


def test_subset_is_stable_and_sorted():
    assert choose_indices(10, 0, 0) == list(range(10))
    assert choose_indices(10, None, 0) == list(range(10))
    first = choose_indices(50, 7, 3)
    assert first == choose_indices(50, 7, 3)
    assert first == sorted(first)
    assert len(first) == 7
    assert first != choose_indices(50, 7, 4)


def test_parsers_use_the_reference_prompts():
    vqa = parse_vqa(
        {"question_id": 7, "question": "What color?", "answers": [{"answer": "red"}, {"answer": "red"}]},
        0,
    )
    assert vqa.id == "7" and vqa.answers == ("red", "red") and vqa.scoring == "vqa"
    assert vqa.text == "What color?\nAnswer the question using a single word or phrase."
    majority = parse_vqa({"question": "Two ladders?", "answers": ["no", "yes", "yes"]}, 0)
    assert majority.gold == "yes"

    pope = parse_pope({"question": "Is there a cat?", "answer": "Yes", "category": "Adversarial"}, 1)
    assert pope.gold == "yes" and pope.group == "adversarial"
    assert pope.text == "Is there a cat?\nAnswer the question using a single word or phrase."

    diagram = parse_ai2d({"question": "Which way?", "options": ["left", "right"], "answer": "1"}, 0)
    assert diagram.gold == "B" and diagram.letters == ("A", "B") and diagram.choices == ("left", "right")
    assert diagram.text == (
        "Which way?\nA. left\nB. right\nAnswer with the option's letter from the given choices directly."
    )

    arc = parse_arc(
        {"id": "q1", "question": "Why?", "choices": {"label": ["1", "2"], "text": ["heat", "cold"]}, "answerKey": "2"},
        0,
    )
    assert arc.text == "Question: Why?\nAnswer:" and arc.gold == "1" and arc.choices == ("heat", "cold")
    assert arc.scoring == "loglikelihood" and arc.chance == 0.5

    multi = parse_erqa({"question_id": "ERQA_1", "question": "Where?", "answer": "C", "images": [1, 2]}, 0)
    single = parse_erqa({"question_id": "ERQA_2", "question": "Where?", "answer": "c", "visual_indices": [0]}, 1)
    assert multi.keep is False and multi.drop_reason == "multi_image"
    assert single.keep is True and single.gold == "C" and single.text == "Where?"

    math = parse_gsm8k({"question": "What is 2+2?", "answer": "Add them.\n#### 4"}, 0)
    assert math.gold == "4" and math.scoring == "gsm8k"
    assert math.text.startswith("Q: There are 15 trees in the grove.")
    assert math.text.endswith("\n\nQ: What is 2+2?\nA:")
    assert math.text.count("The answer is") == len(GSM8K_COT_EXEMPLARS) == 8


def test_questions_with_embedded_options_are_used_verbatim():
    question = "Which side?\n\nA. Left\nB. Straight\nC. Right\nPlease answer directly with only the letter of the correct option and nothing else."
    parsed = parse_realworldqa({"question": question, "answer": "C"}, 0)
    assert parsed.text == question and parsed.letters == ("A", "B", "C")
    assert parsed.choices == ("Left", "Straight", "Right")
    assert inline_options("Pick. Choices: A. Point A. B. Point B. Please answer directly.") == (
        ("A", "B"),
        ("Point A.", "Point B."),
    )
    assert inline_options("no options here") == ((), ())
