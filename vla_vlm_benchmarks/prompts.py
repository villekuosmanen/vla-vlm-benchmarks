from __future__ import annotations

# Bumped whenever the text a style feeds the model changes, so finished results
# from the old prompt are rerun and never paired with new ones.
PROMPT_REVISIONS = {"paligemma_pt": 2, "qwen_base": 2}


def prompt_revision(style: str) -> int:
    return PROMPT_REVISIONS.get(style, 1)


def format_prompt(style: str, text: str) -> str:
    """Wrap a question in the prompt the original VLM was trained to read.

    Baseline and VLA of one family share this string. The VLA's robot-task
    template is intentionally not used: several of these policies have no slot
    for a free-form question, so that template would not measure the backbone.
    """

    body = text.strip()
    if style == "paligemma_pt":
        # Prefix used by the PaliGemma pt checkpoint (SigLIP-So400m + Gemma 2B).
        # The processor appends the "\n" separator; adding it here as well
        # produced a "\n\n" token that neither PaliGemma nor OpenPI trains on.
        return f"answer en {body}"
    if style == "prismatic_pure":
        # Llama-2 pure prompt from Prismatic / OpenVLA. Newlines inside the
        # question are flattened the way PurePromptBuilder does, and get_prompt()
        # strips the trailing space after "Out:".
        flat = " ".join(body.split())
        return f"In: {flat}\nOut:"
    if style == "qwen_base":
        # Raw completion after the image tokens, with no chat turn: G0.5's tokenizer
        # resolves its template's chat placeholders to empty strings, and a base
        # model is prompted without a chat template, as in lm-eval. An instruction
        # ends with a newline; a prompt ending in an answer cue (GSM8K's "A:") is
        # continued as is.
        return body if body.endswith(":") else f"{body}\n"
    if style in {"qwen_chat", "molmo"}:
        return body
    raise KeyError(f"unknown prompt style {style!r}")
