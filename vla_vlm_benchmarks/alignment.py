from __future__ import annotations

import re
from dataclasses import dataclass

_LAYER = re.compile(r"(?:^|\.)(?:layers|blocks)\.(\d+)\.")


def group_of(key: str) -> str:
    """Bucket a parameter name so coverage can be required per stack."""

    lowered = key.lower()
    if any(part in lowered for part in ("vision_tower", "vision_backbone", "visual")):
        return "vision"
    if "multi_modal_projector" in lowered or "projector." in lowered or lowered.startswith("projector"):
        return "projector"
    if "lm_head" in lowered or _is_output_projection(key):
        return "lm_head"
    if "embed_tokens" in lowered or lowered.endswith("wte.weight") or ".wte." in lowered:
        return "embedding"
    return "language"


@dataclass(frozen=True)
class TensorRef:
    key: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class CopyOp:
    target_key: str
    donor_key: str
    mode: str  # "full" or "vocab_prefix"
    donor_rows: int | None = None


@dataclass(frozen=True)
class AlignmentPlan:
    copies: tuple[CopyOp, ...]
    missing: tuple[str, ...]
    ambiguous: tuple[tuple[str, tuple[str, ...]], ...]

    def copied_keys(self) -> set[str]:
        return {op.target_key for op in self.copies}


def _is_output_projection(key: str) -> bool:
    """Molmo's untied output matrix. Block MLP ``ff_out`` weights are not it."""

    return key.endswith("ff_out.weight") and ".blocks." not in key and ".mlp." not in key


def _is_vocab_key(key: str) -> bool:
    return (
        "embed_tokens" in key
        or "lm_head" in key
        or key.endswith("wte.weight")
        or ".wte." in key
        or _is_output_projection(key)
    )


def _aliases(key: str, rewrites: tuple[tuple[str, str], ...]) -> list[str]:
    """Donor names some stacks differently from the stock VLM. Rewrites are applied in order."""

    aliases = [key]
    rewritten = key
    for old, new in rewrites:
        if old in rewritten:
            rewritten = rewritten.replace(old, new)
            aliases.append(rewritten)
    return aliases


def _boundary_suffix(donor_key: str, target_key: str) -> bool:
    if not donor_key.endswith(target_key):
        return False
    start = len(donor_key) - len(target_key)
    return start == 0 or donor_key[start - 1] == "."


def _vocab_mode(donor_shape: tuple[int, ...], target_shape: tuple[int, ...], key: str) -> str | None:
    if donor_shape == target_shape:
        return "full"
    vocab_key = _is_vocab_key(key)
    if (
        vocab_key
        and len(donor_shape) == 2
        and len(target_shape) == 2
        and donor_shape[1] == target_shape[1]
        and donor_shape[0] != target_shape[0]
    ):
        return "vocab_prefix"
    return None


def plan_alignment(
    target: list[TensorRef],
    donor: list[TensorRef],
    *,
    prefer: tuple[str, ...] = (),
    reject: tuple[str, ...] = (),
    embed_fallback: bool = False,
    rewrites: tuple[tuple[str, str], ...] = (),
) -> AlignmentPlan:
    """Match VLA tensors onto a stock VLM by suffix and shape.

    The action expert in these VLAs reuses block names (``layers.0...``). A
    donor key is eligible only when the shape matches and it does not contain
    a rejected substring. Prefer-substrings break the remaining ties toward the
    VLM stack (``paligemma``, not ``gemma_expert``). Anything still ambiguous
    is left uncopied so a silent mix of two experts cannot be scored.
    """

    copies: list[CopyOp] = []
    missing: list[str] = []
    ambiguous: list[tuple[str, tuple[str, ...]]] = []
    target_shape = {item.key: item.shape for item in target}

    for item in target:
        hits: list[tuple[str, str]] = []
        for source in donor:
            if any(token and token in source.key for token in reject):
                continue
            mode = _vocab_mode(source.shape, item.shape, item.key)
            if mode is None:
                continue
            if not any(_boundary_suffix(alias, item.key) for alias in _aliases(source.key, rewrites)):
                continue
            hits.append((source.key, mode))
        preferred = [hit for hit in hits if any(token and token in hit[0] for token in prefer)]
        chosen = preferred or hits
        if not chosen:
            missing.append(item.key)
            continue
        unique_keys = sorted({hit[0] for hit in chosen})
        if len(unique_keys) > 1:
            ambiguous.append((item.key, tuple(unique_keys)))
            continue
        donor_key, mode = next(hit for hit in chosen if hit[0] == unique_keys[0])
        donor_rows = None
        if mode == "vocab_prefix":
            donor_rows = next(source.shape[0] for source in donor if source.key == donor_key)
        copies.append(CopyOp(item.key, donor_key, mode, donor_rows))

    if embed_fallback:
        missing = _tie_vocab_matrices(
            copies,
            missing,
            target,
            donor,
            prefer=prefer,
            reject=reject,
            rewrites=rewrites,
        )

    return AlignmentPlan(tuple(copies), tuple(missing), tuple(ambiguous))


def _tie_vocab_matrices(
    copies: list[CopyOp],
    missing: list[str],
    target: list[TensorRef],
    donor: list[TensorRef],
    *,
    prefer: tuple[str, ...],
    reject: tuple[str, ...],
    rewrites: tuple[tuple[str, str], ...],
) -> list[str]:
    """Fill a missing input embedding from the output head, or the reverse.

    PaliGemma stores only ``lm_head`` and ties it to the input embedding.
    G0.5 and WALL-OSS store the input matrix under another name and omit ``lm_head``.
    """

    target_shape = {item.key: item.shape for item in target}
    copied = {op.target_key for op in copies}

    def add(target_key: str, donor_key: str) -> None:
        mode = _vocab_mode(
            next(item.shape for item in donor if item.key == donor_key),
            target_shape[target_key],
            target_key,
        )
        if mode is None:
            return
        rows = None
        if mode == "vocab_prefix":
            rows = next(item.shape[0] for item in donor if item.key == donor_key)
        copies.append(CopyOp(target_key, donor_key, mode, rows))
        copied.add(target_key)

    def donor_heads() -> list[str]:
        found = []
        for source in donor:
            if any(token and token in source.key for token in reject):
                continue
            aliases = _aliases(source.key, rewrites)
            if any(alias.endswith("lm_head.weight") or _is_output_projection(alias) for alias in aliases):
                found.append(source.key)
        preferred = [key for key in found if any(token and token in key for token in prefer)]
        return preferred or found

    still_missing = []
    ordered = [key for key in missing if group_of(key) == "embedding"]
    ordered += [key for key in missing if group_of(key) == "lm_head"]
    ordered += [key for key in missing if group_of(key) not in {"embedding", "lm_head"}]
    for key in ordered:
        group = group_of(key)
        if group == "embedding":
            heads = [
                op.donor_key
                for op in copies
                if group_of(op.target_key) == "lm_head"
                and _vocab_mode(
                    next(item.shape for item in donor if item.key == op.donor_key),
                    target_shape[key],
                    key,
                )
            ]
            if len(heads) != 1:
                heads = [
                    donor_key
                    for donor_key in donor_heads()
                    if _vocab_mode(
                        next(item.shape for item in donor if item.key == donor_key),
                        target_shape[key],
                        key,
                    )
                ]
            unique = sorted(set(heads))
            if len(unique) == 1 and key not in copied:
                add(key, unique[0])
            else:
                still_missing.append(key)
            continue
        if group == "lm_head":
            embeds = [
                op.donor_key
                for op in copies
                if group_of(op.target_key) == "embedding"
                and _vocab_mode(
                    next(item.shape for item in donor if item.key == op.donor_key),
                    target_shape[key],
                    key,
                )
            ]
            unique = sorted(set(embeds))
            if len(unique) == 1 and key not in copied:
                add(key, unique[0])
            else:
                still_missing.append(key)
            continue
        still_missing.append(key)
    return still_missing


def coverage_by_group(target_keys: list[str], plan: AlignmentPlan) -> dict[str, dict[str, int]]:
    copied = plan.copied_keys()
    groups: dict[str, dict[str, int]] = {}
    for key in target_keys:
        bucket = groups.setdefault(group_of(key), {"target": 0, "matched": 0})
        bucket["target"] += 1
        if key in copied:
            bucket["matched"] += 1
    return groups


def coverage_problems(
    groups: dict[str, dict[str, int]],
    plan: AlignmentPlan,
    *,
    lm_head_fallback: bool,
) -> list[str]:
    """100% of every present group must be copied. Ambiguous keys always fail.

    ``lm_head_fallback`` allows a missing lm_head only: GR00T's checkpoint has
    no generative head, and the caller keeps the baseline model's head.
    """

    problems: list[str] = []
    if plan.ambiguous:
        sample = ", ".join(key for key, _ in plan.ambiguous[:8])
        problems.append(f"{len(plan.ambiguous)} ambiguous keys, for example {sample}")
    for name, row in sorted(groups.items()):
        if row["target"] == 0:
            continue
        if name == "lm_head" and lm_head_fallback and row["matched"] == 0:
            continue
        if row["matched"] != row["target"]:
            problems.append(f"{name} {row['matched']}/{row['target']}")
    return problems


def text_layer_count(keys: list[str], reject: tuple[str, ...] = ()) -> int | None:
    """Depth of the text stack implied by parameter names, or None if unseen."""

    found: list[int] = []
    for key in keys:
        if any(token and token in key for token in reject):
            continue
        if group_of(key) != "language":
            continue
        match = _LAYER.search(key)
        if match:
            found.append(int(match.group(1)))
    if not found:
        return None
    return max(found) + 1
