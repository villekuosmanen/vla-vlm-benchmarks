from __future__ import annotations

import io
from collections import Counter

from .errors import EvalSetupError
from .registry import BenchmarkSpec
from .samples import (
    Example,
    parse_ai2d,
    parse_arc,
    parse_erqa,
    parse_gsm8k,
    parse_pope,
    parse_realworldqa,
    parse_vqa,
)
from .subset import choose_indices

_IMAGE_COLUMNS = ("image", "images", "image_1", "img")
_PARSERS = {
    "pope": parse_pope,
    "vqav2": parse_vqa,
    "textvqa": parse_vqa,
    "realworldqa": parse_realworldqa,
    "ai2d": parse_ai2d,
    "erqa": parse_erqa,
    "arc_easy": parse_arc,
    "gsm8k": parse_gsm8k,
}


def load_examples(spec: BenchmarkSpec, limit: int, seed: int) -> tuple[list[Example], dict]:
    """The scored subset of ``spec``. With ``spec.categories``, ``limit`` applies per category."""

    dataset, split_name, warnings = _load_split(spec)
    image_column = _image_column(dataset.column_names) if spec.has_images else None
    if spec.has_images and image_column is None:
        raise EvalSetupError(
            f"{spec.dataset} has no image column in {dataset.column_names}"
        )
    scan = dataset
    if image_column and not spec.decode_images_during_scan:
        scan = dataset.remove_columns(image_column)

    parser = _PARSERS[spec.key]
    kept = []
    dropped: Counter[str] = Counter()
    per_category: dict[str, int] = {}
    if spec.categories:
        column = _find_column(scan.column_names, spec.category_column, "category", "pope_split")
        if column is None:
            raise EvalSetupError(f"{spec.dataset} has no category column in {scan.column_names}")
        values = [str(value).lower() for value in scan[column]]
        indexes = []
        for category in spec.categories:
            pool = [index for index, value in enumerate(values) if value == category]
            if not pool:
                raise EvalSetupError(f"{spec.dataset} has no rows in category {category!r}")
            per_category[category] = len(pool)
            indexes.extend(pool[position] for position in choose_indices(len(pool), limit, seed))
        preselected = True
    else:
        # Single-image filters have to see every row before the subset is drawn.
        # Unfiltered benchmarks draw the subset first: VQAv2 validation is 214k
        # rows, and scanning it once per model dominates.
        preselected = not (spec.single_image_only or spec.decode_images_during_scan)
        indexes = choose_indices(len(scan), limit, seed) if preselected else range(len(scan))
    for index in indexes:
        row = _plain_row(scan[index], image_column if spec.decode_images_during_scan else None)
        if spec.categories and spec.category_column in row and "category" not in row:
            row["category"] = row[spec.category_column]
        parsed = parser(row, index)
        if not parsed.keep:
            dropped[parsed.drop_reason or "dropped"] += 1
            continue
        kept.append(parsed)

    if preselected:
        selected = kept
    else:
        selected_positions = choose_indices(len(kept), limit, seed)
        selected = [kept[position] for position in selected_positions]
    examples = []
    for parsed in selected:
        images: tuple = ()
        if image_column is not None:
            images = tuple(_to_pils(dataset[parsed.row_index][image_column]))
            if spec.single_image_only and len(images) != 1:
                raise EvalSetupError(
                    f"{spec.key} example {parsed.id} has {len(images)} images; "
                    "the single-image filter should have removed it before subsetting"
                )
        examples.append(
            Example(
                id=parsed.id,
                text=parsed.text,
                gold=parsed.gold,
                scoring=parsed.scoring,
                answers=parsed.answers,
                letters=parsed.letters,
                choices=parsed.choices,
                group=parsed.group,
                chance=parsed.chance,
                images=images,
            )
        )
    if not examples:
        raise EvalSetupError(f"no examples kept for {spec.key} ({spec.dataset} {split_name})")
    meta = {
        "dataset": spec.dataset,
        "dataset_config": spec.dataset_config,
        "split": split_name,
        "dataset_revision": _dataset_revision(spec.dataset),
        "n_available": sum(per_category.values()) if per_category else len(kept),
        "n_available_per_category": per_category or None,
        "n_scored": len(examples),
        "n_dropped": int(sum(dropped.values())),
        "drop_reasons": dict(dropped),
        "warnings": warnings,
    }
    return examples, meta


def _load_split(spec: BenchmarkSpec):
    from datasets import load_dataset

    errors = []
    warnings: list[str] = []
    for split in spec.split_candidates:
        kwargs = {"path": spec.dataset, "split": split}
        if spec.dataset_config:
            kwargs["name"] = spec.dataset_config
        try:
            return load_dataset(**kwargs), split, warnings
        except Exception as exc:  # dataset ids and splits differ across mirrors
            errors.append(f"{spec.dataset_config or '-'}/{split}: {type(exc).__name__}: {exc}")
    detail = "\n".join(errors[-8:])
    raise EvalSetupError(f"could not load {spec.key} ({spec.dataset})\n{detail}")


def _image_column(columns: list[str]) -> str | None:
    for name in _IMAGE_COLUMNS:
        if name in columns:
            return name
    return None


def _find_column(columns: list[str], *names: str | None) -> str | None:
    for name in names:
        if name and name in columns:
            return name
    return None


def _plain_row(row, keep_image: str | None) -> dict:
    plain = {}
    for key in row.keys():
        if keep_image is None and key in _IMAGE_COLUMNS:
            continue
        plain[key] = row[key]
    return plain


def _to_pils(value):
    from PIL import Image

    if value is None:
        return []
    if isinstance(value, Image.Image):
        return [value.convert("RGB")]
    if isinstance(value, list):
        images = []
        for item in value:
            images.extend(_to_pils(item))
        return images
    if isinstance(value, dict) and value.get("bytes"):
        return [Image.open(io.BytesIO(value["bytes"])).convert("RGB")]
    if isinstance(value, str):
        return [Image.open(value).convert("RGB")]
    raise EvalSetupError(f"unrecognized image value {type(value).__name__}")


def _dataset_revision(repo: str) -> str | None:
    try:
        from huggingface_hub import dataset_info

        return dataset_info(repo).sha
    except Exception:
        return None
