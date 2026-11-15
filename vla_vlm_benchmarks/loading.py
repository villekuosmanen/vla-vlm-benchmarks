from __future__ import annotations

import gc
from pathlib import Path

from .alignment import (
    TensorRef,
    coverage_by_group,
    coverage_problems,
    plan_alignment,
    text_layer_count,
)
from .errors import EvalSetupError
from .prompts import format_prompt
from .prismatic import PRISMATIC_GROUPS, prismatic_hf_key
from .registry import (
    THINK_BUDGET,
    ModelSpec,
    architecture_checkpoint,
    processor_checkpoint,
)

_WEIGHT_PATTERNS = [
    "model.safetensors",
    "model.safetensors.index.json",
    "*.safetensors",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "pytorch_model-*.bin",
    "*.pt",
    "config.json",
]


class Adapter:
    """One loaded VLM. ``generate_text`` and ``continuation_logprobs`` share its processor."""

    def __init__(self, spec: ModelSpec, model, processor, tokenizer, device: str, coverage: dict, warnings: list[str], revisions: dict, text_layers: int):
        self.spec = spec
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.coverage = coverage
        self.warnings = warnings
        self.revisions = revisions
        self.text_layers = text_layers
        self.blank = _blank_image()

    def _move(self, encoded):
        inputs = _to_device(encoded, self.device)
        if self.spec.family == "prismatic" and "pixel_values" in inputs:
            # OpenVLA's remote code feeds pixel_values to the timm patch embedding without a cast.
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        return inputs

    def generate_text(self, images, text: str, max_new_tokens: int, stop: tuple[str, ...] = ()) -> str:
        """Greedy reply to ``text`` wrapped in the family's prompt format.

        ``max_new_tokens`` bounds the answer. A reply that opens with ``<think>``
        may first spend up to ``THINK_BUDGET`` tokens reasoning; the reply is
        returned whole and ``split_reasoning`` separates the two.
        """

        import torch

        prepared = self._prepare_images(images)
        encoded = self._encode(prepared, text)
        inputs = self._move(encoded)
        input_length = inputs["input_ids"].shape[1]
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        kwargs = {
            "max_new_tokens": max_new_tokens + THINK_BUDGET,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": eos_id if pad_id is None else pad_id,
            "eos_token_id": eos_id,
            "stopping_criteria": _stopping_criteria(self.tokenizer, input_length, max_new_tokens, stop),
        }
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        if output.shape[1] <= input_length:
            new_tokens = output[0]
        else:
            new_tokens = output[0, input_length:]
        return self.tokenizer.decode(new_tokens.detach().cpu(), skip_special_tokens=True).strip()

    def continuation_logprobs(self, images, context: str, continuations: list[str]) -> list[float]:
        """Summed log-probability of each continuation after a raw ``context`` (lm-eval loglikelihood).

        No chat template or task prefix: the context is the benchmark's text as
        is, the way lm-eval scores multiple-choice tasks.
        """

        prepared = self._prepare_images(images)
        return [self._continuation_logprob(prepared, context, continuation) for continuation in continuations]

    def close(self) -> None:
        import torch

        self.model = None
        self.processor = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_images(self, images) -> list:
        prepared = [image for image in images if image is not None]
        if not prepared and self.spec.requires_image:
            prepared = [self.blank]
        if self.spec.prompt_style == "prismatic_pure" and len(prepared) != 1:
            raise EvalSetupError("OpenVLA and the Prismatic baseline take one image")
        return prepared

    def _encode(self, images: list, text: str):
        style = self.spec.prompt_style
        if style == "paligemma_pt":
            return self.processor(
                text=self.processor.image_token + format_prompt(style, text),
                images=images[0],
                return_tensors="pt",
            )
        if style == "prismatic_pure":
            return _call_prismatic(self.processor, format_prompt(style, text), images[0])
        if style == "qwen_base":
            return _encode_raw(self.processor, images, format_prompt(style, text))
        if style not in {"qwen_chat", "molmo"}:
            raise EvalSetupError(f"unknown prompt style {style}")
        return _encode_qwen(self.processor, self.tokenizer, images, format_prompt(style, text), self.warnings)

    def _encode_plain(self, images: list, text: str):
        if self.spec.prompt_style == "prismatic_pure":
            return _call_prismatic(self.processor, text, images[0])
        return _encode_raw(self.processor, images, text)

    def _continuation_logprob(self, images: list, context: str, continuation: str) -> float:
        import torch

        if self.spec.prompt_style == "paligemma_pt":
            # PaliGemma's prefix-LM puts the answer in the suffix after its "\n"
            # separator, which stands in for lm-eval's " " target delimiter.
            encoded = self.processor(
                text=self.processor.image_token + context,
                images=images[0],
                suffix=continuation.lstrip(),
                return_tensors="pt",
            )
            labels = encoded.pop("labels")[0]
            positions = labels.ne(-100).nonzero().flatten().tolist()
            if positions and int(labels[positions[-1]]) == self.tokenizer.eos_token_id:
                positions = positions[:-1]
            if not positions:
                raise EvalSetupError(f"{continuation!r} produced no suffix tokens")
            inputs = self._move(encoded)
            with torch.inference_mode():
                logits = self.model(**inputs).logits[0].float()
            predicted_at = torch.tensor(positions, device=logits.device) - 1
            log_probs = torch.log_softmax(logits[predicted_at], dim=-1)
            targets = inputs["input_ids"][0, positions]
            return float(log_probs.gather(-1, targets.unsqueeze(-1)).sum().item())

        # lm-eval's _encode_pair: the continuation is whatever the whole string
        # encodes to past the context's own token count.
        context_length = self._encode_plain(images, context)["input_ids"].shape[1]
        whole = self._encode_plain(images, context + continuation)
        length = whole["input_ids"].shape[1] - context_length
        if length <= 0:
            raise EvalSetupError(f"{continuation!r} added no tokens")
        inputs = self._move(whole)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0].float()
        # Counted from the end: Prismatic splices its image patches into the
        # sequence inside forward(), so logits are longer than input_ids.
        log_probs = torch.log_softmax(logits[-length - 1 : -1], dim=-1)
        targets = inputs["input_ids"][0, -length:]
        return float(log_probs.gather(-1, targets.unsqueeze(-1)).sum().item())


def load_adapter(spec: ModelSpec, device: str) -> Adapter:
    import torch

    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    warnings: list[str] = []
    try:
        model, coverage = _load_model(spec, dtype, warnings)
    except EvalSetupError:
        raise
    except Exception as exc:
        hint = ""
        message = str(exc).lower()
        if spec.gated and any(token in message for token in ("401", "403", "gated", "access")):
            hint = " Accept the license on the Hub page and run `hf auth login`."
        raise EvalSetupError(f"failed to load {spec.checkpoint}.{hint}\n{exc}") from exc
    _freeze(model)
    model.to(device=device, dtype=dtype)
    processor = _load_processor(processor_checkpoint(spec), _processor_trust(spec))
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.append("tokenizer had no pad token; eos is used")
    text_layers = _live_text_layers(model)
    if spec.expected_text_layers is not None and text_layers != spec.expected_text_layers:
        warnings.append(
            f"expected {spec.expected_text_layers} text layers after loading, found {text_layers}"
        )
    return Adapter(
        spec=spec,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
        coverage=coverage,
        warnings=warnings,
        revisions={
            "checkpoint": _revision(spec.checkpoint),
            "architecture": _revision(architecture_checkpoint(spec)),
            "processor": _revision(processor_checkpoint(spec)),
        },
        text_layers=text_layers,
    )


def _load_model(spec: ModelSpec, dtype, warnings: list[str]):
    if spec.loader == "hf":
        return _load_pretrained(spec.checkpoint, dtype, spec.trust_remote_code), {"source": "huggingface"}
    if spec.loader == "openvla":
        return (
            _load_pretrained(spec.checkpoint, dtype, spec.trust_remote_code),
            {"source": "huggingface"},
        )
    if spec.loader == "truncate":
        if spec.truncate_layers is None:
            raise EvalSetupError(f"{spec.key} has no truncate_layers")
        model = _load_pretrained(architecture_checkpoint(spec), dtype, spec.trust_remote_code)
        _truncate_text_layers(model, spec.truncate_layers)
        return model, {"source": "truncated", "text_layers": spec.truncate_layers}
    if spec.loader == "prismatic":
        model = _load_from_config(architecture_checkpoint(spec), dtype, True)
        donor = _prismatic_donor(spec, model.config, warnings)
        coverage = _copy_donor(model, donor, spec, warnings, pretrained_architecture=False)
        return model, coverage
    if spec.loader == "vla_weights":
        from .registry import MODELS

        trust = MODELS[spec.baseline_key].trust_remote_code if spec.baseline_key else spec.trust_remote_code
        model = _load_pretrained(architecture_checkpoint(spec), dtype, trust)
        if spec.donor_via_architecture:
            donor = _architecture_donor(spec, model, dtype, warnings)
        else:
            donor = _open_donor(spec.checkpoint)
        if spec.fused_splits:
            donor = _SplitDonor(donor, spec.fused_splits)
        if spec.truncate_to_donor:
            depth = text_layer_count(donor.keys(), spec.reject)
            if depth is None:
                raise EvalSetupError(
                    f"no text-layer tensors in {spec.checkpoint}; first keys: {donor.keys()[:20]}"
                )
            _truncate_text_layers(model, depth)
            if spec.expected_text_layers is not None and depth != spec.expected_text_layers:
                warnings.append(
                    f"donor text depth is {depth}, the published select_layer is {spec.expected_text_layers}"
                )
        coverage = _copy_donor(model, donor, spec, warnings, pretrained_architecture=True)
        return model, coverage
    raise EvalSetupError(f"unknown loader {spec.loader}")


def _architecture_donor(spec: ModelSpec, model, dtype, warnings: list[str]):
    """Load the VLA checkpoint through the baseline's own class.

    Transformers converts some checkpoint layouts while loading (fused
    projections split, layers renamed). Loading the VLA the same way applies
    the conversion the baseline got, so the copy compares like with like.
    """

    import copy
    import json

    import torch
    from huggingface_hub import hf_hub_download

    donor_text = json.loads(Path(hf_hub_download(spec.checkpoint, "config.json")).read_text())
    donor_text = donor_text.get("text_config", donor_text)
    config = copy.deepcopy(model.config)
    text = getattr(config, "text_config", config)
    base, extra = text.vocab_size, getattr(text, "additional_vocab_size", 0) or 0
    donor_base = donor_text["vocab_size"]
    if (donor_text.get("additional_vocab_size") or 0) != extra:
        raise EvalSetupError(f"{spec.checkpoint} has a different number of special-token rows")
    text.vocab_size = donor_base
    donor_model, info = type(model).from_pretrained(
        spec.checkpoint, config=config, dtype=dtype, low_cpu_mem_usage=True, output_loading_info=True
    )
    if info["missing_keys"]:
        raise EvalSetupError(
            f"{spec.checkpoint} leaves {len(info['missing_keys'])} tensors of "
            f"{type(model).__name__} unset: {sorted(info['missing_keys'])[:10]}"
        )
    dropped = [
        key for key in info["unexpected_keys"]
        if not any(token in key for token in spec.reject) and not key.endswith("inv_freq")
    ]
    if dropped:
        raise EvalSetupError(f"{spec.checkpoint} has tensors the conversion did not place: {sorted(dropped)[:10]}")
    tensors = {key: value.detach() for key, value in donor_model.state_dict().items()}
    del donor_model
    if extra and donor_base > base:
        for key, value in tensors.items():
            if "embed_tokens" in key and value.shape[0] == donor_base + extra:
                tensors[key] = torch.cat([value[:base], value[donor_base:]])
                warnings.append(
                    f"{key}: copied rows 0..{base - 1} and the {extra} special-token rows "
                    f"(donor rows {donor_base}..{donor_base + extra - 1})"
                )
    return _MemoryDonor(tensors)


def _multimodal_factories():
    import transformers

    factories = []
    for name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForVision2Seq"):
        factory = getattr(transformers, name, None)
        if factory is not None:
            factories.append(factory)
    if not factories:
        raise EvalSetupError("this transformers build has no multimodal auto class")
    return factories


def _load_pretrained(repo: str, dtype, trust_remote_code: bool):
    kwargs = {"dtype": dtype, "trust_remote_code": trust_remote_code, "low_cpu_mem_usage": True}
    last_error: Exception | None = None
    for factory in _multimodal_factories():
        try:
            model = factory.from_pretrained(repo, **kwargs)
            model.eval()
            return model
        except TypeError as exc:
            if "dtype" not in str(exc):
                last_error = exc
                continue
            model = factory.from_pretrained(
                repo, torch_dtype=dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=True
            )
            model.eval()
            return model
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if any(token in message for token in ("401", "403", "gated", "repository not found")):
                raise
    remote = _remote_model_class(repo, trust_remote_code)
    if remote is not None:
        try:
            model = remote.from_pretrained(repo, **kwargs)
        except TypeError:
            model = remote.from_pretrained(
                repo, torch_dtype=dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=True
            )
        model.eval()
        return model
    assert last_error is not None
    raise last_error


def _remote_model_class(repo: str, trust_remote_code: bool):
    if not trust_remote_code:
        return None
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    auto_map = getattr(config, "auto_map", None) or {}
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModel"):
        dotted = auto_map.get(name)
        if dotted:
            return get_class_from_dynamic_module(dotted, repo)
    return None


def _processor_trust(spec: ModelSpec) -> bool:
    from .registry import MODELS

    processor = processor_checkpoint(spec)
    if spec.baseline_key and processor == MODELS[spec.baseline_key].checkpoint:
        return MODELS[spec.baseline_key].trust_remote_code
    return spec.trust_remote_code


def _load_from_config(repo: str, dtype, trust_remote_code: bool):
    del dtype
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(repo, trust_remote_code=trust_remote_code)
    remote = _remote_model_class(repo, trust_remote_code)
    if remote is not None:
        model = remote(config)
        model.eval()
        return model
    last_error: Exception | None = None
    for factory in _multimodal_factories():
        try:
            try:
                model = factory.from_config(config, trust_remote_code=trust_remote_code)
            except TypeError:
                model = factory.from_config(config)
            model.eval()
            return model
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _load_processor(repo: str, trust_remote_code: bool):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(repo, trust_remote_code=trust_remote_code)


def _copy_donor(model, donor, spec: ModelSpec, warnings: list[str], pretrained_architecture: bool) -> dict:
    del warnings
    parameters = _parameters(model)
    plan = plan_alignment(
        [TensorRef(key, tuple(param.shape)) for key, param in parameters],
        [TensorRef(key, donor.shape(key)) for key in donor.keys()],
        prefer=spec.prefer,
        reject=spec.reject,
        embed_fallback=spec.embed_fallback,
        rewrites=spec.rewrites,
    )
    groups = coverage_by_group([key for key, _ in parameters], plan)
    problems = coverage_problems(groups, plan, lm_head_fallback=spec.lm_head_fallback)
    if problems:
        missing = ", ".join(plan.missing[:30])
        donor_sample = ", ".join(donor.keys()[:20])
        raise EvalSetupError(
            f"refusing to score {spec.key}: copied weights would leave baseline or random values in place\n"
            + "\n".join(problems)
            + f"\nmissing: {missing}\ndonor keys: {donor_sample}"
        )
    named = dict(parameters)
    vocab_notes = []
    for op in plan.copies:
        parameter = named[op.target_key]
        tensor = donor.tensor(op.donor_key).to(device=parameter.device, dtype=parameter.dtype)
        if op.mode == "full":
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise EvalSetupError(f"shape changed for {op.target_key}")
            parameter.data.copy_(tensor)
            continue
        copied = min(parameter.shape[0], tensor.shape[0])
        parameter.data[:copied].copy_(tensor[:copied])
        extra = parameter.shape[0] - copied
        if extra > 0 and not pretrained_architecture:
            raise EvalSetupError(
                f"{op.target_key} has {extra} rows past the donor's {copied}; "
                "they would stay randomly initialized"
            )
        output_row = "lm_head" in op.target_key or op.target_key.endswith("ff_out.weight")
        if extra > 0 and output_row:
            _mask_logits_from(model.get_submodule(op.target_key.rsplit(".", 1)[0]), copied)
            vocab_notes.append(f"{op.target_key}: copied {copied}, masked {extra} logits the VLA has no rows for")
        elif extra > 0:
            vocab_notes.append(f"{op.target_key}: copied {copied}, kept the baseline's {extra} trailing rows")
        elif tensor.shape[0] > copied:
            vocab_notes.append(f"{op.target_key}: copied first {copied} of {tensor.shape[0]}")
    lm_head = groups.get("lm_head", {"target": 0, "matched": 0})
    return {
        "source": spec.loader,
        "groups": groups,
        "lm_head_from_baseline": bool(spec.lm_head_fallback and lm_head["target"] and lm_head["matched"] == 0),
        "vocab": vocab_notes,
        "n_copied": len(plan.copies),
    }


def _mask_logits_from(head, first_masked: int) -> None:
    """Make vocabulary ids past the donor's output rows unreachable, in generation and scoring alike."""

    if getattr(head, "_vla_masked_from", None) == first_masked:
        return

    def hook(_module, _inputs, logits):
        logits[..., first_masked:] = float("-inf")
        return logits

    head.register_forward_hook(hook)
    head._vla_masked_from = first_masked


def _parameters(model) -> list[tuple[str, object]]:
    try:
        return list(model.named_parameters(remove_duplicate=False))
    except TypeError:
        return list(model.named_parameters())


def _freeze(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _truncate_text_layers(model, depth: int) -> None:
    import torch.nn as nn

    name, module, attr, current = _text_stack(model)
    if current < depth:
        raise EvalSetupError(f"{name} has {current} text layers; cannot keep {depth}")
    if current == depth:
        return
    stack = getattr(module, attr)
    setattr(module, attr, nn.ModuleList(list(stack)[:depth]))
    config = getattr(module, "config", None)
    if config is not None and hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = depth
    root = getattr(model, "config", None)
    text = getattr(root, "text_config", None) if root is not None else None
    if text is not None and hasattr(text, "num_hidden_layers"):
        text.num_hidden_layers = depth


def _module_stack(module):
    import torch.nn as nn

    for attr in ("layers", "blocks"):
        stack = getattr(module, attr, None)
        if isinstance(stack, nn.ModuleList) and len(stack) > 0:
            return attr, stack
    return None, None


def _text_stack(model):
    found = []
    for name, module in model.named_modules():
        if any(part in name for part in ("vision", "visual", "action_expert")):
            continue
        attr, stack = _module_stack(module)
        if stack is None:
            continue
        found.append((name, module, attr, len(stack)))
    if not found:
        raise EvalSetupError("the architecture has no text layer stack")
    found.sort(key=lambda item: item[3], reverse=True)
    if len(found) > 1 and found[0][3] == found[1][3]:
        raise EvalSetupError(f"two text stacks of depth {found[0][3]}: {found[0][0]}, {found[1][0]}")
    return found[0]


def _live_text_layers(model) -> int:
    return _text_stack(model)[3]


class _MemoryDonor:
    def __init__(self, tensors: dict):
        self._tensors = tensors

    def keys(self) -> list[str]:
        return list(self._tensors)

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._tensors[key].shape)

    def tensor(self, key: str):
        return self._tensors[key]


class _SafetensorsDonor:
    def __init__(self, key_to_file: dict[str, Path]):
        from safetensors import safe_open

        self._files = key_to_file
        self._shapes: dict[str, tuple[int, ...]] = {}
        for path in sorted(set(key_to_file.values())):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in key_to_file:
                        self._shapes[key] = tuple(handle.get_slice(key).get_shape())

    def keys(self) -> list[str]:
        return list(self._files)

    def shape(self, key: str) -> tuple[int, ...]:
        return self._shapes[key]

    def tensor(self, key: str):
        from safetensors import safe_open

        with safe_open(self._files[key], framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


class _SplitDonor:
    """A donor whose fused tensors are exposed as their row blocks, read lazily."""

    def __init__(self, donor, fused_splits):
        self._donor = donor
        self._parts: dict[str, tuple[str, int, int]] = {}
        self._plain: list[str] = []
        for key in donor.keys():
            rule = next(((infix, parts) for infix, parts in fused_splits if infix in key), None)
            if rule is None:
                self._plain.append(key)
                continue
            infix, parts = rule
            rows = donor.shape(key)[0]
            if rows != sum(size for _, size in parts):
                raise EvalSetupError(f"{key} has {rows} rows, the split expects {sum(size for _, size in parts)}")
            start = 0
            for part, size in parts:
                self._parts[key.replace(infix, part)] = (key, start, size)
                start += size

    def keys(self) -> list[str]:
        return self._plain + list(self._parts)

    def shape(self, key: str) -> tuple[int, ...]:
        if key in self._parts:
            source, _, size = self._parts[key]
            return (size, *self._donor.shape(source)[1:])
        return self._donor.shape(key)

    def tensor(self, key: str):
        if key in self._parts:
            source, start, size = self._parts[key]
            return self._donor.tensor(source)[start : start + size].contiguous()
        return self._donor.tensor(key)


def _open_donor(checkpoint: str):
    path = Path(checkpoint)
    if not path.exists():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(checkpoint, allow_patterns=_WEIGHT_PATTERNS))
    if path.is_file():
        return _donor_from_file(path)
    return _donor_from_directory(path)


def _donor_from_directory(path: Path):
    index = path / "model.safetensors.index.json"
    if index.exists():
        return _donor_from_index(path, index)
    single = path / "model.safetensors"
    if single.exists():
        return _SafetensorsDonor(_safetensor_map(single))
    shards = sorted(path.glob("*.safetensors"))
    if len(shards) == 1:
        return _SafetensorsDonor(_safetensor_map(shards[0]))
    bin_index = path / "pytorch_model.bin.index.json"
    if bin_index.exists():
        return _memory_from_bin_index(path, bin_index)
    single_bin = path / "pytorch_model.bin"
    if single_bin.exists():
        return _MemoryDonor(_torch_tensors(single_bin))
    points = sorted(path.glob("*.pt"))
    if len(points) == 1:
        return _MemoryDonor(_torch_tensors(points[0]))
    names = sorted(child.name for child in path.iterdir())[:40]
    raise EvalSetupError(f"no unique weight file in {path}: {names}")


def _donor_from_file(path: Path):
    if path.suffix == ".safetensors":
        return _SafetensorsDonor(_safetensor_map(path))
    return _MemoryDonor(_torch_tensors(path))


def _safetensor_map(path: Path) -> dict[str, Path]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: path for key in handle.keys()}


def _donor_from_index(directory: Path, index_path: Path):
    import json

    weight_map = json.loads(index_path.read_text())["weight_map"]
    return _SafetensorsDonor({key: directory / filename for key, filename in weight_map.items()})


def _memory_from_bin_index(directory: Path, index_path: Path):
    import json

    import torch

    weight_map = json.loads(index_path.read_text())["weight_map"]
    merged = {}
    for filename in sorted(set(weight_map.values())):
        blob = torch.load(directory / filename, map_location="cpu", weights_only=True)
        merged.update(blob)
    return _MemoryDonor(merged)


def _torch_tensors(path: Path) -> dict:
    import torch

    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    return _flatten_torch_blob(blob)


def _flatten_torch_blob(blob) -> dict:
    import torch

    if isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
        blob = blob["state_dict"]
    if isinstance(blob, dict) and "model" in blob and isinstance(blob["model"], dict):
        model = blob["model"]
        if {"projector", "llm_backbone"} <= set(model):
            return _remap_prismatic(model)
        if model and all(torch.is_tensor(value) for value in model.values()):
            blob = model
    if isinstance(blob, dict) and blob and all(torch.is_tensor(value) for value in blob.values()):
        return blob
    preview = list(blob)[:20] if isinstance(blob, dict) else type(blob).__name__
    raise EvalSetupError(f"unrecognized checkpoint layout: {preview}")


def _prismatic_donor(spec: ModelSpec, config, warnings: list[str]):
    if not spec.prismatic_repo or not spec.prismatic_file:
        raise EvalSetupError(f"{spec.key} is missing the Prismatic file name")
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(spec.prismatic_repo, spec.prismatic_file))
    tensors = _torch_tensors(path)
    if not any(key.startswith("vision_backbone.") for key in tensors):
        tensors.update(_timm_vision_tensors(config))
        warnings.append(
            "Prismatic checkpoints omit the frozen vision backbone; "
            f"loaded {', '.join(config.timm_model_ids)} from timm as Prismatic does"
        )
    return _MemoryDonor(tensors)


def _timm_vision_tensors(config) -> dict:
    """Pretrained DINOv2 / SigLIP towers, built the way PrismaticVLM.from_pretrained builds them."""

    import timm

    featurizers = ("dino_featurizer", "siglip_featurizer")
    if not config.use_fused_vision_backbone or len(config.timm_model_ids) != 2:
        raise EvalSetupError(f"expected the fused DINOv2 + SigLIP backbone, got {config.timm_model_ids}")
    tensors = {}
    for name, model_id, size in zip(featurizers, config.timm_model_ids, config.image_sizes):
        tower = timm.create_model(model_id, pretrained=True, num_classes=0, img_size=size)
        for key, value in tower.state_dict().items():
            tensors[prismatic_hf_key("vision_backbone", f"{name}.{key}")] = value
    return tensors


def _remap_prismatic(model: dict) -> dict:
    import torch

    renamed = {}
    for group in PRISMATIC_GROUPS:
        if group not in model:
            continue
        group_tensors = model[group]
        if not isinstance(group_tensors, dict):
            raise EvalSetupError(f"Prismatic group {group} is {type(group_tensors).__name__}")
        for key, value in group_tensors.items():
            if not torch.is_tensor(value):
                continue
            renamed[prismatic_hf_key(group, key)] = value
    return renamed


def _attach_chat_template(processor, tokenizer) -> bool:
    """Qwen3.5-Base ships the template on the tokenizer, not the processor."""

    if getattr(processor, "chat_template", None):
        return True
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return False
    processor.chat_template = template
    return True


def _vision_prefixed(processor, images: list, text: str) -> str:
    token = getattr(processor, "image_token", None)
    if not images or not token or token in text:
        return text
    start = getattr(processor, "vision_start_token", "") or ""
    end = getattr(processor, "vision_end_token", "") or ""
    return "".join(f"{start}{token}{end}" for _ in images) + text


def _encode_raw(processor, images: list, text: str):
    kwargs = {"text": _vision_prefixed(processor, images, text), "return_tensors": "pt"}
    if images:
        kwargs["images"] = images[0] if len(images) == 1 else images
    return processor(**kwargs)


def _encode_qwen(processor, tokenizer, images: list, text: str, warnings: list[str]):
    if not _attach_chat_template(processor, tokenizer):
        if "fell back to raw text with vision placeholders" not in warnings:
            warnings.append("processor has no chat template; fell back to raw text with vision placeholders")
        return _encode_raw(processor, images, text)
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and "chat template" not in str(exc).lower():
            raise
        if "fell back to raw text with vision placeholders" not in warnings:
            warnings.append("processor has no chat template; fell back to raw text with vision placeholders")
        return _encode_raw(processor, images, text)


def _call_prismatic(processor, prompt: str, image):
    try:
        return processor(text=prompt, images=image, return_tensors="pt")
    except TypeError:
        return processor(prompt, image, return_tensors="pt")


def _to_device(encoded, device: str):
    if hasattr(encoded, "to"):
        return encoded.to(device)
    return {key: _move_value(value, device) for key, value in encoded.items()}


def _move_value(value, device: str):
    import torch

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_move_value(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_value(item, device) for key, item in value.items()}
    return value


def _stopping_criteria(tokenizer, prompt_length: int, answer_budget: int, stops: tuple[str, ...]):
    """Stop at ``answer_budget`` answer tokens or at any stop string.

    Stop strings follow lm-eval's stop_sequences_criteria, since
    ``generate(stop_strings=...)`` does not exist in the pinned transformers
    4.40. Batches hold one example.
    """

    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    def flag(input_ids, done: bool):
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)

    class StopOnText(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            text = tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=False)
            return flag(input_ids, any(stop in text for stop in stops))

    class AnswerBudget(StoppingCriteria):
        def __init__(self):
            self.closed_at = None

        def __call__(self, input_ids, scores, **kwargs):
            generated = input_ids.shape[1] - prompt_length
            if generated < answer_budget:
                return flag(input_ids, False)
            if self.closed_at is None:
                text = tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=True).lstrip()
                if not text.startswith("<think>"):
                    return flag(input_ids, True)
                if "</think>" not in text:
                    return flag(input_ids, False)
                self.closed_at = generated
            return flag(input_ids, generated - self.closed_at >= answer_budget)

    criteria = [AnswerBudget()]
    if stops:
        criteria.append(StopOnText())
    return StoppingCriteriaList(criteria)


def _blank_image():
    from PIL import Image

    return Image.new("RGB", (224, 224), "white")


def _revision(repo: str) -> str | None:
    if Path(repo).exists():
        return "local"
    try:
        from huggingface_hub import model_info

        return model_info(repo).sha
    except Exception:
        return None
