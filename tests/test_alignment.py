import pytest

from vla_vlm_benchmarks.alignment import (
    TensorRef,
    coverage_by_group,
    coverage_problems,
    plan_alignment,
    text_layer_count,
)


def test_paligemma_stack_wins_over_the_action_expert():
    shape = (100, 8)
    target = [
        TensorRef("model.language_model.layers.0.self_attn.q_proj.weight", shape),
        TensorRef("lm_head.weight", shape),
        TensorRef("model.language_model.embed_tokens.weight", shape),
    ]
    donor = [
        TensorRef(
            "model.paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
            shape,
        ),
        TensorRef("model.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight", shape),
        TensorRef("model.paligemma_with_expert.paligemma.lm_head.weight", shape),
        TensorRef("model.paligemma_with_expert.gemma_expert.lm_head.weight", shape),
    ]
    plan = plan_alignment(
        target,
        donor,
        prefer=("paligemma_with_expert.paligemma",),
        reject=("gemma_expert",),
        embed_fallback=True,
    )
    assert not plan.ambiguous
    assert not plan.missing
    copied = {op.target_key: op.donor_key for op in plan.copies}
    assert "paligemma.model.language_model" in copied["model.language_model.layers.0.self_attn.q_proj.weight"]
    assert copied["model.language_model.embed_tokens.weight"] == copied["lm_head.weight"]
    assert "gemma_expert" not in copied["lm_head.weight"]


def test_ambiguous_experts_are_not_copied():
    shape = (4, 4)
    plan = plan_alignment(
        [TensorRef("lm_head.weight", shape)],
        [
            TensorRef("a.lm_head.weight", shape),
            TensorRef("b.lm_head.weight", shape),
        ],
    )
    assert plan.copies == ()
    assert plan.ambiguous[0][0] == "lm_head.weight"
    problems = coverage_problems(coverage_by_group(["lm_head.weight"], plan), plan, lm_head_fallback=False)
    assert problems


def test_vocab_prefix_and_lm_head_fallback():
    plan = plan_alignment(
        [TensorRef("lm_head.weight", (100, 4))],
        [TensorRef("model.lm_head.weight", (120, 4))],
    )
    assert plan.copies[0].mode == "vocab_prefix"
    assert plan.copies[0].donor_rows == 120
    groups = coverage_by_group(["lm_head.weight"], plan)
    assert coverage_problems(groups, plan, lm_head_fallback=False) == []

    missing_head = plan_alignment(
        [TensorRef("lm_head.weight", (4, 4)), TensorRef("model.layers.0.weight", (4, 4))],
        [TensorRef("backbone.model.layers.0.weight", (4, 4))],
    )
    assert [op.target_key for op in missing_head.copies] == ["model.layers.0.weight"]
    assert coverage_problems(
        coverage_by_group(["lm_head.weight", "model.layers.0.weight"], missing_head),
        missing_head,
        lm_head_fallback=True,
    ) == []


def test_paligemma_vision_rename_and_tied_embedding():
    plan = plan_alignment(
        [
            TensorRef("model.vision_tower.embeddings.patch_embedding.weight", (4, 4)),
            TensorRef("model.language_model.embed_tokens.weight", (10, 4)),
        ],
        [
            TensorRef(
                "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight",
                (4, 4),
            ),
            TensorRef("paligemma_with_expert.gemma_expert.lm_head.weight", (10, 4)),
            TensorRef("paligemma_with_expert.paligemma.lm_head.weight", (10, 4)),
        ],
        prefer=("paligemma_with_expert.paligemma",),
        reject=("gemma_expert",),
        embed_fallback=True,
        rewrites=((".vision_tower.vision_model.", ".vision_tower."),),
    )
    copied = {op.target_key: op.donor_key for op in plan.copies}
    assert not plan.missing
    assert "vision_model.embeddings" in copied["model.vision_tower.embeddings.patch_embedding.weight"]
    assert copied["model.language_model.embed_tokens.weight"].endswith("paligemma.lm_head.weight")


def test_wall_oss_prefix_and_tied_output():
    plan = plan_alignment(
        [
            TensorRef("model.language_model.layers.0.mlp.down_proj.weight", (8, 8)),
            TensorRef("model.visual.blocks.0.attn.qkv.weight", (8, 8)),
            TensorRef("model.language_model.embed_tokens.weight", (12, 8)),
            TensorRef("lm_head.weight", (12, 8)),
        ],
        [
            TensorRef("model.layers.0.mlp.down_proj.weight", (8, 8)),
            TensorRef("visual.blocks.0.attn.qkv.weight", (8, 8)),
            TensorRef("model.embed_tokens.weight", (20, 8)),
        ],
        embed_fallback=True,
        rewrites=(
            ("model.layers.", "model.language_model.layers."),
            ("model.embed_tokens.", "model.language_model.embed_tokens."),
            ("visual.", "model.visual."),
        ),
    )
    copied = {op.target_key: op for op in plan.copies}
    assert not plan.missing, plan.missing
    assert copied["model.language_model.embed_tokens.weight"].mode == "vocab_prefix"
    assert copied["lm_head.weight"].donor_key == "model.embed_tokens.weight"


def test_wall_oss_copies_the_split_vl_expert_of_each_layer():
    from vla_vlm_benchmarks.loading import _MemoryDonor, _SplitDonor
    from vla_vlm_benchmarks.registry import MODELS

    wall = MODELS["wall_oss"]
    layer = "model.language_model.layers.0."
    target = [
        TensorRef(layer + "self_attn.q_proj.weight", (2048, 2048)),
        TensorRef(layer + "self_attn.q_proj.bias", (2048,)),
        TensorRef(layer + "self_attn.k_proj.weight", (256, 2048)),
        TensorRef(layer + "self_attn.v_proj.weight", (256, 2048)),
        TensorRef(layer + "self_attn.o_proj.weight", (2048, 2048)),
        TensorRef(layer + "mlp.gate_proj.weight", (11008, 2048)),
        TensorRef(layer + "mlp.up_proj.weight", (11008, 2048)),
        TensorRef(layer + "mlp.down_proj.weight", (2048, 11008)),
        TensorRef(layer + "input_layernorm.weight", (2048,)),
        TensorRef(layer + "post_attention_layernorm.weight", (2048,)),
        TensorRef("model.language_model.norm.weight", (2048,)),
    ]
    shapes = {}
    for expert, width in ((0, 2048), (1, 1024)):
        prefix = "model.layers.0."
        shapes[f"{prefix}self_attn.qkv_proj_experts.{expert}.weight"] = (2560, width)
        shapes[f"{prefix}self_attn.qkv_proj_experts.{expert}.bias"] = (2560,)
        shapes[f"{prefix}self_attn.o_proj_experts.{expert}.weight"] = (width, 2048)
        shapes[f"{prefix}moe.experts.{expert}.gate_up_proj.weight"] = (22016 if expert == 0 else 4096, width)
        shapes[f"{prefix}moe.experts.{expert}.down_proj.weight"] = (width, 11008 if expert == 0 else 2048)
        shapes[f"{prefix}input_layernorms.{expert}.weight"] = (width,)
        shapes[f"{prefix}post_attention_layernorms.{expert}.weight"] = (width,)
        shapes[f"model.norms.{expert}.weight"] = (width,)

    class Shape:
        def __init__(self, shape):
            self.shape = shape

    donor = _SplitDonor(_MemoryDonor({key: Shape(shape) for key, shape in shapes.items()}), wall.fused_splits)
    refs = [TensorRef(key, donor.shape(key)) for key in donor.keys()]
    plan = plan_alignment(target, refs, prefer=wall.prefer, reject=wall.reject, rewrites=wall.rewrites)
    assert not plan.missing and not plan.ambiguous
    copied = {op.target_key: op.donor_key for op in plan.copies}
    assert len(copied) == len(target)
    assert copied[layer + "self_attn.k_proj.weight"] == "model.layers.0.self_attn.k_proj.weight"
    assert copied[layer + "mlp.up_proj.weight"] == "model.layers.0.mlp.up_proj.weight"
    assert not any("1." in source.split("layers.0.")[-1] for source in copied.values())


def test_molmo_vocab_prefix_and_block_depth():
    plan = plan_alignment(
        [
            TensorRef("model.transformer.wte.weight", (100, 8)),
            TensorRef("model.transformer.ff_out.weight", (100, 8)),
            TensorRef("model.transformer.blocks.0.mlp.ff_out.weight", (8, 16)),
        ],
        [
            TensorRef("model.transformer.wte.weight", (130, 8)),
            TensorRef("model.action_expert.wte.weight", (130, 8)),
            TensorRef("model.transformer.ff_out.weight", (130, 8)),
            TensorRef("model.transformer.blocks.0.mlp.ff_out.weight", (8, 16)),
        ],
        reject=("action_expert",),
    )
    copied = {op.target_key: op for op in plan.copies}
    assert copied["model.transformer.wte.weight"].mode == "vocab_prefix"
    assert copied["model.transformer.ff_out.weight"].mode == "vocab_prefix"
    assert "action_expert" not in copied["model.transformer.wte.weight"].donor_key
    assert copied["model.transformer.blocks.0.mlp.ff_out.weight"].mode == "full"
    assert (
        text_layer_count(
            [
                "model.transformer.blocks.35.self_attn.att_proj.weight",
                "model.vision_backbone.blocks.26.attn.weight",
                "model.action_expert.blocks.35.mlp.fc1.weight",
            ],
            reject=("action_expert",),
        )
        == 36
    )


def test_text_stack_uses_molmo_blocks_and_skips_the_action_expert():
    torch = pytest.importorskip("torch")
    from torch import nn

    from vla_vlm_benchmarks.loading import _text_stack

    class Stack(nn.Module):
        def __init__(self, depth: int):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(depth)])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = Stack(36)
            self.vision_backbone = Stack(27)
            self.action_expert = Stack(36)

    name, _module, attr, depth = _text_stack(Model())
    assert depth == 36
    assert attr == "blocks"
    assert name.endswith("transformer")
    del torch


def test_text_layer_count_ignores_vision_and_the_action_head():
    keys = [
        "backbone.language_model.layers.15.self_attn.q_proj.weight",
        "backbone.visual.layers.27.proj.weight",
        "action_head.layers.31.weight",
    ]
    assert text_layer_count(keys, reject=("action_head",)) == 16
