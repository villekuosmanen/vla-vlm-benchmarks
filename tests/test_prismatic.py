from vla_vlm_benchmarks.prismatic import prismatic_hf_key


def test_openvla_conversion_names():
    assert prismatic_hf_key("projector", "projector.0.weight") == "projector.fc1.weight"
    assert prismatic_hf_key("projector", "2.bias") == "projector.fc2.bias"
    assert prismatic_hf_key("projector", "projector.4.weight") == "projector.fc3.weight"
    assert (
        prismatic_hf_key("llm_backbone", "llm.model.layers.0.self_attn.q_proj.weight")
        == "language_model.model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        prismatic_hf_key("vision_backbone", "dino_featurizer.blocks.0.ls1.gamma")
        == "vision_backbone.featurizer.blocks.0.ls1.scale_factor"
    )
    assert (
        prismatic_hf_key("vision_backbone", "siglip_featurizer.blocks.0.attn.qkv.weight")
        == "vision_backbone.fused_featurizer.blocks.0.attn.qkv.weight"
    )
