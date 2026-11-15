from __future__ import annotations

# Key rewrite from the OpenVLA conversion script
# (openvla/vla-scripts/extern/convert_openvla_weights_to_hf.py). The base
# Prismatic checkpoint stores three sub-dicts; the HF OpenVLA class expects
# the names on the right.

PROJECTOR_KEY_MAPPING = {
    "projector.0.weight": "projector.fc1.weight",
    "projector.0.bias": "projector.fc1.bias",
    "projector.2.weight": "projector.fc2.weight",
    "projector.2.bias": "projector.fc2.bias",
    "projector.4.weight": "projector.fc3.weight",
    "projector.4.bias": "projector.fc3.bias",
}

PRISMATIC_GROUPS = ("vision_backbone", "projector", "llm_backbone")


def prismatic_hf_key(group: str, key: str) -> str:
    if group == "projector":
        mapped = key if key in PROJECTOR_KEY_MAPPING else f"projector.{key}"
        if mapped not in PROJECTOR_KEY_MAPPING:
            raise KeyError(key)
        return PROJECTOR_KEY_MAPPING[mapped]
    if group == "llm_backbone":
        prefix = "llm."
        if not key.startswith(prefix):
            raise KeyError(key)
        return "language_model." + key[len(prefix) :]
    if group == "vision_backbone":
        if key.startswith("dino_featurizer."):
            rest = key[len("dino_featurizer.") :]
            if rest.endswith(".gamma"):
                rest = rest[: -len(".gamma")] + ".scale_factor"
            return "vision_backbone.featurizer." + rest
        if key.startswith("siglip_featurizer."):
            return "vision_backbone.fused_featurizer." + key[len("siglip_featurizer.") :]
        raise KeyError(key)
    raise KeyError(group)
