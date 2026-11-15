from __future__ import annotations

from dataclasses import dataclass

# Exact weight identities. The notes are the contract the loader and the
# summary both rely on: a near-miss checkpoint is recorded under do_not_use
# and is never loaded.


@dataclass(frozen=True)
class ModelSpec:
    key: str
    role: str
    name: str
    family: str
    checkpoint: str
    prompt_style: str
    requires_image: bool
    loader: str
    notes: str
    license_note: str
    baseline_key: str | None = None
    matched_baseline_key: str | None = None
    architecture_checkpoint: str | None = None
    processor_checkpoint: str | None = None
    trust_remote_code: bool = False
    gated: bool = False
    prefer: tuple[str, ...] = ()
    reject: tuple[str, ...] = ()
    embed_fallback: bool = False
    rewrites: tuple[tuple[str, str], ...] = ()
    # (fused infix, ((part infix, rows), ...)): a donor tensor whose name holds the
    # fused infix is cut along dim 0 into one tensor per part before alignment.
    fused_splits: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = ()
    lm_head_fallback: bool = False
    truncate_layers: int | None = None
    truncate_to_donor: bool = False
    expected_text_layers: int | None = None
    do_not_use: tuple[str, ...] = ()
    ancestor_checkpoint: str | None = None
    prismatic_repo: str | None = None
    prismatic_file: str | None = None
    env: str = "main"
    donor_via_architecture: bool = False


# OpenVLA's remote modeling code checks for these exact versions and does not
# construct under transformers 5, so the Prismatic pair runs in envs/prismatic.
ENV_TRANSFORMERS = {"prismatic": "4.40.1"}
ENV_PROJECTS = {"prismatic": "envs/prismatic"}


def env_mismatch(spec: ModelSpec, transformers_version: str) -> str | None:
    """Why ``spec`` cannot run under this transformers version, or None."""

    pinned = ENV_TRANSFORMERS.get(spec.env)
    if pinned is not None and transformers_version != pinned:
        return (
            f"needs transformers=={pinned}; run "
            f"`uv run --project {ENV_PROJECTS[spec.env]} python -m vla_vlm_benchmarks --model {spec.key}`"
        )
    if pinned is None and transformers_version in ENV_TRANSFORMERS.values():
        return "needs the main env; run `uv run python -m vla_vlm_benchmarks` from the experiment root"
    return None


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark run the way its reference harness runs it.

    ``scoring`` is ``pope``, ``vqa``, ``choice`` (generate, then extract the
    option letter), ``loglikelihood`` (lm-eval cloze), or ``gsm8k``.
    ``multiple_choice`` marks the ones whose charts get a chance line.
    ``secondary_metric`` is reported beside the primary one in the summary.
    """

    key: str
    title: str
    difficulty: str
    scoring: str
    dataset: str
    dataset_config: str | None
    split_candidates: tuple[str, ...]
    default_limit: int
    max_new_tokens: int
    primary_metric: str
    convention: str
    why: str
    multiple_choice: bool = False
    categories: tuple[str, ...] = ()
    category_column: str | None = None
    stop: tuple[str, ...] = ()
    single_image_only: bool = False
    decode_images_during_scan: bool = False
    has_images: bool = True
    secondary_metric: str | None = None


PALIGEMMA_NEAR_MISS = (
    "google/paligemma-3b-mix-224",
    "google/paligemma2-3b-pt-224",
    "google/gemma-2b",
    "google/siglip-so400m-patch14-224",
)

MODELS: dict[str, ModelSpec] = {
    "paligemma_3b_pt224": ModelSpec(
        key="paligemma_3b_pt224",
        role="baseline",
        name="PaliGemma 3B pt-224",
        family="paligemma",
        checkpoint="google/paligemma-3b-pt-224",
        prompt_style="paligemma_pt",
        requires_image=True,
        loader="hf",
        gated=True,
        expected_text_layers=18,
        do_not_use=PALIGEMMA_NEAR_MISS,
        license_note="Gemma license. Gated on the Hub.",
        notes=(
            "Init checkpoint for π₀, π₀-FAST, and π₀.₅. OpenPI's "
            "PaliGemmaWeightLoader reads gs://vertex-model-garden-paligemma-us/"
            "paligemma/pt_224.npz; this repo is the official PyTorch conversion "
            "of that file (SigLIP-So400m/14 at 224 + Gemma 2B). The mix checkpoint "
            "is a later multitask finetune, and PaliGemma 2 is a Gemma 2 model."
        ),
    ),
    "qwen3_5_2b_base": ModelSpec(
        key="qwen3_5_2b_base",
        role="baseline",
        name="Qwen3.5-2B-Base",
        family="qwen3_5",
        checkpoint="Qwen/Qwen3.5-2B-Base",
        prompt_style="qwen_base",
        requires_image=False,
        loader="hf",
        expected_text_layers=24,
        do_not_use=("Qwen/Qwen3.5-2B",),
        license_note="Apache 2.0.",
        notes=(
            "Init checkpoint for G0.5. The released config points at "
            "qwen3_5_2b_base_processor and matches this model: vocab 248320, "
            "24 layers, hidden size 2048, the 3× linear-attention + 1× full-attention "
            "block. Qwen/Qwen3.5-2B is the post-trained instruct model."
        ),
    ),
    "cosmos_reason2_2b": ModelSpec(
        key="cosmos_reason2_2b",
        role="baseline",
        name="Cosmos-Reason2-2B",
        family="cosmos",
        checkpoint="nvidia/Cosmos-Reason2-2B",
        prompt_style="qwen_chat",
        requires_image=False,
        loader="hf",
        gated=True,
        expected_text_layers=28,
        ancestor_checkpoint="Qwen/Qwen3-VL-2B-Instruct",
        do_not_use=("Qwen/Qwen3-VL-2B-Instruct",),
        license_note="NVIDIA Open Model License. Gated on the Hub.",
        notes=(
            "VLM that GR00T N1.7 loads (vlm_model_path). It is a Qwen3-VL-2B "
            "post-train, so Qwen/Qwen3-VL-2B-Instruct is the ancestor, not the "
            "weights the VLA started from."
        ),
    ),
    "cosmos_reason2_2b_layer16": ModelSpec(
        key="cosmos_reason2_2b_layer16",
        role="baseline_probe",
        name="Cosmos-Reason2-2B, first 16 text layers",
        family="cosmos",
        checkpoint="nvidia/Cosmos-Reason2-2B",
        prompt_style="qwen_chat",
        requires_image=False,
        loader="truncate",
        gated=True,
        truncate_layers=16,
        expected_text_layers=16,
        baseline_key="cosmos_reason2_2b",
        matched_baseline_key="cosmos_reason2_2b",
        ancestor_checkpoint="Qwen/Qwen3-VL-2B-Instruct",
        license_note="NVIDIA Open Model License. Gated on the Hub.",
        notes=(
            "Same weights as Cosmos-Reason2-2B with language_model.layers "
            "truncated to 16, which is GR00T N1.7's select_layer. The original "
            "final norm and lm_head stay. This is the matched control for the "
            "GR00T checkpoint, which does not contain the other 12 text layers. "
            "Scores are not comparable to the full Cosmos model."
        ),
    ),
    "qwen2_5_vl_3b_instruct": ModelSpec(
        key="qwen2_5_vl_3b_instruct",
        role="baseline",
        name="Qwen2.5-VL-3B-Instruct",
        family="qwen2_5_vl",
        checkpoint="Qwen/Qwen2.5-VL-3B-Instruct",
        prompt_style="qwen_chat",
        requires_image=False,
        loader="hf",
        expected_text_layers=36,
        license_note="Apache 2.0.",
        notes=(
            "Init checkpoint for WALL-OSS and WALL-OSS-0.5. The papers say "
            "Qwen2.5-VL-3B; the only 3B checkpoint on the Hub is the Instruct "
            "model, and the WALL-OSS-0.5 release names it explicitly."
        ),
    ),
    "molmo2_er": ModelSpec(
        key="molmo2_er",
        role="baseline",
        name="Molmo2-ER",
        family="molmo",
        checkpoint="allenai/Molmo2-ER",
        prompt_style="molmo",
        requires_image=False,
        loader="hf",
        trust_remote_code=False,
        expected_text_layers=36,
        ancestor_checkpoint="https://storage.googleapis.com/oe-training-public/Molmo2-1225/Molmo2-4B.tar",
        do_not_use=(
            "allenai/Molmo-7B-D-0924",
            "allenai/Molmo2-8B",
            "allenai/Molmo2-4B",
            "Qwen/Qwen3-4B",
        ),
        license_note="Apache 2.0. trust_remote_code executes the checkpoint's modeling code.",
        notes=(
            "Embodied-reasoning VLM that MolmoAct2 pretraining starts from. "
            "Molmo2 architecture: Qwen3-4B geometry (36 layers, hidden 2560, "
            "32 query heads, 8 KV heads) plus a SigLIP2 vision encoder at 378px. "
            "The original Molmo-7B, Molmo2-8B, and a bare Qwen3-4B are different "
            "models. Molmo2-ER itself was trained from the Molmo2-4B archive."
        ),
    ),
    "prism_dinosiglip_224px_7b": ModelSpec(
        key="prism_dinosiglip_224px_7b",
        role="baseline",
        name="Prismatic DINO+SigLIP 224px + Llama-2 7B",
        family="prismatic",
        checkpoint="TRI-ML/prismatic-vlms",
        architecture_checkpoint="openvla/openvla-7b",
        processor_checkpoint="openvla/openvla-7b",
        prompt_style="prismatic_pure",
        requires_image=True,
        loader="prismatic",
        trust_remote_code=True,
        gated=False,
        expected_text_layers=32,
        prismatic_repo="TRI-ML/prismatic-vlms",
        prismatic_file="prism-dinosiglip-224px+7b/checkpoints/latest-checkpoint.pt",
        env="prismatic",
        do_not_use=(
            "meta-llama/Llama-2-7b-chat-hf",
            "TRI-ML/prismatic-vlms/prism-dinosiglip+7b",
            "TRI-ML/prismatic-vlms/prism-dinosiglip-224px-controlled+7b",
        ),
        license_note="Prismatic weights from TRI-ML. The Llama-2 7B inside them is under the Llama 2 Community License.",
        notes=(
            "OpenVLA's base_vlm, from the training config: prism-dinosiglip-224px+7b. "
            "LLM is llama2-7b-pure (meta-llama/Llama-2-7b-hf), not Llama-2-Chat. "
            "Vision is fused DINOv2 ViT-L/14 (vit_large_patch14_reg4_dinov2.lvd142m) "
            "and SigLIP So400m/14 (vit_so400m_patch14_siglip_224) at 224px. "
            "The 384px and controlled Prismatic runs are different checkpoints. "
            "The checkpoint stores only the projector and Llama-2 (32064 rows, "
            "same as OpenVLA); the vision towers were frozen in Prismatic training "
            "and are loaded from timm, as PrismaticVLM.from_pretrained does."
        ),
    ),
    "pi0": ModelSpec(
        key="pi0",
        role="vla",
        name="π₀ base",
        family="paligemma",
        checkpoint="lerobot/pi0_base",
        prompt_style="paligemma_pt",
        requires_image=True,
        loader="vla_weights",
        baseline_key="paligemma_3b_pt224",
        matched_baseline_key="paligemma_3b_pt224",
        gated=True,
        embed_fallback=True,
        rewrites=((".vision_tower.vision_model.", ".vision_tower."),),
        prefer=("paligemma_with_expert.paligemma",),
        reject=("gemma_expert", "action_in_proj", "action_out_proj", "action_time", "state_proj"),
        do_not_use=PALIGEMMA_NEAR_MISS,
        license_note="Apache 2.0 checkpoint. PaliGemma weights inside it follow the Gemma license.",
        notes=(
            "LeRobot conversion of the π₀ base policy. Flow matching. The Gemma "
            "300M action expert is a separate stack and is not part of this eval; "
            "only paligemma_with_expert.paligemma is copied onto the pt-224 model."
        ),
    ),
    "pi05": ModelSpec(
        key="pi05",
        role="vla",
        name="π₀.₅ base",
        family="paligemma",
        checkpoint="lerobot/pi05_base",
        prompt_style="paligemma_pt",
        requires_image=True,
        loader="vla_weights",
        baseline_key="paligemma_3b_pt224",
        matched_baseline_key="paligemma_3b_pt224",
        gated=True,
        embed_fallback=True,
        rewrites=((".vision_tower.vision_model.", ".vision_tower."),),
        prefer=("paligemma_with_expert.paligemma",),
        reject=("gemma_expert", "action_in_proj", "action_out_proj", "action_time", "state_proj", "time_mlp"),
        do_not_use=PALIGEMMA_NEAR_MISS,
        license_note="Apache 2.0 checkpoint. PaliGemma weights inside it follow the Gemma license.",
        notes=(
            "LeRobot conversion of the π₀.₅ base policy. Same PaliGemma init as π₀. "
            "π₀.₅ feeds discretized state as text and puts adaRMS on the action "
            "expert only (use_adarms=[False, True]); the VLM stack is still the "
            "pt-224 PaliGemma. There is no released π₀.₅-FAST base checkpoint."
        ),
    ),
    "pi0_fast": ModelSpec(
        key="pi0_fast",
        role="vla",
        name="π₀-FAST base",
        family="paligemma",
        checkpoint="lerobot/pi0fast-base",
        prompt_style="paligemma_pt",
        requires_image=True,
        loader="vla_weights",
        baseline_key="paligemma_3b_pt224",
        matched_baseline_key="paligemma_3b_pt224",
        gated=True,
        embed_fallback=True,
        rewrites=((".vision_tower.vision_model.", ".vision_tower."),),
        prefer=("paligemma",),
        reject=("gemma_expert", "action_in_proj", "action_out_proj"),
        do_not_use=PALIGEMMA_NEAR_MISS,
        license_note="Apache 2.0 checkpoint. PaliGemma weights inside it follow the Gemma license.",
        notes=(
            "The checkpoint on the LeRobot π₀-FAST page. Its PaliGemma tensors "
            "(603 keys) are bit-identical to lerobot/pi05_base, so on this "
            "suite it reproduces the π₀.₅ numbers rather than measuring a "
            "separately trained FAST VLM. No action expert is stored."
        ),
    ),
    "g05": ModelSpec(
        key="g05",
        role="vla",
        name="Galaxea G0.5 base",
        family="qwen3_5",
        checkpoint="lerobot/g05_base",
        prompt_style="qwen_base",
        requires_image=False,
        loader="vla_weights",
        baseline_key="qwen3_5_2b_base",
        matched_baseline_key="qwen3_5_2b_base",
        gated=True,
        embed_fallback=True,
        rewrites=(
            ("backend.model.vision_tower.", "model.visual."),
            ("backend.model.vlm.input_proj.", "model.language_model.embed_tokens."),
            ("backend.model.vlm.", "model.language_model."),
        ),
        prefer=("backend.model.vlm", "backend.model.vision_tower"),
        reject=("action_expert", "action_in_proj", "action_out_proj", "proprio"),
        do_not_use=("Qwen/Qwen3.5-2B",),
        license_note="G0.5 Community License, non-commercial. Gated on the Hub.",
        notes=(
            "LeRobot conversion of G0.5 base. The VLM is Qwen3.5-2B-Base "
            "(hidden 2048); the action expert is a separate width-1024 stack "
            "and is dropped by the shape check. G0.5's own prompt has no "
            "free-form question slot, so this eval uses the base VLM processor "
            "on the copied backbone weights."
        ),
    ),
    "groot_n17": ModelSpec(
        key="groot_n17",
        role="vla",
        name="GR00T N1.7 3B",
        family="cosmos",
        checkpoint="nvidia/GR00T-N1.7-3B",
        prompt_style="qwen_chat",
        requires_image=False,
        loader="vla_weights",
        baseline_key="cosmos_reason2_2b",
        matched_baseline_key="cosmos_reason2_2b_layer16",
        gated=True,
        truncate_to_donor=True,
        expected_text_layers=16,
        lm_head_fallback=True,
        prefer=("backbone",),
        reject=("action_head", "action_encoder", "vl_self_attention"),
        ancestor_checkpoint="Qwen/Qwen3-VL-2B-Instruct",
        do_not_use=("Qwen/Qwen3-VL-2B-Instruct",),
        license_note="NVIDIA Open Model License. Cosmos-Reason2-2B, which this checkpoint loads, is gated.",
        notes=(
            "GR00T N1.7 keeps 16 of Cosmos-Reason2-2B's 28 text layers "
            "(select_layer=16) and reads that hidden state from the flow-matching "
            "action head. The eval truncates the Cosmos architecture to however "
            "many text layers the checkpoint actually contains, copies those "
            "weights, and keeps the Cosmos lm_head when the checkpoint has none. "
            "Compare against cosmos_reason2_2b_layer16, not the full 28-layer model."
        ),
    ),
    "wall_oss": ModelSpec(
        key="wall_oss",
        role="vla",
        name="WALL-OSS 0.5",
        family="qwen2_5_vl",
        checkpoint="x-square-robot/wall-oss-0.5",
        prompt_style="qwen_chat",
        requires_image=False,
        loader="vla_weights",
        baseline_key="qwen2_5_vl_3b_instruct",
        matched_baseline_key="qwen2_5_vl_3b_instruct",
        embed_fallback=True,
        fused_splits=(
            (
                ".self_attn.qkv_proj_experts.0.",
                ((".self_attn.q_proj.", 2048), (".self_attn.k_proj.", 256), (".self_attn.v_proj.", 256)),
            ),
            (".moe.experts.0.gate_up_proj.", ((".mlp.gate_proj.", 11008), (".mlp.up_proj.", 11008))),
            (".mlp.gate_up_proj.", ((".mlp.gate_proj.", 3420), (".mlp.up_proj.", 3420))),
        ),
        rewrites=(
            (".input_layernorms.0.", ".input_layernorm."),
            (".post_attention_layernorms.0.", ".post_attention_layernorm."),
            (".self_attn.o_proj_experts.0.", ".self_attn.o_proj."),
            (".moe.experts.0.", ".mlp."),
            ("model.norms.0.", "model.norm."),
            ("model.layers.", "model.language_model.layers."),
            ("model.embed_tokens.", "model.language_model.embed_tokens."),
            ("model.norm.", "model.language_model.norm."),
            ("visual.", "model.visual."),
        ),
        prefer=("model.language_model", "model.visual", "model.layers", "visual."),
        reject=(
            "action_expert", "action_proj", "action_in_proj", "action_out_proj", "propri",
            "action_preprocessor", ".moe.experts.1.", "_experts.1.", "layernorms.1.", "norms.1.",
        ),
        license_note="No license on the model card; the wall-x code is Apache 2.0.",
        notes=(
            "WALL-OSS 0.5, the May 2026 release (4B: the Qwen2.5-VL-3B backbone "
            "plus action components), co-trained with a multimodal objective; its "
            "card claims no erosion of vision-language ability. Every layer has "
            "two experts with separate attention projections, norms and FFNs: "
            "expert 0 has Qwen's widths and is the VL expert copied onto "
            "Qwen2.5-VL-3B-Instruct, with its fused qkv (q 2048, k 256, v 256 "
            "rows, as WallOSS05Attention reads them) and fused gate_up (chunked "
            "in half, also in the vision blocks) split first; the q and k bias "
            "slices match Qwen's exactly, which pins the boundaries. Expert 1 "
            "(width 1024) is the action expert "
            "and is excluded. The original x-square-robot/wall-oss-flow (Sept "
            "2025) stores per-expert FFNs only and was replaced by this release."
        ),
    ),
    "molmoact2": ModelSpec(
        key="molmoact2",
        role="vla",
        name="MolmoAct2",
        family="molmo",
        checkpoint="allenai/MolmoAct2",
        prompt_style="molmo",
        requires_image=False,
        loader="vla_weights",
        trust_remote_code=True,
        baseline_key="molmo2_er",
        matched_baseline_key="molmo2_er",
        expected_text_layers=36,
        donor_via_architecture=True,
        reject=("action_expert",),
        ancestor_checkpoint="https://storage.googleapis.com/oe-training-public/Molmo2-1225/Molmo2-4B.tar",
        do_not_use=(
            "allenai/Molmo-7B-D-0924",
            "allenai/MolmoAct-7B-D-0812",
            "allenai/Molmo2-8B",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen2.5-7B",
        ),
        license_note="Apache 2.0. trust_remote_code executes the checkpoint's modeling code.",
        notes=(
            "Released MolmoAct2 policy, the checkpoint the LeRobot docs fine-tune. "
            "Pretraining starts from allenai/Molmo2-ER. The flow-matching action "
            "expert is a separate width-768 stack with one block per text layer "
            "and is excluded. Both checkpoints use the original Molmo2 layout, "
            "which transformers converts while loading (fused att_proj/ff_proj "
            "split, vision renamed), so MolmoAct2 is loaded through the same "
            "native Molmo2 class and copied name for name. Its base vocabulary "
            "is 154624 against Molmo2-ER's 151936; the shared Qwen3 prefix and "
            "the 128 image/frame special-token rows (same offsets in both "
            "configs) are copied, and its extra action/state rows are dropped. "
            "MolmoAct-7B-D is the previous MolmoAct (Qwen2.5-7B), not this model."
        ),
    ),
    "openvla": ModelSpec(
        key="openvla",
        role="vla",
        name="OpenVLA 7B",
        family="prismatic",
        checkpoint="openvla/openvla-7b",
        prompt_style="prismatic_pure",
        requires_image=True,
        loader="openvla",
        trust_remote_code=True,
        baseline_key="prism_dinosiglip_224px_7b",
        matched_baseline_key="prism_dinosiglip_224px_7b",
        processor_checkpoint="openvla/openvla-7b",
        env="prismatic",
        do_not_use=(
            "meta-llama/Llama-2-7b-chat-hf",
            "TRI-ML/prismatic-vlms/prism-dinosiglip+7b",
        ),
        license_note="MIT. trust_remote_code executes the checkpoint's modeling code.",
        notes=(
            "OXE-pretrained OpenVLA. The HF checkpoint already includes the "
            "Llama-2 backbone, the fused DINO+SigLIP towers, and the projector. "
            "Its 256 action bins reuse the last 256 Llama tokens. Those stay "
            "available: emitting them instead of an answer is a real outcome "
            "of VLA training and is scored as wrong."
        ),
    ),
}


# Extra tokens a reply that opens with "<think>" may spend before "</think>".
# The benchmark's max_new_tokens then applies to the answer after it, as
# VLMEvalKit and lmms-eval budget reasoning models.
THINK_BUDGET = 1024


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "pope": BenchmarkSpec(
        key="pope",
        title="POPE",
        difficulty="easy",
        scoring="pope",
        dataset="lmms-lab/POPE",
        dataset_config=None,
        split_candidates=("test", "validation"),
        default_limit=300,
        max_new_tokens=16,
        primary_metric="f1",
        convention=(
            "LLaVA / lmms-eval prompt: '{question}\\nAnswer the question using a single word or "
            "phrase.', greedy; VLMEvalKit YOrN_Extraction (unparsed is wrong and not-yes); F1 with "
            "yes as positive, averaged over random, popular and adversarial (POPE paper)"
        ),
        why=(
            "Yes/no object hallucination. Accuracy and yes-rate are stored next to F1 "
            "because an always-yes model already reaches F1 0.667 on balanced splits."
        ),
        multiple_choice=True,
        categories=("random", "popular", "adversarial"),
        category_column="category",
    ),
    "vqav2": BenchmarkSpec(
        key="vqav2",
        title="VQAv2",
        difficulty="easy-medium",
        scoring="vqa",
        dataset="lmms-lab/VQAv2",
        dataset_config=None,
        split_candidates=("validation", "val", "test"),
        default_limit=500,
        max_new_tokens=16,
        primary_metric="vqa_accuracy",
        convention=(
            "LLaVA / lmms-eval: '{question}\\nAnswer the question using a single word or phrase.', "
            "greedy; official VQA accuracy with the EvalAI answer processor"
        ),
        why="Standard short-answer visual questions.",
        secondary_metric="vqa_accuracy_contains",
    ),
    "textvqa": BenchmarkSpec(
        key="textvqa",
        title="TextVQA",
        difficulty="medium",
        scoring="vqa",
        dataset="lmms-lab/textvqa",
        dataset_config=None,
        split_candidates=("validation", "val", "test"),
        default_limit=500,
        max_new_tokens=16,
        primary_metric="vqa_accuracy",
        convention=(
            "lmms-eval textvqa_val without OCR tokens: '{question}\\nAnswer the question using a "
            "single word or phrase.', greedy; VQA accuracy with the EvalAI answer processor"
        ),
        why="Reading text in photos. Harder than VQAv2, still short answers.",
        secondary_metric="vqa_accuracy_contains",
    ),
    "realworldqa": BenchmarkSpec(
        key="realworldqa",
        title="RealWorldQA",
        difficulty="medium",
        scoring="choice",
        dataset="lmms-lab/RealWorldQA",
        dataset_config=None,
        split_candidates=("test", "validation"),
        default_limit=0,
        max_new_tokens=16,
        primary_metric="accuracy",
        convention=(
            "question used verbatim (it carries its options and answer instruction), greedy; "
            "lettered items scored by VLMEvalKit can_infer exact matching, open items by "
            "normalized exact match"
        ),
        why="Real photographs, mostly multiple choice. Run in full (765 items).",
        multiple_choice=True,
    ),
    "ai2d": BenchmarkSpec(
        key="ai2d",
        title="AI2D",
        difficulty="medium",
        scoring="choice",
        dataset="lmms-lab/ai2d",
        dataset_config=None,
        split_candidates=("test", "validation"),
        default_limit=500,
        max_new_tokens=16,
        primary_metric="accuracy",
        convention=(
            "LLaVA / lmms-eval: '{question}\\nA. ...\\nAnswer with the option's letter from the given "
            "choices directly.', greedy; VLMEvalKit can_infer exact matching, unparsed is wrong"
        ),
        why="Science diagrams, multiple choice. A step up from natural photos.",
        multiple_choice=True,
    ),
    "erqa": BenchmarkSpec(
        key="erqa",
        title="ERQA single-image",
        difficulty="medium",
        scoring="choice",
        dataset="FlagEval/ERQA",
        dataset_config=None,
        split_candidates=("test", "train", "validation"),
        default_limit=0,
        max_new_tokens=16,
        primary_metric="accuracy",
        convention=(
            "question used verbatim (it carries 'Choices: A. ...' and the answer instruction), "
            "greedy; VLMEvalKit can_infer exact matching, unparsed is wrong"
        ),
        why=(
            "Embodied Reasoning QA: pointing, trajectory, and action choices on "
            "robot scenes. Multi-image items are dropped so OpenVLA, which has one "
            "image slot, sees the same set as the others."
        ),
        multiple_choice=True,
        single_image_only=True,
        decode_images_during_scan=True,
    ),
    "arc_easy": BenchmarkSpec(
        key="arc_easy",
        title="ARC-Easy",
        difficulty="easy",
        scoring="loglikelihood",
        dataset="allenai/ai2_arc",
        dataset_config="ARC-Easy",
        split_candidates=("test",),
        default_limit=0,
        max_new_tokens=0,
        primary_metric="acc",
        convention=(
            "lm-evaluation-harness arc_easy, 0-shot: raw 'Question: {question}\\nAnswer:' and each "
            "choice text as a continuation; argmax of summed log-probability (acc) and of "
            "log-probability per character (acc_norm)"
        ),
        why=(
            "Grade-school science questions, text only, run in full (2,376 items). "
            "Easy enough that a 2B–7B model which still speaks lands above chance."
        ),
        multiple_choice=True,
        has_images=False,
    ),
    "gsm8k": BenchmarkSpec(
        key="gsm8k",
        title="GSM8K",
        difficulty="medium-hard",
        scoring="gsm8k",
        dataset="openai/gsm8k",
        dataset_config="main",
        split_candidates=("test",),
        default_limit=250,
        max_new_tokens=256,
        primary_metric="exact_match_flexible",
        secondary_metric="exact_match_strict",
        convention=(
            "lm-evaluation-harness gsm8k_cot: the 8 Wei et al. exemplars as 'Q: ...\\nA: ...', "
            "greedy until 'Q:'; flexible-extract (last number) primary, strict-match "
            "('The answer is N') stored"
        ),
        why=(
            "Grade-school math with a single final number. This is the hard end "
            "of the suite. HumanEval is a poor fit: these 2B–3B backbones, and "
            "PaliGemma-pt in particular, almost never emit runnable code, so both "
            "sides sit at zero."
        ),
        stop=("Q:", "</s>", "<|im_end|>"),
        has_images=False,
    ),
}

SUITES = {
    "baselines": tuple(key for key, spec in MODELS.items() if spec.role in {"baseline", "baseline_probe"}),
    "vlas": tuple(key for key, spec in MODELS.items() if spec.role == "vla"),
    "all": tuple(MODELS),
}


def matched_baseline_key(spec: ModelSpec) -> str | None:
    if spec.role == "baseline":
        return None
    return spec.matched_baseline_key or spec.baseline_key


def architecture_checkpoint(spec: ModelSpec) -> str:
    if spec.architecture_checkpoint:
        return spec.architecture_checkpoint
    if spec.loader in {"hf", "truncate", "openvla"}:
        return spec.checkpoint
    if spec.baseline_key is None:
        raise KeyError(spec.key)
    return MODELS[spec.baseline_key].checkpoint


def processor_checkpoint(spec: ModelSpec) -> str:
    if spec.processor_checkpoint:
        return spec.processor_checkpoint
    return architecture_checkpoint(spec)


def format_catalog() -> str:
    lines = ["models"]
    for spec in MODELS.values():
        baseline = matched_baseline_key(spec)
        extra = f"  matched={baseline}" if baseline else ""
        lines.append(f"  {spec.key:28} {spec.role:16} {spec.checkpoint}{extra}")
    lines.append("benchmarks")
    for spec in BENCHMARKS.values():
        limit = "all" if spec.default_limit <= 0 else str(spec.default_limit)
        lines.append(
            f"  {spec.key:16} {spec.difficulty:14} limit={limit:4} primary={spec.primary_metric}"
        )
    return "\n".join(lines)
