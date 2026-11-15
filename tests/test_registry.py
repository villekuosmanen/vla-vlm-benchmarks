from vla_vlm_benchmarks.registry import (
    BENCHMARKS,
    MODELS,
    architecture_checkpoint,
    matched_baseline_key,
)


def test_eight_benchmarks_cover_easy_and_hard():
    assert len(BENCHMARKS) == 8
    difficulties = {spec.difficulty for spec in BENCHMARKS.values()}
    assert "easy" in difficulties
    assert "medium-hard" in difficulties
    assert BENCHMARKS["erqa"].single_image_only
    assert BENCHMARKS["gsm8k"].primary_metric == "exact_match_flexible"
    assert BENCHMARKS["gsm8k"].stop[0] == "Q:"
    assert BENCHMARKS["pope"].primary_metric == "f1"
    assert BENCHMARKS["pope"].categories == ("random", "popular", "adversarial")
    assert BENCHMARKS["arc_easy"].scoring == "loglikelihood"
    assert BENCHMARKS["arc_easy"].has_images is False
    for spec in BENCHMARKS.values():
        assert spec.convention
    multiple_choice = {key for key, spec in BENCHMARKS.items() if spec.multiple_choice}
    assert multiple_choice == {"pope", "realworldqa", "ai2d", "erqa", "arc_easy"}


def test_vla_inits_are_the_exact_public_checkpoints():
    paligemma = "google/paligemma-3b-pt-224"
    assert MODELS["paligemma_3b_pt224"].checkpoint == paligemma
    for key in ("pi0", "pi05", "pi0_fast"):
        spec = MODELS[key]
        assert spec.baseline_key == "paligemma_3b_pt224"
        assert matched_baseline_key(spec) == "paligemma_3b_pt224"
        assert architecture_checkpoint(spec) == paligemma
        assert "google/paligemma-3b-mix-224" in spec.do_not_use
        assert "google/paligemma2-3b-pt-224" in spec.do_not_use
    assert MODELS["pi0"].checkpoint == "lerobot/pi0_base"
    assert MODELS["pi05"].checkpoint == "lerobot/pi05_base"
    assert MODELS["pi0_fast"].checkpoint == "lerobot/pi0fast-base"

    assert MODELS["g05"].checkpoint == "lerobot/g05_base"
    assert architecture_checkpoint(MODELS["g05"]) == "Qwen/Qwen3.5-2B-Base"
    assert "Qwen/Qwen3.5-2B" in MODELS["g05"].do_not_use

    assert MODELS["wall_oss"].checkpoint == "x-square-robot/wall-oss-0.5"
    assert architecture_checkpoint(MODELS["wall_oss"]) == "Qwen/Qwen2.5-VL-3B-Instruct"

    molmo = MODELS["molmoact2"]
    assert molmo.checkpoint == "allenai/MolmoAct2"
    assert architecture_checkpoint(molmo) == "allenai/Molmo2-ER"
    assert matched_baseline_key(molmo) == "molmo2_er"
    assert molmo.trust_remote_code
    assert "allenai/Molmo-7B-D-0924" in molmo.do_not_use
    assert "allenai/MolmoAct-7B-D-0812" in molmo.do_not_use
    assert "Qwen/Qwen3-4B" in molmo.do_not_use
    assert MODELS["molmo2_er"].checkpoint == "allenai/Molmo2-ER"
    assert MODELS["molmo2_er"].expected_text_layers == 36
    assert MODELS["molmo2_er"].prompt_style == "molmo"

    groot = MODELS["groot_n17"]
    assert groot.checkpoint == "nvidia/GR00T-N1.7-3B"
    assert matched_baseline_key(groot) == "cosmos_reason2_2b_layer16"
    assert architecture_checkpoint(groot) == "nvidia/Cosmos-Reason2-2B"
    assert groot.ancestor_checkpoint == "Qwen/Qwen3-VL-2B-Instruct"
    assert groot.truncate_to_donor and groot.lm_head_fallback
    assert MODELS["cosmos_reason2_2b_layer16"].truncate_layers == 16
    assert "Qwen/Qwen3-VL-2B-Instruct" not in {spec.checkpoint for spec in MODELS.values()}

    prism = MODELS["prism_dinosiglip_224px_7b"]
    assert prism.prismatic_file == "prism-dinosiglip-224px+7b/checkpoints/latest-checkpoint.pt"
    assert "meta-llama/Llama-2-7b-chat-hf" in prism.do_not_use
    assert "384" not in prism.prismatic_file
    openvla = MODELS["openvla"]
    assert openvla.checkpoint == "openvla/openvla-7b"
    assert matched_baseline_key(openvla) == "prism_dinosiglip_224px_7b"
    assert openvla.trust_remote_code

    loaded = {spec.checkpoint for spec in MODELS.values()}
    for spec in MODELS.values():
        for rejected in spec.do_not_use:
            assert rejected not in loaded
