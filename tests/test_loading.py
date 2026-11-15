import pytest
import torch

from vla_vlm_benchmarks.errors import EvalSetupError
from vla_vlm_benchmarks.loading import _copy_donor, _MemoryDonor
from vla_vlm_benchmarks.registry import MODELS, ModelSpec, env_mismatch


class _TinyLM(torch.nn.Module):
    def __init__(self, rows: int):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(rows, 4)
        self.lm_head = torch.nn.Linear(4, rows, bias=False)


def _spec() -> ModelSpec:
    return ModelSpec(
        key="tiny",
        role="vla",
        name="tiny",
        family="tiny",
        checkpoint="tiny",
        prompt_style="raw",
        requires_image=False,
        loader="vla_weights",
        notes="",
        license_note="",
    )


def _donor(rows: int) -> _MemoryDonor:
    return _MemoryDonor(
        {
            "vla.model.embed_tokens.weight": torch.full((rows, 4), 2.0),
            "vla.lm_head.weight": torch.full((rows, 4), 3.0),
        }
    )


def test_output_rows_past_the_donor_vocab_cannot_be_predicted():
    model = _TinyLM(6)
    baseline_embed = model.model.embed_tokens.weight.detach().clone()
    coverage = _copy_donor(model, _donor(4), _spec(), [], pretrained_architecture=True)
    assert torch.equal(model.model.embed_tokens.weight[:4], torch.full((4, 4), 2.0))
    assert torch.equal(model.model.embed_tokens.weight[4:], baseline_embed[4:])
    with torch.no_grad():
        logits = model.lm_head(-torch.ones(2, 4))
    assert torch.equal(logits[:, :4], torch.full((2, 4), -12.0))
    assert torch.isneginf(logits[:, 4:]).all()
    assert any("masked 2 logits" in note for note in coverage["vocab"])


def test_rows_past_the_donor_vocab_are_refused_on_a_random_architecture():
    with pytest.raises(EvalSetupError, match="randomly initialized"):
        _copy_donor(_TinyLM(6), _donor(4), _spec(), [], pretrained_architecture=False)


def test_the_prismatic_pair_only_runs_under_its_pinned_transformers():
    assert "envs/prismatic" in env_mismatch(MODELS["openvla"], "5.23.0")
    assert env_mismatch(MODELS["openvla"], "4.40.1") is None
    assert env_mismatch(MODELS["prism_dinosiglip_224px_7b"], "4.40.1") is None
    assert "main env" in env_mismatch(MODELS["pi05"], "4.40.1")
    assert env_mismatch(MODELS["pi05"], "5.23.0") is None
