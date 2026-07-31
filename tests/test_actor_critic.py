"""Tests for the shared-encoder actor-critic wrapper.

Engine-free: observations come from sampling the interface's own space (via the
shared ``sample_observation_batch`` helper), and the mask is a simple legal
prefix built inline - no dependency on the policy-head suite's RNG mask helper.
The gradient test proves the single shared encode is differentiable through both
heads at once.
"""

from __future__ import annotations

import torch

from sts_rl.agent.actor_critic import ActorCritic
from sts_rl.agent.encoder import HIDDEN_DIM
from sts_rl.interface import ACTION_BLOCK_BY_NAME, ACTION_DIM, PAD_ID
from conftest import sample_observation_batch

BATCH = 4
# Legal prefix width: mask[:, :K] True guarantees >=1 legal action per row, so
# every sampled action must land in [0, K).
K = 5


def _legal_prefix_mask(batch: int = BATCH, k: int = K) -> torch.Tensor:
    """Bool mask (B, ACTION_DIM) with the first ``k`` actions legal in every row."""
    mask = torch.zeros(batch, ACTION_DIM, dtype=torch.bool)
    mask[:, :k] = True
    return mask


def test_pad_slot_perturbation_does_not_change_masked_logits_or_value():
    """Perturbing a masked (PAD) slot's raw features changes neither the masked
    logits nor the value.

    The actor-critic face of the encoder's no-leakage guarantee. The value reads
    the pooled CLS context, which the padding mask makes PAD-invariant. The pointer
    head reads per-token embeddings, so the PAD slot's OWN raw logit does move - but
    that slot is masked illegal (an empty hand slot has no legal play), and every
    LEGAL slot's logit is PAD-invariant (the PAD token is excluded as an attention
    key, and no other token reads it), so the masked distribution is unchanged.
    """
    ac = ActorCritic()
    ac.eval()
    obs = sample_observation_batch(BATCH)
    obs["hand_ids"][:, 0] = PAD_ID  # force a PAD (masked) hand slot
    # END_TURN is a learned-query action that does not read the PAD hand token; the
    # PAD slot's own PLAY actions stay illegal, exactly as the env would mask them.
    mask = torch.zeros(BATCH, ACTION_DIM, dtype=torch.bool)
    mask[:, ACTION_BLOCK_BY_NAME["END_TURN"].start] = True
    perturbed = {name: t.clone() for name, t in obs.items()}
    perturbed["hand_feats"][:, 0, :] = 5.0  # perturb that PAD slot's raw features
    with torch.no_grad():
        per_token, pooled, kpm = ac.encoder(obs)
        per_token_p, pooled_p, kpm_p = ac.encoder(perturbed)
        logits = ac.policy(per_token, kpm, mask)
        logits_p = ac.policy(per_token_p, kpm_p, mask)
        value = ac.value(pooled)
        value_p = ac.value(pooled_p)
    assert torch.allclose(logits, logits_p, atol=1e-6)
    assert torch.allclose(value, value_p, atol=1e-6)


def test_act_returns_four_finite_batch_tensors():
    """act -> (action, log_prob, entropy, value), each (BATCH,) and finite."""
    ac = ActorCritic()
    outputs = ac.act(sample_observation_batch(BATCH), _legal_prefix_mask())
    assert len(outputs) == 4
    for tensor in outputs:
        assert tensor.shape == (BATCH,)
        assert torch.isfinite(tensor).all()


def test_sampled_actions_are_legal():
    """Sampled actions land inside the legal prefix [0, K)."""
    ac = ActorCritic()
    action, _, _, _ = ac.act(sample_observation_batch(BATCH), _legal_prefix_mask())
    assert (action < K).all()


def test_deterministic_act_is_reproducible_and_legal():
    """deterministic=True gives identical, legal actions across two calls."""
    ac = ActorCritic()
    obs = sample_observation_batch(BATCH)
    mask = _legal_prefix_mask()
    first, _, _, _ = ac.act(obs, mask, deterministic=True)
    second, _, _, _ = ac.act(obs, mask, deterministic=True)
    assert torch.equal(first, second)
    assert (first < K).all()


def test_evaluate_actions_is_consistent_with_value_head():
    """evaluate_actions -> four values; value matches the value head, aux_pred is None.

    Re-running the encoder + value head on the same obs must reproduce the value
    returned by evaluate_actions (eval mode: no dropout/randomness in the encoder).
    A default ActorCritic has no aux head, so aux_pred must be None.
    """
    ac = ActorCritic()
    ac.eval()
    obs = sample_observation_batch(BATCH)
    mask = _legal_prefix_mask()
    actions = torch.zeros(BATCH, dtype=torch.long)  # index 0 is legal under the prefix
    log_prob, entropy, value, aux_pred = ac.evaluate_actions(obs, mask, actions)
    for tensor in (log_prob, entropy, value):
        assert tensor.shape == (BATCH,)
    assert torch.isfinite(log_prob).all()
    assert aux_pred is None
    _per_token, pooled_cls, _mask = ac.encoder(obs)
    expected_value = ac.value(pooled_cls)
    assert torch.allclose(value, expected_value)


def test_gradient_reaches_encoder_and_both_heads():
    """A combined (log_prob + value) scalar backprops into the encoder and both heads.

    Proves the single shared encode is differentiable through the policy and
    value heads simultaneously. ``card_embed`` is the encoder's card id
    embedding (confirmed by reading encoder.py).
    """
    ac = ActorCritic()
    obs = sample_observation_batch(BATCH)
    mask = _legal_prefix_mask()
    actions = torch.zeros(BATCH, dtype=torch.long)
    log_prob, _, value, _aux = ac.evaluate_actions(obs, mask, actions)
    (log_prob.sum() + value.sum()).backward()

    encoder_grad = ac.encoder.card_embed.weight.grad
    assert encoder_grad is not None
    assert torch.isfinite(encoder_grad).all()
    assert ac.policy.learned_logit.weight.grad is not None
    assert ac.value.value.weight.grad is not None


def test_head_input_dims_wired_from_encoder():
    """Both heads' Linear in_features track the encoder output width, not a literal."""
    ac = ActorCritic()
    assert ac.policy.input_dim == ac.encoder.output_dim
    assert ac.value.value.in_features == ac.encoder.output_dim
    # Sanity: default construction uses the encoder's HIDDEN_DIM width.
    assert ac.encoder.output_dim == HIDDEN_DIM


def test_get_value_is_flat_finite_and_matches_value_head():
    """get_value returns (B,) finite values equal to the value head on encoded obs."""
    ac = ActorCritic()
    ac.eval()  # deterministic encode for exact equality
    obs = sample_observation_batch(BATCH)
    values = ac.get_value(obs)
    assert values.shape == (BATCH,)
    assert torch.isfinite(values).all()
    _per_token, pooled_cls, _mask = ac.encoder(obs)
    assert torch.allclose(values, ac.value(pooled_cls))
