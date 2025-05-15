def compute_policy_gradient_full_correction(
    action_distrib, action_distrib_mu, action_value, v, truncation_threshold
):
    """Compute off-policy bias correction term wrt all actions."""
    assert isinstance(action_distrib, torch.distributions.Categorical)
    assert isinstance(action_distrib_mu, torch.distributions.Categorical)
    assert truncation_threshold is not None
    assert np.isscalar(v)
    with torch.no_grad():
        rho_all_inv = compute_full_importance(action_distrib_mu, action_distrib)
        correction_weight = (
            torch.nn.functional.relu(1 - truncation_threshold * rho_all_inv)
            * action_distrib.probs[0]
        )
        correction_advantage = action_value.q_values[0] - v
    # Categorical.logits is already normalized, i.e., logits[i] = log(probs[i])
    return -(correction_weight * action_distrib.logits * correction_advantage).sum(1)