def maximum_weight_matching(logits):
    from scipy.optimize import linear_sum_assignment

    cost = -logits.cpu()
    cost = torch.cat([cost, cost], dim=-1)  # Duplicate destinations.
    value = linear_sum_assignment(cost.numpy())[1]
    value = torch.tensor(value, dtype=torch.long, device=logits.device)
    value %= logits.size(1)
    return value