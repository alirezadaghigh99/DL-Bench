def binary_focal_loss_with_logits(
    pred: Tensor,
    target: Tensor,
    alpha: Optional[float] = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
    pos_weight: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    ignore_index: Optional[int] = -100,
) -> Tensor:
    r"""Criterion that computes Binary Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        pred: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with the same shape as pred :math:`(N, C, *)`
          where each value is between 0 and 1.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        pos_weight: a weight of positive examples with shape :math:`(num\_of\_classes,)`.
          It is possible to trade off recall and precision by adding weights to positive examples.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.
        ignore_index: labels with this value are ignored in the loss computation.

    Returns:
        the computed loss.

    Examples:
        >>> C = 3  # num_classes
        >>> pred = torch.randn(1, C, 5, requires_grad=True)
        >>> target = torch.randint(2, (1, C, 5))
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> output = binary_focal_loss_with_logits(pred, target, **kwargs)
        >>> output.backward()
    """

    KORNIA_CHECK_SHAPE(pred, ["B", "C", "*"])
    KORNIA_CHECK(pred.shape == target.shape, f"Expected target size {pred.shape}, got {target.shape}")
    KORNIA_CHECK(
        pred.device == target.device,
        f"pred and target must be in the same device. Got: {pred.device} and {target.device}",
    )

    log_probs_pos: Tensor = nn.functional.logsigmoid(pred)
    log_probs_neg: Tensor = nn.functional.logsigmoid(-pred)

    target, target_mask = mask_ignore_pixels(target, ignore_index)

    if target_mask is not None:
        #  mask ignore pixels
        log_probs_neg = log_probs_neg * target_mask
        log_probs_pos = log_probs_pos * target_mask

    pos_term: Tensor = -log_probs_neg.exp().pow(gamma) * target * log_probs_pos
    neg_term: Tensor = -log_probs_pos.exp().pow(gamma) * (1.0 - target) * log_probs_neg
    if alpha is not None:
        pos_term = alpha * pos_term
        neg_term = (1.0 - alpha) * neg_term

    num_of_classes = pred.shape[1]
    broadcast_dims = [-1] + [1] * len(pred.shape[2:])
    if pos_weight is not None:
        KORNIA_CHECK_IS_TENSOR(pos_weight, "pos_weight must be Tensor or None.")
        KORNIA_CHECK(
            (pos_weight.shape[0] == num_of_classes and pos_weight.numel() == num_of_classes),
            f"pos_weight shape must be (num_of_classes,): ({num_of_classes},), got {pos_weight.shape}",
        )
        KORNIA_CHECK(
            pos_weight.device == pred.device,
            f"pos_weight and pred must be in the same device. Got: {pos_weight.device} and {pred.device}",
        )

        pos_weight = pos_weight.view(broadcast_dims)
        pos_term = pos_weight * pos_term

    loss_tmp: Tensor = pos_term + neg_term
    if weight is not None:
        KORNIA_CHECK_IS_TENSOR(weight, "weight must be Tensor or None.")
        KORNIA_CHECK(
            (weight.shape[0] == num_of_classes and weight.numel() == num_of_classes),
            f"weight shape must be (num_of_classes,): ({num_of_classes},), got {weight.shape}",
        )
        KORNIA_CHECK(
            weight.device == pred.device,
            f"weight and pred must be in the same device. Got: {weight.device} and {pred.device}",
        )

        weight = weight.view(broadcast_dims)
        loss_tmp = weight * loss_tmp

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss