    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, Tensor, None]:  # type: ignore[override]
        if ctx.grad_fn is None:
            return grad_output.sum_to_size(ctx.in_shape), grad_output.sum_to_size(ctx.out_shape), None
        return (
            ctx.grad_fn(grad_output.sum_to_size(ctx.in_shape)),
            ctx.grad_fn(grad_output.sum_to_size(ctx.out_shape)),
            None,
        )