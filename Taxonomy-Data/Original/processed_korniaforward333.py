    def forward(ctx: Any, input: Tensor, output: Tensor, grad_fn: Optional[Callable[..., Any]] = None) -> Tensor:
        ctx.in_shape = input.shape
        ctx.out_shape = output.shape
        ctx.grad_fn = grad_fn
        return output