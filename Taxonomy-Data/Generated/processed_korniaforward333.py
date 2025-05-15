    @staticmethod
    def forward(ctx: Any, input: Tensor, output: Tensor, grad_fn: Optional[Callable[..., Any]] = None) -> Tensor:
        # Store the grad_fn in the context for use in the backward pass
        ctx.grad_fn = grad_fn
        # Return the output tensor as is, since STE acts as an identity function in the forward pass
        return output