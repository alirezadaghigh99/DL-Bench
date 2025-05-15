    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        gamma_factor = params["gamma_factor"].to(input)
        gain_factor = params["gain_factor"].to(input)
        return adjust_gamma(input, gamma_factor, gain_factor)