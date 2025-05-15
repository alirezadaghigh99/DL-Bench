    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        hue_factor = params["hue_factor"].to(input)
        return adjust_hue(input, hue_factor * 2 * pi)