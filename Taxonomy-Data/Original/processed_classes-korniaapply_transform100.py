    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        saturation_factor = params["saturation_factor"].to(input)
        return adjust_saturation(input, saturation_factor)