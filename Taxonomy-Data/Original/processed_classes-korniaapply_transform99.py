    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        contrast_factor = params["contrast_factor"].to(input)
        return adjust_contrast(input, contrast_factor, self.clip_output)