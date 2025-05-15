    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        brightness_factor = params["brightness_factor"].to(input)
        return adjust_brightness(input, brightness_factor - 1, self.clip_output)