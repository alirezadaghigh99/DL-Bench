    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        transforms = [
            lambda img: adjust_brightness(img, params["brightness_factor"] - 1)
            if (params["brightness_factor"] - 1 != 0).any()
            else img,
            lambda img: adjust_contrast(img, params["contrast_factor"])
            if (params["contrast_factor"] != 1).any()
            else img,
            lambda img: adjust_saturation(img, params["saturation_factor"])
            if (params["saturation_factor"] != 1).any()
            else img,
            lambda img: adjust_hue(img, params["hue_factor"] * 2 * pi) if (params["hue_factor"] != 0).any() else img,
        ]

        jittered = input
        for idx in params["order"].tolist():
            t = transforms[idx]
            jittered = t(jittered)

        return jittered