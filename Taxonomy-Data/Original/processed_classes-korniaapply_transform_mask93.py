    def apply_transform_mask(
        self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:
        _, c, h, w = input.size()

        values = params["values"][..., None, None, None].repeat(1, *input.shape[1:]).to(input)
        # Erase the corresponding areas on masks.
        values = values.zero_()

        bboxes = bbox_generator(params["xs"], params["ys"], params["widths"], params["heights"])
        mask = bbox_to_mask(bboxes, w, h)  # Returns B, H, W
        mask = mask.unsqueeze(1).repeat(1, c, 1, 1).to(input)  # Transform to B, c, H, W
        transformed = where(mask == 1.0, values, input)
        return transformed