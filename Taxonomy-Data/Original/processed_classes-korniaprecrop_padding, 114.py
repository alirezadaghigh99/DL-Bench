    def precrop_padding(self, input: Tensor, flags: Optional[Dict[str, Any]] = None) -> Tensor:
        flags = self.flags if flags is None else flags
        padding = flags["padding"]
        if padding is not None:
            if isinstance(padding, int):
                padding = [padding, padding, padding, padding, padding, padding]
            elif isinstance(padding, (tuple, list)) and len(padding) == 3:
                padding = [padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]]
            elif isinstance(padding, (tuple, list)) and len(padding) == 6:
                padding = [padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]]
            else:
                raise ValueError(f"`padding` must be an integer, 3-element-list or 6-element-list. Got {padding}.")
            input = pad(input, padding, value=flags["fill"], mode=flags["padding_mode"])

        if flags["pad_if_needed"] and input.shape[-3] < flags["size"][0]:
            padding = [0, 0, 0, 0, flags["size"][0] - input.shape[-3], flags["size"][0] - input.shape[-3]]
            input = pad(input, padding, value=flags["fill"], mode=flags["padding_mode"])

        if flags["pad_if_needed"] and input.shape[-2] < flags["size"][1]:
            padding = [0, 0, (flags["size"][1] - input.shape[-2]), flags["size"][1] - input.shape[-2], 0, 0]
            input = pad(input, padding, value=flags["fill"], mode=flags["padding_mode"])

        if flags["pad_if_needed"] and input.shape[-1] < flags["size"][2]:
            padding = [flags["size"][2] - input.shape[-1], flags["size"][2] - input.shape[-1], 0, 0, 0, 0]
            input = pad(input, padding, value=flags["fill"], mode=flags["padding_mode"])

        return input