    def _parse_auxiliary_input(
        self, aux_input
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor], Optional[int]]:
        """
        Interpret the auxiliary inputs (normals, features) given to __init__.

        Args:
            aux_input:
              Can be either

                - List where each element is a tensor of shape (num_points, C)
                  containing the features for the points in the cloud.
                - Padded float tensor of shape (num_clouds, num_points, C).
              For normals, C = 3

        Returns:
            3-element tuple of list, padded, num_channels.
            If aux_input is list, then padded is None. If aux_input is a tensor,
            then list is None.
        """
        if aux_input is None or self._N == 0:
            return None, None, None

        aux_input_C = None

        if isinstance(aux_input, list):
            return self._parse_auxiliary_input_list(aux_input)
        if torch.is_tensor(aux_input):
            if aux_input.dim() != 3:
                raise ValueError("Auxiliary input tensor has incorrect dimensions.")
            if self._N != aux_input.shape[0]:
                raise ValueError("Points and inputs must be the same length.")
            if self._P != aux_input.shape[1]:
                raise ValueError(
                    "Inputs tensor must have the right maximum \
                    number of points in each cloud."
                )
            if aux_input.device != self.device:
                raise ValueError(
                    "All auxiliary inputs must be on the same device as the points."
                )
            aux_input_C = aux_input.shape[2]
            return None, aux_input, aux_input_C
        else:
            raise ValueError(
                "Auxiliary input must be either a list or a tensor with \
                    shape (batch_size, P, C) where P is the maximum number of \
                    points in a cloud."
            )