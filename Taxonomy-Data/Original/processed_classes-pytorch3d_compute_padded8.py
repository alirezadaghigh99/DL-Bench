    def _compute_padded(self, refresh: bool = False):
        """
        Computes the padded version from points_list, normals_list and features_list.

        Args:
            refresh: whether to force the recalculation.
        """
        if not (refresh or self._points_padded is None):
            return

        self._normals_padded, self._features_padded = None, None
        if self.isempty():
            self._points_padded = torch.zeros((self._N, 0, 3), device=self.device)
        else:
            self._points_padded = struct_utils.list_to_padded(
                self.points_list(),
                (self._P, 3),
                pad_value=0.0,
                equisized=self.equisized,
            )
            normals_list = self.normals_list()
            if normals_list is not None:
                self._normals_padded = struct_utils.list_to_padded(
                    normals_list,
                    (self._P, 3),
                    pad_value=0.0,
                    equisized=self.equisized,
                )
            features_list = self.features_list()
            if features_list is not None:
                self._features_padded = struct_utils.list_to_padded(
                    features_list,
                    (self._P, self._C),
                    pad_value=0.0,
                    equisized=self.equisized,
                )