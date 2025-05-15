    def transform(
        self, y: Iterable, return_norm: bool = False, target_scale=None, ignore_na: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode iterable with integers.

        Args:
            y (Iterable): iterable to encode
            return_norm: only exists for compatability with other encoders - returns a tuple if true.
            target_scale: only exists for compatability with other encoders - has no effect.
            ignore_na (bool): if to ignore na values and map them to zeros
                (this is different to `add_nan=True` option which maps ONLY NAs to zeros
                while this options maps the first class and NAs to zeros)

        Returns:
            Union[torch.Tensor, np.ndarray]: returns encoded data as torch tensor or numpy array depending on input type
        """
        if self.add_nan:
            if self.warn:
                cond = np.array([item not in self.classes_ for item in y])
                if cond.any():
                    warnings.warn(
                        f"Found {np.unique(np.asarray(y)[cond]).size} unknown classes which were set to NaN",
                        UserWarning,
                    )

            encoded = [self.classes_.get(v, 0) for v in y]

        else:
            if ignore_na:
                na_fill_value = next(iter(self.classes_.values()))
                encoded = [self.classes_.get(v, na_fill_value) for v in y]
            else:
                try:
                    encoded = [self.classes_[v] for v in y]
                except KeyError as e:
                    raise KeyError(
                        f"Unknown category '{e.args[0]}' encountered. Set `add_nan=True` to allow unknown categories"
                    )

        if isinstance(y, torch.Tensor):
            encoded = torch.tensor(encoded, dtype=torch.long, device=y.device)
        else:
            encoded = np.array(encoded)

        if return_norm:
            return encoded, self.get_parameters()
        else:
            return encoded