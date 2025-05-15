    def fit_transform(self, y: pd.Series, overwrite: bool = False) -> np.ndarray:
        """
        Fit and transform data.

        Args:
            y (pd.Series): input data
            overwrite (bool): if to overwrite current mappings or if to add to it.

        Returns:
            np.ndarray: encoded data
        """
        self.fit(y, overwrite=overwrite)
        return self.transform(y)