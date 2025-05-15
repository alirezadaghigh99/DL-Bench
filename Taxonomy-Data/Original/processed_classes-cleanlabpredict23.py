    def predict(self, *args, **kwargs) -> np.ndarray:
        """Predict class labels using your wrapped classifier `clf`.
        Works just like ``clf.predict()``.

        Parameters
        ----------
        X : np.ndarray or DatasetLike
          Test data in the same format expected by your wrapped classifier.

        Returns
        -------
        class_predictions : np.ndarray
          Vector of class predictions for the test examples.
        """
        if self._default_clf:
            if args:
                X = args[0]
            elif "X" in kwargs:
                X = kwargs["X"]
                del kwargs["X"]
            else:
                raise ValueError("No input provided to predict, please provide X.")
            X = force_two_dimensions(X)
            new_args = (X,) + args[1:]
            return self.clf.predict(*new_args, **kwargs)
        else:
            return self.clf.predict(*args, **kwargs)