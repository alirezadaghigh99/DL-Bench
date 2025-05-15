    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        """Predict class probabilities ``P(true label=k)`` using your wrapped classifier `clf`.
        Works just like ``clf.predict_proba()``.

        Parameters
        ----------
        X : np.ndarray or DatasetLike
          Test data in the same format expected by your wrapped classifier.

        Returns
        -------
        pred_probs : np.ndarray
          ``(N x K)`` array of predicted class probabilities, one row for each test example.
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
            return self.clf.predict_proba(*new_args, **kwargs)
        else:
            return self.clf.predict_proba(*args, **kwargs)