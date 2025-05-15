    def fit(
        self,
        X,
        labels=None,
        *,
        pred_probs=None,
        thresholds=None,
        noise_matrix=None,
        inverse_noise_matrix=None,
        label_issues=None,
        sample_weight=None,
        clf_kwargs={},
        clf_final_kwargs={},
        validation_func=None,
        y=None,
    ) -> "Self":
        """
        Train the model `clf` with error-prone, noisy labels as if
        the model had been instead trained on a dataset with the correct labels.
        `fit` achieves this by first training `clf` via cross-validation on the noisy data,
        using the resulting predicted probabilities to identify label issues,
        pruning the data with label issues, and finally training `clf` on the remaining clean data.

        Parameters
        ----------
        X : np.ndarray or DatasetLike
          Data features (i.e. training inputs for ML), typically an array of shape ``(N, ...)``,
          where N is the number of examples.
          Supported `DatasetLike` types beyond ``np.ndarray`` include:
          ``pd.DataFrame``, ``scipy.sparse.csr_matrix``, ``torch.utils.data.Dataset``, ``tensorflow.data.Dataset``,
          or any dataset object ``X`` that supports list-based indexing:
          ``X[index_list]`` to select a subset of training examples.
          Your classifier that this instance was initialized with,
          ``clf``, must be able to ``fit()`` and ``predict()`` data of this format.

          Note
          ----
          If providing `X` as a ``tensorflow.data.Dataset``,
          make sure ``shuffle()`` has been called before ``batch()`` (if shuffling)
          and no other order-destroying operation (eg. ``repeat()``) has been applied.

        labels : array_like
          An array of shape ``(N,)`` of noisy classification labels, where some labels may be erroneous.
          Elements must be integers in the set 0, 1, ..., K-1, where K is the number of classes.
          Supported `array_like` types include: ``np.ndarray``, ``pd.Series``, or ``list``.

        pred_probs : np.ndarray, optional
          An array of shape ``(N, K)`` of model-predicted probabilities,
          ``P(label=k|x)``. Each row of this matrix corresponds
          to an example `x` and contains the model-predicted probabilities that
          `x` belongs to each possible class, for each of the K classes. The
          columns must be ordered such that these probabilities correspond to class 0, 1, ..., K-1.
          `pred_probs` should be :ref:`out-of-sample, eg. computed via cross-validation <pred_probs_cross_val>`.
          If provided, `pred_probs` will be used to find label issues rather than the ``clf`` classifier.

          Note
          ----
          If you are not sure, leave ``pred_probs=None`` (the default) and it
          will be computed for you using cross-validation with the provided model.

        thresholds : array_like, optional
          An array of shape ``(K, 1)`` or ``(K,)`` of per-class threshold
          probabilities, used to determine the cutoff probability necessary to
          consider an example as a given class label (see `Northcutt et al.,
          2021 <https://jair.org/index.php/jair/article/view/12125>`_, Section
          3.1, Equation 2).

          This is for advanced users only. If not specified, these are computed
          for you automatically. If an example has a predicted probability
          greater than this threshold, it is counted as having true_label =
          k. This is not used for pruning/filtering, only for estimating the
          noise rates using confident counts.

        noise_matrix : np.ndarray, optional
          An array of shape ``(K, K)`` representing the conditional probability
          matrix ``P(label=k_s | true label=k_y)``, the
          fraction of examples in every class, labeled as every other class.
          Assumes columns of `noise_matrix` sum to 1.

        inverse_noise_matrix : np.ndarray, optional
          An array of shape ``(K, K)`` representing the conditional probability
          matrix ``P(true label=k_y | label=k_s)``,
          the estimated fraction observed examples in each class ``k_s``
          that are mislabeled examples from every other class ``k_y``,
          Assumes columns of `inverse_noise_matrix` sum to 1.

        label_issues : pd.DataFrame or np.ndarray, optional
          Specifies the label issues for each example in dataset.
          If ``pd.DataFrame``, must be formatted as the one returned by:
          :py:meth:`CleanLearning.find_label_issues
          <cleanlab.classification.CleanLearning.find_label_issues>` or
          `~cleanlab.classification.CleanLearning.get_label_issues`.
          If ``np.ndarray``, must contain either boolean `label_issues_mask` as output by:
          default :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`,
          or integer indices as output by
          :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>`
          with its `return_indices_ranked_by` argument specified.
          Providing this argument significantly reduces the time this method takes to run by
          skipping the slow cross-validation step necessary to find label issues.
          Examples identified to have label issues will be
          pruned from the data before training the final `clf` model.

          Caution: If you provide `label_issues` without having previously called
          `~cleanlab.classification.CleanLearning.find_label_issues`
          e.g. as a ``np.ndarray``, then some functionality like training with sample weights may be disabled.

        sample_weight : array_like, optional
          Array of weights with shape ``(N,)`` that are assigned to individual samples,
          assuming total number of examples in dataset is `N`.
          If not provided, samples may still be weighted by the estimated noise in the class they are labeled as.

        clf_kwargs : dict, optional
          Optional keyword arguments to pass into `clf`'s ``fit()`` method.

        clf_final_kwargs : dict, optional
          Optional extra keyword arguments to pass into the final `clf` ``fit()`` on the cleaned data
          but not the `clf` ``fit()`` in each fold of cross-validation on the noisy data.
          The final ``fit()`` will also receive `clf_kwargs`,
          but these may be overwritten by values in `clf_final_kwargs`.
          This can be useful for training differently in the final ``fit()``
          than during cross-validation.

        validation_func : callable, optional
          Optional callable function that takes two arguments, `X_val`, `y_val`, and returns a dict
          of keyword arguments passed into to ``clf.fit()`` which may be functions of the validation
          data in each cross-validation fold. Specifies how to map the validation data split in each
          cross-validation fold into the appropriate format to pass into `clf`'s ``fit()`` method, assuming
          ``clf.fit()`` can utilize validation data if it is appropriately passed in (eg. for early-stopping).
          Eg. if your model's ``fit()`` method is called using ``clf.fit(X, y, X_validation, y_validation)``,
          then you could set ``validation_func = f`` where
          ``def f(X_val, y_val): return {"X_validation": X_val, "y_validation": y_val}``

          Note that `validation_func` will be ignored in the final call to `clf.fit()` on the
          cleaned subset of the data. This argument is only for allowing `clf` to access the
          validation data in each cross-validation fold (eg. for early-stopping or hyperparameter-selection
          purposes). If you want to pass in validation data even in the final training call to ``clf.fit()``
          on the cleaned data subset, you should explicitly pass in that data yourself
          (eg. via `clf_final_kwargs` or `clf_kwargs`).

        y: array_like, optional
          Alternative argument that can be specified instead of `labels`.
          Specifying `y` has the same effect as specifying `labels`,
          and is offered as an alternative for compatibility with sklearn.

        Returns
        -------
        self : CleanLearning
          Fitted estimator that has all the same methods as any sklearn estimator.


          After calling ``self.fit()``, this estimator also stores extra attributes such as:

          * *self.label_issues_df*: a ``pd.DataFrame`` accessible via
          `~cleanlab.classification.CleanLearning.get_label_issues`
          of similar format as the one returned by: `~cleanlab.classification.CleanLearning.find_label_issues`.
          See documentation of :py:meth:`CleanLearning.find_label_issues<cleanlab.classification.CleanLearning.find_label_issues>`
          for column descriptions.


          After calling ``self.fit()``, `self.label_issues_df` may also contain an extra column:

          * *sample_weight*: Numeric values that were used to weight examples during
            the final training of `clf` in ``CleanLearning.fit()``.
            `sample_weight` column will only be present if automatic sample weights were actually used.
            These automatic weights are assigned to each example based on the class it belongs to,
            i.e. there are only num_classes unique sample_weight values.
            The sample weight for an example belonging to class k is computed as ``1 / p(given_label = k | true_label = k)``.
            This sample_weight normalizes the loss to effectively trick `clf` into learning with the distribution
            of the true labels by accounting for the noisy data pruned out prior to training on cleaned data.
            In other words, examples with label issues were removed, so this weights the data proportionally
            so that the classifier trains as if it had all the true labels,
            not just the subset of cleaned data left after pruning out the label issues.

        Note
        ----
        If ``CleanLearning.fit()`` does not work for your data/model, you can run the same procedure yourself:
        * Utilize :ref:`cross-validation <pred_probs_cross_val>` to get out-of-sample `pred_probs` for each example.
        * Call :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` with `pred_probs`.
        * Filter the examples with detected issues and train your model on the remaining data.
        """

        if labels is not None and y is not None:
            raise ValueError("You must specify either `labels` or `y`, but not both.")
        if y is not None:
            labels = y
        if labels is None:
            raise ValueError("You must specify `labels`.")
        if self._default_clf:
            X = force_two_dimensions(X)

        self.clf_final_kwargs = {**clf_kwargs, **clf_final_kwargs}

        if "sample_weight" in clf_kwargs:
            raise ValueError(
                "sample_weight should be provided directly in fit() or in clf_final_kwargs rather than in clf_kwargs"
            )

        if sample_weight is not None:
            if "sample_weight" not in inspect.signature(self.clf.fit).parameters:
                raise ValueError(
                    "sample_weight must be a supported fit() argument for your model in order to be specified here"
                )

        if label_issues is None:
            if self.label_issues_df is not None and self.verbose:
                print(
                    "If you already ran self.find_label_issues() and don't want to recompute, you "
                    "should pass the label_issues in as a parameter to this function next time."
                )
            label_issues = self.find_label_issues(
                X,
                labels,
                pred_probs=pred_probs,
                thresholds=thresholds,
                noise_matrix=noise_matrix,
                inverse_noise_matrix=inverse_noise_matrix,
                clf_kwargs=clf_kwargs,
                validation_func=validation_func,
            )

        else:  # set args that may not have been set if `self.find_label_issues()` wasn't called yet
            assert_valid_inputs(X, labels, pred_probs)
            if self.num_classes is None:
                if noise_matrix is not None:
                    label_matrix = noise_matrix
                else:
                    label_matrix = inverse_noise_matrix
                self.num_classes = get_num_classes(labels, pred_probs, label_matrix)
            if self.verbose:
                print("Using provided label_issues instead of finding label issues.")
                if self.label_issues_df is not None:
                    print(
                        "These will overwrite self.label_issues_df and will be returned by "
                        "`self.get_label_issues()`. "
                    )

        # label_issues always overwrites self.label_issues_df. Ensure it is properly formatted:
        self.label_issues_df = self._process_label_issues_arg(label_issues, labels)

        if "label_quality" not in self.label_issues_df.columns and pred_probs is not None:
            if self.verbose:
                print("Computing label quality scores based on given pred_probs ...")
            self.label_issues_df["label_quality"] = get_label_quality_scores(
                labels, pred_probs, **self.label_quality_scores_kwargs
            )

        self.label_issues_mask = self.label_issues_df["is_label_issue"].to_numpy()
        x_mask = np.invert(self.label_issues_mask)
        x_cleaned, labels_cleaned = subset_X_y(X, labels, x_mask)
        if self.verbose:
            print(f"Pruning {np.sum(self.label_issues_mask)} examples with label issues ...")
            print(f"Remaining clean data has {len(labels_cleaned)} examples.")

        if sample_weight is None:
            # Check if sample_weight in args of clf.fit()
            if (
                "sample_weight" in inspect.signature(self.clf.fit).parameters
                and "sample_weight" not in self.clf_final_kwargs
                and self.noise_matrix is not None
            ):
                # Re-weight examples in the loss function for the final fitting
                # such that the "apparent" original number of examples in each class
                # is preserved, even though the pruned sets may differ.
                if self.verbose:
                    print(
                        "Assigning sample weights for final training based on estimated label quality."
                    )
                sample_weight_auto = np.ones(np.shape(labels_cleaned))
                for k in range(self.num_classes):
                    sample_weight_k = 1.0 / max(
                        self.noise_matrix[k][k], 1e-3
                    )  # clip sample weights
                    sample_weight_auto[labels_cleaned == k] = sample_weight_k

                sample_weight_expanded = np.zeros(
                    len(labels)
                )  # pad pruned examples with zeros, length of original dataset
                sample_weight_expanded[x_mask] = sample_weight_auto
                # Store the sample weight for every example in the original, unfiltered dataset
                self.label_issues_df["sample_weight"] = sample_weight_expanded
                self.sample_weight = self.label_issues_df[
                    "sample_weight"
                ]  # pointer to here to avoid duplication
                self.clf_final_kwargs["sample_weight"] = sample_weight_auto
                if self.verbose:
                    print("Fitting final model on the clean data ...")
            else:
                if self.verbose:
                    if "sample_weight" in self.clf_final_kwargs:
                        print("Fitting final model on the clean data with custom sample_weight ...")
                    else:
                        if (
                            "sample_weight" in inspect.signature(self.clf.fit).parameters
                            and self.noise_matrix is None
                        ):
                            print(
                                "Cannot utilize sample weights for final training! "
                                "Why this matters: during final training, sample weights help account for the amount of removed data in each class. "
                                "This helps ensure the correct class prior for the learned model. "
                                "To use sample weights, you need to either provide the noise_matrix or have previously called self.find_label_issues() instead of filter.find_label_issues() which computes them for you."
                            )
                        print("Fitting final model on the clean data ...")

        elif sample_weight is not None and "sample_weight" not in self.clf_final_kwargs:
            self.clf_final_kwargs["sample_weight"] = sample_weight[x_mask]
            if self.verbose:
                print("Fitting final model on the clean data with custom sample_weight ...")

        else:  # pragma: no cover
            if self.verbose:
                if "sample_weight" in self.clf_final_kwargs:
                    print("Fitting final model on the clean data with custom sample_weight ...")
                else:
                    print("Fitting final model on the clean data ...")

        self.clf.fit(x_cleaned, labels_cleaned, **self.clf_final_kwargs)

        if self.verbose:
            print(
                "Label issues stored in label_issues_df DataFrame accessible via: self.get_label_issues(). "
                "Call self.save_space() to delete this potentially large DataFrame attribute."
            )
        return self