"""
cleanlab can be used for learning with noisy labels for any dataset and model.

For regular (multi-class) classification tasks,
the `~cleanlab.classification.CleanLearning` class wraps an instance of an
sklearn classifier. The wrapped classifier must adhere to the `sklearn estimator API
<https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_,
meaning it must define four functions:

* ``clf.fit(X, y, sample_weight=None)``
* ``clf.predict_proba(X)``
* ``clf.predict(X)``
* ``clf.score(X, y, sample_weight=None)``

where `X` contains data (i.e. features), `y` contains labels (with elements in 0, 1, ..., K-1,
where K is the number of classes). The first index of `X` and of `y` should correspond to the different examples in the dataset,
such that ``len(X) = len(y) = N`` (sample-size). Here `sample_weight` re-weights examples in
the loss function while training (supporting `sample_weight` in your classifier is recommended but optional).

Furthermore, your estimator should be correctly clonable via
`sklearn.base.clone <https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html>`_:
cleanlab internally creates multiple instances of the
estimator, and if you e.g. manually wrap a PyTorch model, you must ensure that
every call to the estimator's ``__init__()`` creates an independent instance of
the model (for sklearn compatibility, the weights of neural network models should typically be initialized inside of ``clf.fit()``).

Note
----
There are two new notions of confidence in this package:

1. Confident *examples* --- examples we are confident are labeled correctly.
We prune everything else. Mathematically, this means keeping the examples
with high probability of belong to their provided label class.

2. Confident *errors* --- examples we are confident are labeled erroneously.
We prune these. Mathematically, this means pruning the examples with
high probability of belong to a different class.

Examples
--------
>>> from cleanlab.classification import CleanLearning
>>> from sklearn.linear_model import LogisticRegression as LogReg
>>> cl = CleanLearning(clf=LogReg()) # Pass in any classifier.
>>> cl.fit(X_train, labels_maybe_with_errors)
>>> # Estimate the predictions as if you had trained without label issues.
>>> pred = cl.predict(X_test)

If the model is not sklearn-compatible by default, it might be the case that
standard packages can adapt the model. For example, you can adapt PyTorch
models using `skorch <https://skorch.readthedocs.io/>`_ and adapt Keras models
using `SciKeras <https://www.adriangb.com/scikeras/>`_.

If an open-source adapter doesn't already exist, you can manually wrap the
model to be sklearn-compatible. This is made easy by inheriting from
`sklearn.base.BaseEstimator
<https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`_:

.. code:: python

    from sklearn.base import BaseEstimator

    class YourModel(BaseEstimator):
        def __init__(self, ):
            pass
        def fit(self, X, y, sample_weight=None):
            pass
        def predict(self, X):
            pass
        def predict_proba(self, X):
            pass
        def score(self, X, y, sample_weight=None):
            pass

Note
----

* `labels` refers to the given labels in the original dataset, which may have errors
* labels must be integers in 0, 1, ..., K-1, where K is the total number of classes

Note
----

Confident learning is the state-of-the-art (`Northcutt et al., 2021 <https://jair.org/index.php/jair/article/view/12125>`_) for
weak supervision, finding label issues in datasets, learning with noisy
labels, uncertainty estimation, and more. It works with *any* classifier,
including deep neural networks. See the `clf` parameter.

Confident learning is a subfield of theory and algorithms of machine learning with noisy labels.
Cleanlab achieves state-of-the-art performance of any open-sourced implementation of confident
learning across a variety of tasks like multi-class classification, multi-label classification,
and PU learning.

Given any classifier having the `predict_proba` method, an input feature
matrix `X`, and a discrete vector of noisy labels `labels`, confident learning estimates the
classifications that would be obtained if the *true labels* had instead been provided
to the classifier during training. `labels` denotes the noisy labels instead of
the :math:`\\tilde{y}` used in confident learning paper.
"""
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import inspect
import warnings
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from typing_extensions import Self
from cleanlab.rank import get_label_quality_scores
from cleanlab import filter
from cleanlab.internal.util import value_counts, compress_int_array, subset_X_y, get_num_classes, force_two_dimensions
from cleanlab.count import estimate_py_noise_matrices_and_cv_pred_proba, estimate_py_and_noise_matrices_from_probabilities, estimate_cv_predicted_probabilities, estimate_latent, compute_confident_joint
from cleanlab.internal.latent_algebra import compute_py_inv_noise_matrix, compute_noise_matrix_from_inverse
from cleanlab.internal.validation import assert_valid_inputs, labels_to_array
from cleanlab.experimental.label_issues_batched import find_label_issues_batched

class CleanLearning(BaseEstimator):
    """
    CleanLearning = Machine Learning with cleaned data (even when training on messy, error-ridden data).

    Automated and robust learning with noisy labels using any dataset and any model. This class
    trains a model `clf` with error-prone, noisy labels as if the model had been instead trained
    on a dataset with perfect labels. It achieves this by cleaning out the error and providing
    cleaned data while training. This class is currently intended for standard (multi-class) classification tasks.

    Parameters
    ----------
    clf : estimator instance, optional
      A classifier implementing the `sklearn estimator API
      <https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_,
      defining the following functions:

      * ``clf.fit(X, y, sample_weight=None)``
      * ``clf.predict_proba(X)``
      * ``clf.predict(X)``
      * ``clf.score(X, y, sample_weight=None)``

      See :py:mod:`cleanlab.models`, the tutorials, and examples/ repo
      for examples of sklearn wrappers, e.g. around PyTorch, Keras, or FastText.

      If the model is not sklearn-compatible by default, it might be the case that
      standard packages can adapt the model. For example, you can adapt PyTorch
      models using `skorch <https://skorch.readthedocs.io/>`_ and adapt Keras models
      using `SciKeras <https://www.adriangb.com/scikeras/>`_.

      Stores the classifier used in Confident Learning.
      Default classifier used is `sklearn.linear_model.LogisticRegression
      <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.
      Default classifier assumes that indexing along the first dimension of the dataset corresponds to
      selecting different training examples.

    seed : int, optional
      Set the default state of the random number generator used to split
      the cross-validated folds. By default, uses `np.random` current random state.

    cv_n_folds : int, default=5
      This class needs holdout predicted probabilities for every data example
      and if not provided, uses cross-validation to compute them.
      `cv_n_folds` sets the number of cross-validation folds used to compute
      out-of-sample probabilities for each example in `X`.

    converge_latent_estimates : bool, optional
      If true, forces numerical consistency of latent estimates. Each is
      estimated independently, but they are related mathematically with closed
      form equivalences. This will iteratively enforce consistency.

    pulearning : {None, 0, 1}, default=None
      Only works for 2 class datasets. Set to the integer of the class that is
      perfectly labeled (you are certain that there are no errors in that class).

    find_label_issues_kwargs : dict, optional
      Keyword arguments to pass into :py:func:`filter.find_label_issues
      <cleanlab.filter.find_label_issues>`. Particularly useful options include:
      `filter_by`, `frac_noise`, `min_examples_per_class` (which all impact ML accuracy),
      `n_jobs` (set this to 1 to disable multi-processing if it's causing issues).

    label_quality_scores_kwargs : dict, optional
      Keyword arguments to pass into :py:func:`rank.get_label_quality_scores
      <cleanlab.rank.get_label_quality_scores>`. Options include: `method`, `adjust_pred_probs`.

    verbose : bool, default=False
      Controls how much output is printed. Set to ``False`` to suppress print
      statements.

    low_memory: bool, default=False
      Set as ``True`` if you have a big dataset with limited memory.
      Uses :py:func:`experimental.label_issues_batched.find_label_issues_batched <cleanlab.experimental.label_issues_batched>`
      to find label issues.
    """

    def __init__(self, clf=None, *, seed=None, cv_n_folds=5, converge_latent_estimates=False, pulearning=None, find_label_issues_kwargs={}, label_quality_scores_kwargs={}, verbose=False, low_memory=False):
        self._default_clf = False
        if clf is None:
            clf = LogReg(solver='lbfgs')
            self._default_clf = True
        if not hasattr(clf, 'fit'):
            raise ValueError('The classifier (clf) must define a .fit() method.')
        if not hasattr(clf, 'predict_proba'):
            raise ValueError('The classifier (clf) must define a .predict_proba() method.')
        if not hasattr(clf, 'predict'):
            raise ValueError('The classifier (clf) must define a .predict() method.')
        if seed is not None:
            np.random.seed(seed=seed)
        self.clf = clf
        self.seed = seed
        self.cv_n_folds = cv_n_folds
        self.converge_latent_estimates = converge_latent_estimates
        self.pulearning = pulearning
        self.find_label_issues_kwargs = find_label_issues_kwargs
        self.label_quality_scores_kwargs = label_quality_scores_kwargs
        self.verbose = verbose
        self.label_issues_df = None
        self.label_issues_mask = None
        self.sample_weight = None
        self.confident_joint = None
        self.py = None
        self.ps = None
        self.num_classes = None
        self.noise_matrix = None
        self.inverse_noise_matrix = None
        self.clf_kwargs = None
        self.clf_final_kwargs = None
        self.low_memory = low_memory

    def fit(self, X, labels=None, *, pred_probs=None, thresholds=None, noise_matrix=None, inverse_noise_matrix=None, label_issues=None, sample_weight=None, clf_kwargs={}, clf_final_kwargs={}, validation_func=None, y=None) -> 'Self':
        """Generate a Python function `fit` for the class `CleanLearning` that trains a model `clf` with error-prone, noisy labels as if it had been trained on a dataset with perfect labels. The function should handle the following steps: 

1. Validate input parameters, ensuring either `labels` or `y` is provided, but not both.
2. If the classifier `clf` is the default one, ensure the input data `X` is two-dimensional.
3. Combine keyword arguments for `clf.fit` using `clf_kwargs` and `clf_final_kwargs`.
4. Check if sample weights are provided and ensure they are supported by the classifier.
5. If `label_issues` is not provided, call the method `find_label_issues` to detect label issues using cross-validation, predicted probabilities, and optionally, noise matrices.
6. Process `label_issues` to ensure it is correctly formatted and contains label quality scores if predicted probabilities are available.
7. Prune the data to exclude examples with label issues and prepare cleaned data `x_cleaned` and `labels_cleaned`.
8. Assign sample weights if the classifier supports them and include them in the final training step if necessary.
9. Fit the classifier `clf` on the cleaned data `x_cleaned` with the corresponding labels `labels_cleaned` using the combined keyword arguments.
10. Store the detected label issues in the class attribute `label_issues_df`.

```python
class CleanLearning(BaseEstimator):
    def __init__(
        self,
        clf=None,
        *,
        seed=None,
        cv_n_folds=5,
        converge_latent_estimates=False,
        pulearning=None,
        find_label_issues_kwargs={},
        label_quality_scores_kwargs={},
        verbose=False,
        low_memory=False,
    ):
        self.clf = clf
        self.seed = seed
        self.cv_n_folds = cv_n_folds
        self.converge_latent_estimates = converge_latent_estimates
        self.pulearning = pulearning
        self.find_label_issues_kwargs = find_label_issues_kwargs
        self.label_quality_scores_kwargs = label_quality_scores_kwargs
        self.verbose = verbose
        self.label_issues_df = None
        self.label_issues_mask = None
        self.sample_weight = None
        self.confident_joint = None
        self.py = None
        self.ps = None
        self.num_classes = None
        self.noise_matrix = None
        self.inverse_noise_matrix = None
        self.clf_kwargs = None
        self.clf_final_kwargs = None
        self.low_memory = low_memory
```"""
        # Step 1: Validate input - either labels or y, not both
        if labels is not None and y is not None:
            raise ValueError(
                "Cannot provide both `labels` and `y`. Please use `labels` (`y` is provided for sklearn compatibility)."
            )
        if labels is None and y is None:
            raise ValueError("Must provide `labels` (or `y` for sklearn compatibility).")
        if y is not None:
            labels = y

        # Step 2: Ensure X is two-dimensional if using default clf
        if self._default_clf:
            X = force_two_dimensions(X)

        # Step 3: Combine keyword arguments
        self.clf_kwargs = clf_kwargs
        self.clf_final_kwargs = clf_final_kwargs
        clf_final_kwargs = {**clf_kwargs, **clf_final_kwargs}

        # Step 4: Check sample weight support
        clf_supports_sample_weight = 'sample_weight' in inspect.signature(self.clf.fit).parameters
        if sample_weight is not None and not clf_supports_sample_weight:
            raise ValueError(
                "The classifier `clf` does not support `sample_weight` in its fit() method."
            )

        labels = labels_to_array(labels)

        # Step 5: Find label issues if not provided
        if label_issues is None:
            label_issues_df = self.find_label_issues(
                X=X,
                labels=labels,
                pred_probs=pred_probs,
                thresholds=thresholds,
                noise_matrix=noise_matrix,
                inverse_noise_matrix=inverse_noise_matrix,
                clf_kwargs=clf_kwargs,
                validation_func=validation_func,
            )
        else:
            if self.verbose:
                print("Using provided label_issues ...")
            # Step 6: Process and format label_issues, adding quality scores if pred_probs available
            label_issues_df = self._process_label_issues_arg(label_issues, labels)
            label_issues_df = label_issues_df.copy()
            if pred_probs is not None:
                num_classes = get_num_classes(labels, pred_probs)
                if 'label_quality' not in label_issues_df.columns:
                    label_quality_scores = get_label_quality_scores(labels, pred_probs, **self.label_quality_scores_kwargs)
                    label_issues_df['label_quality'] = label_quality_scores
                if 'given_label' not in label_issues_df.columns:
                    label_issues_df['given_label'] = compress_int_array(labels, num_classes)
                if 'predicted_label' not in label_issues_df.columns:
                    predicted_labels = pred_probs.argmax(axis=1)
                    label_issues_df['predicted_label'] = compress_int_array(predicted_labels, num_classes)

        # Step 7: Prune data to exclude examples with label issues
        label_issues_mask = label_issues_df['is_label_issue'].to_numpy()
        x_mask = ~label_issues_mask
        x_cleaned, labels_cleaned = subset_X_y(X, labels, x_mask)

        # Step 8: Assign sample weights if classifier supports them
        if sample_weight is not None and clf_supports_sample_weight:
            clf_final_kwargs['sample_weight'] = sample_weight[x_mask]

        # Step 9: Fit classifier on cleaned data
        self.clf.fit(x_cleaned, labels_cleaned, **clf_final_kwargs)

        # Step 10: Store detected label issues
        self.label_issues_df = label_issues_df
        self.label_issues_mask = label_issues_df['is_label_issue']

        return self

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
            elif 'X' in kwargs:
                X = kwargs['X']
                del kwargs['X']
            else:
                raise ValueError('No input provided to predict, please provide X.')
            X = force_two_dimensions(X)
            new_args = (X,) + args[1:]
            return self.clf.predict(*new_args, **kwargs)
        else:
            return self.clf.predict(*args, **kwargs)

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
            elif 'X' in kwargs:
                X = kwargs['X']
                del kwargs['X']
            else:
                raise ValueError('No input provided to predict, please provide X.')
            X = force_two_dimensions(X)
            new_args = (X,) + args[1:]
            return self.clf.predict_proba(*new_args, **kwargs)
        else:
            return self.clf.predict_proba(*args, **kwargs)

    def score(self, X, y, sample_weight=None) -> float:
        """Evaluates your wrapped classifier `clf`'s score on a test set `X` with labels `y`.
        Uses your model's default scoring function, or simply accuracy if your model as no ``"score"`` attribute.

        Parameters
        ----------
        X : np.ndarray or DatasetLike
          Test data in the same format expected by your wrapped classifier.

        y : array_like
          Test labels in the same format as labels previously used in ``fit()``.

        sample_weight : np.ndarray, optional
          An array of shape ``(N,)`` or ``(N, 1)`` used to weight each test example when computing the score.

        Returns
        -------
        score: float
          Number quantifying the performance of this classifier on the test data.
        """
        if self._default_clf:
            X = force_two_dimensions(X)
        if hasattr(self.clf, 'score'):
            if 'sample_weight' in inspect.signature(self.clf.score).parameters:
                return self.clf.score(X, y, sample_weight=sample_weight)
            else:
                return self.clf.score(X, y)
        else:
            return accuracy_score(y, self.clf.predict(X), sample_weight=sample_weight)

    def find_label_issues(self, X=None, labels=None, *, pred_probs=None, thresholds=None, noise_matrix=None, inverse_noise_matrix=None, save_space=False, clf_kwargs={}, validation_func=None) -> pd.DataFrame:
        """
        Identifies potential label issues in the dataset using confident learning.

        Runs cross-validation to get out-of-sample pred_probs from `clf`
        and then calls :py:func:`filter.find_label_issues
        <cleanlab.filter.find_label_issues>` to find label issues.
        These label issues are cached internally and returned in a pandas DataFrame.
        Kwargs for :py:func:`filter.find_label_issues
        <cleanlab.filter.find_label_issues>` must have already been specified
        in the initialization of this class, not here.

        Unlike :py:func:`filter.find_label_issues
        <cleanlab.filter.find_label_issues>`, which requires `pred_probs`,
        this method only requires a classifier and it can do the cross-validation for you.
        Both methods return the same boolean mask that identifies which examples have label issues.
        This is the preferred method to use if you plan to subsequently invoke:
        `~cleanlab.classification.CleanLearning.fit`.

        Note: this method computes the label issues from scratch. To access
        previously-computed label issues from this `~cleanlab.classification.CleanLearning` instance, use the
        `~cleanlab.classification.CleanLearning.get_label_issues` method.

        This is the method called to find label issues inside
        `~cleanlab.classification.CleanLearning.fit`
        and they share mostly the same parameters.

        Parameters
        ----------
        save_space : bool, optional
          If True, then returned `label_issues_df` will not be stored as attribute.
          This means some other methods like `self.get_label_issues()` will no longer work.


        For info about the **other parameters**, see the docstring of `~cleanlab.classification.CleanLearning.fit`.

        Returns
        -------
        label_issues_df : pd.DataFrame
          DataFrame with info about label issues for each example.
          Unless `save_space` argument is specified, same DataFrame is also stored as
          `self.label_issues_df` attribute accessible via
          `~cleanlab.classification.CleanLearning.get_label_issues`.
          Each row represents an example from our dataset and
          the DataFrame may contain the following columns:

          * *is_label_issue*: boolean mask for the entire dataset where ``True`` represents a label issue and ``False`` represents an example that is accurately labeled with high confidence. This column is equivalent to `label_issues_mask` output from :py:func:`filter.find_label_issues<cleanlab.filter.find_label_issues>`.
          * *label_quality*: Numeric score that measures the quality of each label (how likely it is to be correct, with lower scores indicating potentially erroneous labels).
          * *given_label*: Integer indices corresponding to the class label originally given for this example (same as `labels` input). Included here for ease of comparison against `clf` predictions, only present if "predicted_label" column is present.
          * *predicted_label*: Integer indices corresponding to the class predicted by trained `clf` model. Only present if ``pred_probs`` were provided as input or computed during label-issue-finding.
          * *sample_weight*: Numeric values used to weight examples during the final training of `clf` in `~cleanlab.classification.CleanLearning.fit`. This column may not be present after `self.find_label_issues()` but may be added after call to `~cleanlab.classification.CleanLearning.fit`. For more precise definition of sample weights, see documentation of `~cleanlab.classification.CleanLearning.fit`
        """
        assert_valid_inputs(X, labels, pred_probs)
        labels = labels_to_array(labels)
        if noise_matrix is not None and np.trace(noise_matrix) <= 1:
            t = np.round(np.trace(noise_matrix), 2)
            raise ValueError('Trace(noise_matrix) is {}, but must exceed 1.'.format(t))
        if inverse_noise_matrix is not None and np.trace(inverse_noise_matrix) <= 1:
            t = np.round(np.trace(inverse_noise_matrix), 2)
            raise ValueError('Trace(inverse_noise_matrix) is {}. Must exceed 1.'.format(t))
        if self._default_clf:
            X = force_two_dimensions(X)
        if noise_matrix is not None:
            label_matrix = noise_matrix
        else:
            label_matrix = inverse_noise_matrix
        self.num_classes = get_num_classes(labels, pred_probs, label_matrix)
        if pred_probs is None and len(labels) / self.num_classes < self.cv_n_folds:
            raise ValueError('Need more data from each class for cross-validation. Try decreasing cv_n_folds (eg. to 2 or 3) in CleanLearning()')
        self.ps = value_counts(labels) / float(len(labels))
        self.clf_kwargs = clf_kwargs
        if self.low_memory:
            if pred_probs is None:
                if self.verbose:
                    print(f'Computing out of sample predicted probabilities via {self.cv_n_folds}-fold cross validation. May take a while ...')
                pred_probs = estimate_cv_predicted_probabilities(X=X, labels=labels, clf=self.clf, cv_n_folds=self.cv_n_folds, seed=self.seed, clf_kwargs=self.clf_kwargs, validation_func=validation_func)
            if self.verbose:
                print('Using predicted probabilities to identify label issues ...')
            if self.find_label_issues_kwargs:
                warnings.warn(f'`find_label_issues_kwargs` is not used when `low_memory=True`.')
            arg_values = {'thresholds': thresholds, 'noise_matrix': noise_matrix, 'inverse_noise_matrix': inverse_noise_matrix}
            for (arg_name, arg_val) in arg_values.items():
                if arg_val is not None:
                    warnings.warn(f'`{arg_name}` is not used when `low_memory=True`.')
            label_issues_mask = find_label_issues_batched(labels, pred_probs, return_mask=True)
        else:
            self._process_label_issues_kwargs(self.find_label_issues_kwargs)
            if self.confident_joint is not None:
                (self.py, noise_matrix, inv_noise_matrix) = estimate_latent(confident_joint=self.confident_joint, labels=labels)
            if noise_matrix is not None:
                self.noise_matrix = noise_matrix
                if inverse_noise_matrix is None:
                    if self.verbose:
                        print('Computing label noise estimates from provided noise matrix ...')
                    (self.py, self.inverse_noise_matrix) = compute_py_inv_noise_matrix(ps=self.ps, noise_matrix=self.noise_matrix)
            if inverse_noise_matrix is not None:
                self.inverse_noise_matrix = inverse_noise_matrix
                if noise_matrix is None:
                    if self.verbose:
                        print('Computing label noise estimates from provided inverse noise matrix ...')
                    self.noise_matrix = compute_noise_matrix_from_inverse(ps=self.ps, inverse_noise_matrix=self.inverse_noise_matrix)
            if noise_matrix is None and inverse_noise_matrix is None:
                if pred_probs is None:
                    if self.verbose:
                        print(f'Computing out of sample predicted probabilities via {self.cv_n_folds}-fold cross validation. May take a while ...')
                    (self.py, self.noise_matrix, self.inverse_noise_matrix, self.confident_joint, pred_probs) = estimate_py_noise_matrices_and_cv_pred_proba(X=X, labels=labels, clf=self.clf, cv_n_folds=self.cv_n_folds, thresholds=thresholds, converge_latent_estimates=self.converge_latent_estimates, seed=self.seed, clf_kwargs=self.clf_kwargs, validation_func=validation_func)
                else:
                    if self.verbose:
                        print('Computing label noise estimates from provided pred_probs ...')
                    (self.py, self.noise_matrix, self.inverse_noise_matrix, self.confident_joint) = estimate_py_and_noise_matrices_from_probabilities(labels=labels, pred_probs=pred_probs, thresholds=thresholds, converge_latent_estimates=self.converge_latent_estimates)
            if pred_probs is None:
                if self.verbose:
                    print(f'Computing out of sample predicted probabilities via {self.cv_n_folds}-fold cross validation. May take a while ...')
                pred_probs = estimate_cv_predicted_probabilities(X=X, labels=labels, clf=self.clf, cv_n_folds=self.cv_n_folds, seed=self.seed, clf_kwargs=self.clf_kwargs, validation_func=validation_func)
            if self.confident_joint is None:
                self.confident_joint = compute_confident_joint(labels=labels, pred_probs=pred_probs, thresholds=thresholds)
            if self.num_classes == 2 and self.pulearning is not None:
                self.noise_matrix[self.pulearning][1 - self.pulearning] = 0
                self.noise_matrix[1 - self.pulearning][1 - self.pulearning] = 1
                self.inverse_noise_matrix[1 - self.pulearning][self.pulearning] = 0
                self.inverse_noise_matrix[self.pulearning][self.pulearning] = 1
                self.confident_joint[self.pulearning][1 - self.pulearning] = 0
                self.confident_joint[1 - self.pulearning][1 - self.pulearning] = 1
            if 'confident_joint' not in self.find_label_issues_kwargs.keys():
                if not self.find_label_issues_kwargs.get('filter_by') == 'confident_learning':
                    self.find_label_issues_kwargs['confident_joint'] = self.confident_joint
            labels = labels_to_array(labels)
            if self.verbose:
                print('Using predicted probabilities to identify label issues ...')
            label_issues_mask = filter.find_label_issues(labels, pred_probs, **self.find_label_issues_kwargs)
        label_quality_scores = get_label_quality_scores(labels, pred_probs, **self.label_quality_scores_kwargs)
        label_issues_df = pd.DataFrame({'is_label_issue': label_issues_mask, 'label_quality': label_quality_scores})
        if self.verbose:
            print(f'Identified {np.sum(label_issues_mask)} examples with label issues.')
        predicted_labels = pred_probs.argmax(axis=1)
        label_issues_df['given_label'] = compress_int_array(labels, self.num_classes)
        label_issues_df['predicted_label'] = compress_int_array(predicted_labels, self.num_classes)
        if not save_space:
            if self.label_issues_df is not None and self.verbose:
                print('Overwriting previously identified label issues stored at self.label_issues_df. self.get_label_issues() will now return the newly identified label issues. ')
            self.label_issues_df = label_issues_df
            self.label_issues_mask = label_issues_df['is_label_issue']
        elif self.verbose:
            print('Not storing label_issues as attributes since save_space was specified.')
        return label_issues_df

    def get_label_issues(self) -> Optional[pd.DataFrame]:
        """
        Accessor. Returns `label_issues_df` attribute if previously already computed.
        This ``pd.DataFrame`` describes the label issues identified for each example
        (each row corresponds to an example).
        For column definitions, see the documentation\xa0of
        `~cleanlab.classification.CleanLearning.find_label_issues`.

        Returns
        -------
        label_issues_df : pd.DataFrame
          DataFrame with (precomputed) info about label issues for each example.
        """
        if self.label_issues_df is None:
            warnings.warn('Label issues have not yet been computed. Run `self.find_label_issues()` or `self.fit()` first.')
        return self.label_issues_df

    def save_space(self):
        """
        Clears non-sklearn attributes of this estimator to save space (in-place).
        This includes the DataFrame attribute that stored label issues which may be large for big datasets.
        You may want to call this method before deploying this model (i.e. if you just care about producing predictions).
        After calling this method, certain non-prediction-related attributes/functionality will no longer be available
        (e.g. you cannot call ``self.fit()`` anymore).
        """
        if self.label_issues_df is None and self.verbose:
            print('self.label_issues_df is already empty')
        self.label_issues_df = None
        self.sample_weight = None
        self.label_issues_mask = None
        self.find_label_issues_kwargs = None
        self.label_quality_scores_kwargs = None
        self.confident_joint = None
        self.py = None
        self.ps = None
        self.num_classes = None
        self.noise_matrix = None
        self.inverse_noise_matrix = None
        self.clf_kwargs = None
        self.clf_final_kwargs = None
        if self.verbose:
            print('Deleted non-sklearn attributes such as label_issues_df to save space.')

    def _process_label_issues_kwargs(self, find_label_issues_kwargs):
        """
        Private helper function that is used to modify the arguments to passed to
        filter.find_label_issues via the CleanLearning.find_label_issues class. Because
        this is a classification task, some default parameters change and some errors should
        be throne if certain unsupported (for classification) arguments are passed in. This method
        handles those parameters inside of find_label_issues_kwargs and throws an error if you pass
        in a kwargs argument to filter.find_label_issues that is not supported by the
        CleanLearning.find_label_issues() function.
        """
        DEFAULT_FIND_LABEL_ISSUES_KWARGS = {'min_examples_per_class': 10}
        find_label_issues_kwargs = {**DEFAULT_FIND_LABEL_ISSUES_KWARGS, **find_label_issues_kwargs}
        unsupported_kwargs = ['return_indices_ranked_by', 'multi_label']
        for unsupported_kwarg in unsupported_kwargs:
            if unsupported_kwarg in find_label_issues_kwargs:
                raise ValueError(f'These kwargs of `find_label_issues()` are not supported for `CleanLearning`: {unsupported_kwargs}')
        if 'confident_joint' in find_label_issues_kwargs:
            self.confident_joint = find_label_issues_kwargs['confident_joint']
        self.find_label_issues_kwargs = find_label_issues_kwargs

    def _process_label_issues_arg(self, label_issues, labels) -> pd.DataFrame:
        """
        Helper method to get the label_issues input arg into a formatted DataFrame.
        """
        labels = labels_to_array(labels)
        if isinstance(label_issues, pd.DataFrame):
            if 'is_label_issue' not in label_issues.columns:
                raise ValueError("DataFrame label_issues must contain column: 'is_label_issue'. See CleanLearning.fit() documentation for label_issues column descriptions.")
            if len(label_issues) != len(labels):
                raise ValueError('label_issues and labels must have same length')
            if 'given_label' in label_issues.columns and np.any(label_issues['given_label'].to_numpy() != labels):
                raise ValueError("labels must match label_issues['given_label']")
            return label_issues
        elif isinstance(label_issues, np.ndarray):
            if not label_issues.dtype in [np.dtype('bool'), np.dtype('int')]:
                raise ValueError("If label_issues is numpy.array, dtype must be 'bool' or 'int'.")
            if label_issues.dtype is np.dtype('bool') and label_issues.shape != labels.shape:
                raise ValueError('If label_issues is boolean numpy.array, must have same shape as labels')
            if label_issues.dtype is np.dtype('int'):
                if len(np.unique(label_issues)) != len(label_issues):
                    raise ValueError("If label_issues.dtype is 'int', must contain unique integer indices corresponding to examples with label issues such as output by: filter.find_label_issues(..., return_indices_ranked_by=...)")
                issue_indices = label_issues
                label_issues = np.full(len(labels), False, dtype=bool)
                if len(issue_indices) > 0:
                    label_issues[issue_indices] = True
            return pd.DataFrame({'is_label_issue': label_issues})
        else:
            raise ValueError('label_issues must be either pandas.DataFrame or numpy.array')
