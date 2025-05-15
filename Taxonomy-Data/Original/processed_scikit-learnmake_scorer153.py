def make_scorer(
    score_func, *, response_method="default", greater_is_better=True, **kwargs
):
    """Make a scorer from a performance metric or loss function.

    A scorer is a wrapper around an arbitrary metric or loss function that is called
    with the signature `scorer(estimator, X, y_true, **kwargs)`.

    It is accepted in all scikit-learn estimators or functions allowing a `scoring`
    parameter.

    The parameter `response_method` allows to specify which method of the estimator
    should be used to feed the scoring/loss function.

    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    response_method : {"predict_proba", "decision_function", "predict"} or \
            list/tuple of such str, default=None

        Specifies the response method to use get prediction from an estimator
        (i.e. :term:`predict_proba`, :term:`decision_function` or
        :term:`predict`). Possible choices are:

        - if `str`, it corresponds to the name to the method to return;
        - if a list or tuple of `str`, it provides the method names in order of
          preference. The method returned corresponds to the first method in
          the list and which is implemented by `estimator`.
        - if `None`, it is equivalent to `"predict"`.

        .. versionadded:: 1.4

        .. deprecated:: 1.6
            None is equivalent to 'predict' and is deprecated. It will be removed in
            version 1.8.

    greater_is_better : bool, default=True
        Whether `score_func` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `score_func`.

    **kwargs : additional arguments
        Additional parameters to be passed to `score_func`.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, response_method='predict', beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    """
    sign = 1 if greater_is_better else -1

    if response_method is None:
        warnings.warn(
            "response_method=None is deprecated in version 1.6 and will be removed "
            "in version 1.8. Leave it to its default value to avoid this warning.",
            FutureWarning,
        )
        response_method = "predict"
    elif response_method == "default":
        response_method = "predict"

    return _Scorer(score_func, sign, kwargs, response_method)