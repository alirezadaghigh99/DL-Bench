def compute_confident_joint(
    labels: LabelLike,
    pred_probs: np.ndarray,
    *,
    thresholds: Optional[Union[np.ndarray, list]] = None,
    calibrate: bool = True,
    multi_label: bool = False,
    return_indices_of_off_diagonals: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """Estimates the confident counts of latent true vs observed noisy labels
    for the examples in our dataset. This array of shape ``(K, K)`` is called the **confident joint**
    and contains counts of examples in every class, confidently labeled as every other class.
    These counts may subsequently be used to estimate the joint distribution of true and noisy labels
    (by normalizing them to frequencies).

    Important: this function assumes that `pred_probs` are out-of-sample
    holdout probabilities. This can be :ref:`done with cross validation <pred_probs_cross_val>`. If
    the probabilities are not computed out-of-sample, overfitting may occur.

    Parameters
    ----------
    labels : np.ndarray or list
      Given class labels for each example in the dataset, some of which may be erroneous,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

    pred_probs : np.ndarray
      Model-predicted class probabilities for each example in the dataset,
      in same format expected by :py:func:`filter.find_label_issues <cleanlab.filter.find_label_issues>` function.

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

    calibrate : bool, default=True
        Calibrates confident joint estimate ``P(label=i, true_label=j)`` such that
        ``np.sum(cj) == len(labels)`` and ``np.sum(cj, axis = 1) == np.bincount(labels)``.
        When ``calibrate=True``, this method returns an estimate of
        the latent true joint counts of noisy and true labels.

    multi_label : bool, optional
      If ``True``, this is multi-label classification dataset (where each example can belong to more than one class)
      rather than a regular (multi-class) classifiction dataset.
      In this case, `labels` should be an iterable (e.g. list) of iterables (e.g. ``List[List[int]]``),
      containing the list of classes to which each example belongs, instead of just a single class.
      Example of `labels` for a multi-label classification dataset: ``[[0,1], [1], [0,2], [0,1,2], [0], [1], [], ...]``.

    return_indices_of_off_diagonals : bool, optional
      If ``True``, returns indices of examples that were counted in off-diagonals
      of confident joint as a baseline proxy for the label issues. This
      sometimes works as well as ``filter.find_label_issues(confident_joint)``.


    Returns
    -------
    confident_joint_counts : np.ndarray
      An array of shape ``(K, K)`` representing counts of examples
      for which we are confident about their given and true label (if `multi_label` is False).
      If `multi_label` is True,
      this array instead has shape ``(K, 2, 2)`` representing a one-vs-rest format for the  confident joint, where for each class `c`:
      Entry ``(c, 0, 0)`` in this one-vs-rest array is the number of examples whose noisy label contains `c` confidently identified as truly belonging to class `c` as well.
      Entry ``(c, 1, 0)`` in this one-vs-rest array is the number of examples whose noisy label contains `c` confidently identified as not actually belonging to class `c`.
      Entry ``(c, 0, 1)`` in this one-vs-rest array is the number of examples whose noisy label does not contain `c` confidently identified as truly belonging to class `c`.
      Entry ``(c, 1, 1)`` in this one-vs-rest array is the number of examples whose noisy label does not contain `c` confidently identified as actually not belonging to class `c` as well.


      Note
      ----
      If `return_indices_of_off_diagonals` is set as True, this function instead returns a tuple `(confident_joint, indices_off_diagonal)`
      where `indices_off_diagonal` is a list of arrays and each array contains the indices of examples counted in off-diagonals of confident joint.

    Note
    ----
    We provide a for-loop based simplification of the confident joint
    below. This implementation is not efficient, not used in practice, and
    not complete, but covers the gist of how the confident joint is computed:

    .. code:: python

        # Confident examples are those that we are confident have true_label = k
        # Estimate (K, K) matrix of confident examples with label = k_s and true_label = k_y
        cj_ish = np.zeros((K, K))
        for k_s in range(K): # k_s is the class value k of noisy labels `s`
            for k_y in range(K): # k_y is the (guessed) class k of true_label k_y
                cj_ish[k_s][k_y] = sum((pred_probs[:,k_y] >= (thresholds[k_y] - 1e-8)) & (labels == k_s))

    The following is a vectorized (but non-parallelized) implementation of the
    confident joint, again slow, using for-loops/simplified for understanding.
    This implementation is 100% accurate, it's just not optimized for speed.

    .. code:: python

        confident_joint = np.zeros((K, K), dtype = int)
        for i, row in enumerate(pred_probs):
            s_label = labels[i]
            confident_bins = row >= thresholds - 1e-6
            num_confident_bins = sum(confident_bins)
            if num_confident_bins == 1:
                confident_joint[s_label][np.argmax(confident_bins)] += 1
            elif num_confident_bins > 1:
                confident_joint[s_label][np.argmax(row)] += 1
    """

    if multi_label:
        if not isinstance(labels, list):
            raise TypeError("`labels` must be list when `multi_label=True`.")

        return _compute_confident_joint_multi_label(
            labels=labels,
            pred_probs=pred_probs,
            thresholds=thresholds,
            calibrate=calibrate,
            return_indices_of_off_diagonals=return_indices_of_off_diagonals,
        )

    # labels needs to be a numpy array
    labels = np.asarray(labels)

    # Estimate the probability thresholds for confident counting
    if thresholds is None:
        # P(we predict the given noisy label is k | given noisy label is k)
        thresholds = get_confident_thresholds(labels, pred_probs, multi_label=multi_label)
    thresholds = np.asarray(thresholds)

    # Compute confident joint (vectorized for speed).

    # pred_probs_bool is a bool matrix where each row represents a training example as a boolean vector of
    # size num_classes, with True if the example confidently belongs to that class and False if not.
    pred_probs_bool = pred_probs >= thresholds - 1e-6
    num_confident_bins = pred_probs_bool.sum(axis=1)
    # The indices where this is false, are often outliers (not confident of any label)
    at_least_one_confident = num_confident_bins > 0
    more_than_one_confident = num_confident_bins > 1
    pred_probs_argmax = pred_probs.argmax(axis=1)
    # Note that confident_argmax is meaningless for rows of all False
    confident_argmax = pred_probs_bool.argmax(axis=1)
    # For each example, choose the confident class (greater than threshold)
    # When there is 2+ confident classes, choose the class with largest prob.
    true_label_guess = np.where(
        more_than_one_confident,
        pred_probs_argmax,
        confident_argmax,
    )
    # true_labels_confident omits meaningless all-False rows
    true_labels_confident = true_label_guess[at_least_one_confident]
    labels_confident = labels[at_least_one_confident]
    confident_joint = confusion_matrix(
        y_true=true_labels_confident,
        y_pred=labels_confident,
        labels=range(pred_probs.shape[1]),
    ).T
    # Guarantee at least one correctly labeled example is represented in every class
    np.fill_diagonal(confident_joint, confident_joint.diagonal().clip(min=1))
    if calibrate:
        confident_joint = calibrate_confident_joint(confident_joint, labels)

    if return_indices_of_off_diagonals:
        true_labels_neq_given_labels = true_labels_confident != labels_confident
        indices = np.arange(len(labels))[at_least_one_confident][true_labels_neq_given_labels]

        return confident_joint, indices

    return confident_joint