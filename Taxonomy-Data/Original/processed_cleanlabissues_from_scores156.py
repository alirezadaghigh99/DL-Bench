def issues_from_scores(
    image_scores: np.ndarray, pixel_scores: Optional[np.ndarray] = None, threshold: float = 0.1
) -> np.ndarray:
    """
    Converts scores output by `~cleanlab.segmentation.rank.get_label_quality_scores`
    to a list of issues of similar format as output by :py:func:`segmentation.filter.find_label_issues <cleanlab.segmentation.filter.find_label_issues>`.

    Only considers as issues those tokens with label quality score lower than `threshold`,
    so this parameter determines the number of issues that are returned.

    Note
    ----
    - This method is intended for converting the most severely mislabeled examples into a format compatible with ``summary`` methods like :py:func:`segmentation.summary.display_issues <cleanlab.segmentation.summary.display_issues>`.
    - This method does not estimate the number of label errors since the `threshold` is arbitrary, for that instead use :py:func:`segmentation.filter.find_label_issues <cleanlab.segmentation.filter.find_label_issues>`, which estimates the label errors via Confident Learning rather than score thresholding.

    Parameters
    ----------
    image_scores:
      Array of shape `(N, )` of overall image scores, where `N` is the number of images in the dataset.
      Same format as the `image_scores` returned by `~cleanlab.segmentation.rank.get_label_quality_scores`.

    pixel_scores:
      Optional array of shape ``(N,H,W)`` of scores between 0 and 1, one per pixel in the dataset.
      Same format as the `pixel_scores` returned by `~cleanlab.segmentation.rank.get_label_quality_scores`.

    threshold:
        Optional quality scores threshold that determines which pixels are included in result. Pixels with with quality scores above the `threshold` are not
        included in the result. If not provided, all pixels are included in result.

    Returns
    ---------
    issues:
      Returns a boolean **mask** for the entire dataset
      where ``True`` represents a pixel label issue and ``False`` represents an example that is
      accurately labeled with using the threshold provided by the user.
      Use :py:func:`segmentation.summary.display_issues <cleanlab.segmentation.summary.display_issues>`
      to view these issues within the original images.

      If `pixel_scores` is not provided, returns array of integer indices (rather than boolean mask) of the images whose label quality score
      falls below the `threshold` (sorted by overall label quality score of each image).

    """

    if image_scores is None:
        raise ValueError("pixel_scores must be provided")
    if threshold < 0 or threshold > 1 or threshold is None:
        raise ValueError("threshold must be between 0 and 1")

    if pixel_scores is not None:
        return pixel_scores < threshold

    ranking = np.argsort(image_scores)
    cutoff = np.searchsorted(image_scores[ranking], threshold)
    return ranking[: cutoff + 1]