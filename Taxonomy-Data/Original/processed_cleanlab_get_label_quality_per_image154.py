def _get_label_quality_per_image(pixel_scores, method=None, temperature=0.1):
    from cleanlab.internal.multilabel_scorer import softmin

    """
    Input pixel scores and get label quality score for that image, currently using the "softmin" method.

    Parameters
    ----------
    pixel_scores:
        Per-pixel label quality scores in flattened array of shape ``(N, )``, where N is the number of pixels in the image.

    method: default "softmin"
        Method to use to calculate the image's label quality score.
        Currently only supports "softmin".
    temperature: default 0.1
        Temperature of the softmax function. Too small values may cause numerical underflow and NaN scores.

        Lower values encourage this method to converge toward the label quality score of the pixel with the lowest quality label in the image.

        Higher values encourage this method to converge toward the average label quality score of all pixels in the image.

    Returns
    ---------
    image_score:
        Float of the image's label quality score from 0 to 1, 0 being the lowest quality and 1 being the highest quality.

    """
    if pixel_scores is None or pixel_scores.size == 0:
        raise Exception("Invalid Input: pixel_scores cannot be None or an empty list")

    if temperature == 0 or temperature is None:
        raise Exception("Invalid Input: temperature cannot be zero or None")

    pixel_scores_64 = pixel_scores.astype("float64")
    if method == "softmin":
        if len(pixel_scores_64) > 0:
            return softmin(
                np.expand_dims(pixel_scores_64, axis=0), axis=1, temperature=temperature
            )[0]
        else:
            raise Exception("Invalid Input: pixel_scores is empty")
    else:
        raise Exception("Invalid Method: Specify correct method. Currently only supports 'softmin'")