    def flatten_and_preprocess_masks(
        labels: np.ndarray, pred_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, num_classes, _, _ = pred_probs.shape
        labels_flat = labels.flatten().astype(int)
        pred_probs_flat = np.moveaxis(pred_probs, 0, 1).reshape(num_classes, -1)

        return labels_flat, pred_probs_flat.T