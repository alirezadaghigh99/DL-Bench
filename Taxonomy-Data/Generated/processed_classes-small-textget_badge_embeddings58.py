import numpy as np

class BADGE:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def get_badge_embeddings(self, embeddings, proba):
        """
        Calculate badge embeddings scaled by class probabilities.

        Parameters:
        - embeddings: np.ndarray of shape (n_samples, embedding_dim)
        - proba: np.ndarray of shape (n_samples, num_classes)

        Returns:
        - badge_embeddings: np.ndarray of shape (n_samples, embedding_dim * num_classes)
        """
        n_samples, embedding_dim = embeddings.shape

        if self.num_classes == 2:
            # For binary classification, return the original embeddings
            return embeddings

        # For multi-class classification, expand and scale embeddings
        badge_embeddings = np.zeros((n_samples, embedding_dim * self.num_classes))

        for i in range(n_samples):
            for c in range(self.num_classes):
                # Calculate the gradient-like vector for each class
                if c == np.argmax(proba[i]):
                    # For the most likely class, use (1 - p_c)
                    scale = 1 - proba[i, c]
                else:
                    # For other classes, use -p_c
                    scale = -proba[i, c]

                # Place the scaled embedding in the appropriate position
                badge_embeddings[i, c * embedding_dim:(c + 1) * embedding_dim] = embeddings[i] * scale

        return badge_embeddings