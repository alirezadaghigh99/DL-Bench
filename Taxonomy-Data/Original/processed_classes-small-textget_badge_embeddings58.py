    def get_badge_embeddings(self, embeddings, proba):

        proba_argmax = np.argmax(proba, axis=1)
        scale = -1 * proba
        scale[proba_argmax] = -1 * proba[proba_argmax]

        if self.num_classes > 2:
            embedding_size = embeddings.shape[1]
            badge_embeddings = np.zeros((embeddings.shape[0], embedding_size * self.num_classes))
            for c in range(self.num_classes):
                badge_embeddings[:, c * embedding_size:(c + 1) * embedding_size] = (
                            scale[:, c] * np.copy(embeddings).T).T
        else:
            badge_embeddings = embeddings

        return badge_embeddings