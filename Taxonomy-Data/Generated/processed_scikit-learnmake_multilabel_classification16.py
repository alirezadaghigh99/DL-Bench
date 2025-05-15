import numpy as np
from scipy.sparse import csr_matrix

def make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, n_labels=2,
                                   length=50, allow_unlabeled=False, sparse=False,
                                   return_probabilities=False, random_state=None):
    rng = np.random.default_rng(random_state)
    
    # Generate prior class probabilities
    p_c = rng.dirichlet(alpha=np.ones(n_classes), size=1).flatten()
    
    # Generate conditional probabilities of features given classes
    p_w_c = rng.dirichlet(alpha=np.ones(n_features), size=n_classes)
    
    # Generate samples
    X = np.zeros((n_samples, n_features))
    Y = np.zeros((n_samples, n_classes), dtype=int)
    
    for i in range(n_samples):
        # Sample the number of labels for this instance
        num_labels = rng.integers(1, n_labels + 1)
        
        # Sample the classes for this instance
        classes = rng.choice(n_classes, size=num_labels, replace=False, p=p_c)
        Y[i, classes] = 1
        
        # Generate features for this instance
        for cls in classes:
            # Sample features based on the conditional probabilities
            features = rng.choice(n_features, size=length, replace=True, p=p_w_c[cls])
            for f in features:
                X[i, f] += 1
    
    # Normalize feature vectors
    X = X / X.sum(axis=1, keepdims=True)
    
    if sparse:
        X = csr_matrix(X)
        Y = csr_matrix(Y)
    
    if return_probabilities:
        return X, Y, p_c, p_w_c
    else:
        return X, Y

