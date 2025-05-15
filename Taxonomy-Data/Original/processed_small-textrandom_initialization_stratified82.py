def random_initialization_stratified(y, n_samples=10, multilabel_strategy='labelsets'):
    """Randomly draws a subset stratified by class labels.

    Parameters
    ----------
    y : np.ndarray[int] or csr_matrix
        Labels to be used for stratification.
    n_samples :  int
        Number of samples to draw.
    multilabel_strategy : {'labelsets'}, default='labelsets'
        The multi-label strategy to be used in case of a multi-label labeling.
        This is only used if `y` is of type csr_matrix.

    Returns
    -------
    indices : np.ndarray[int]
        Indices relative to y.

    See Also
    --------
    small_text.data.sampling.multilabel_stratified_subsets_sampling : Details on the `labelsets`
        multi-label strategy.
    """
    if isinstance(y, csr_matrix):
        if multilabel_strategy == 'labelsets':
            return multilabel_stratified_subsets_sampling(y, n_samples=n_samples)
        else:
            raise ValueError(f'Invalid multilabel_strategy: {multilabel_strategy}')
    else:
        return stratified_sampling(y, n_samples=n_samples)