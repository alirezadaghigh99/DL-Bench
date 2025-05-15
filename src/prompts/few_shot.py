
def fewshot_data_outside_benchmark(prompt):
    return f'''
    Here are some examples of how to generate the code: 
    ```python
    def _make_pair_range(N: int) -> Iterator[Tuple[int, int]]:
        i, j = tee(range(-1, N))
        next(j, None)
        return zip(i, j)
    ```
    
    ```python
    def cameras_from_opencv_projection(
        R: torch.Tensor,
        tvec: torch.Tensor,
        camera_matrix: torch.Tensor,
        image_size: torch.Tensor,
    ) -> PerspectiveCameras:
        
        return _cameras_from_opencv_projection(R, tvec, camera_matrix, image_size)
    ```
    
    ```python
    def build_pbar_context(pbar_type, tqdm_kwargs=dict()):
        if pbar_type == 'tqdm':
            from tqdm import tqdm
            pbar_context = tqdm(**tqdm_kwargs)
        else:
            pbar_context = NullProgressBar()

        return pbar_context
    ```
    
    ```python

        def init_kmeans_plusplus_safe(x: npt.NDArray[np.double],
                                    n_clusters: int,
                                    *,
                                    weights: Union[npt.NDArray[np.double], None] = None,
                                    x_squared_norms: Union[npt.NDArray[np.double], None] = None,
                                    random_state: Union[int, RandomState, None] = None,
                                    n_local_trials: Union[int, None] = None,
                                    suppress_warning: bool = False):
            """
            Calls scikit-learn's k-means++ initialization with a fallback that prevents duplicate
            cluster centers. If duplicate indices are encountered, they are replaced with randomly sampled
            indices and a warning is raised.

            Parameters
            ----------
            x : numpy.ndarray[double]
                The input data to be clustered.
            n_clusters : int
                The number of cluster centers to find.
            weights : numpy.ndarray[double], default=None

                Sample weights or None. Using this requires scikit-learn >= 1.3.0.

                .. versionadded:: 2.0.0

            x_squared_norms : numpy.ndarray[double], default=None
                List of squared norms or None.
            random_state : int or RandomState instance or None, default=None
                A random state or None.
            n_local_trials : int, default=None
                Number of local trials.
            suppress_warning : bool, default=None
                Suppresses the warning given on duplicate indices if True.

            Returns
            -------
            centers : numpy.ndarray
                `n_clusters`-many cluster centers.
            indices :
                Indices (relativ to `x`) of the returned cluster centers.

            See Also
            --------
            sklearn.cluster import kmeans_plusplus :
                The actual k-means++ initialization provided by scikit-learn.
            """
            kmeans_plusplus_kwargs 
            if weights is not None:
                from sklearn import __version__
                if parse(__version__) < Version('1.3.0'):
                    raise ValueError('scikit-learn>=1.3.0 is required to use the "weights" argument.')

                kmeans_plusplus_kwargs['sample_weight'] = weights

            centers, indices = kmeans_plusplus(x,
                                            n_clusters,
                                            x_squared_norms=x_squared_norms,
                                            random_state=random_state,
                                            n_local_trials=n_local_trials,
                                            **kmeans_plusplus_kwargs)

            unique_indices, counts = np.unique(indices, return_counts=True)

            if unique_indices.shape[0] != n_clusters:
                if not suppress_warning:
                    warnings.warn('kmeans_plusplus returned identical cluster centers.')

                remaining_indices = np.arange(x.shape[0])
                remaining_indices = np.delete(remaining_indices, unique_indices)

                centers = np.delete(centers, np.arange(unique_indices.shape[0])[counts > 1], axis=0)

                fill_indices = choice(remaining_indices, size=n_clusters-unique_indices.shape[0], replace=False)

                indices = np.hstack((unique_indices, fill_indices))
                centers = np.vstack((centers, x[fill_indices]))

                return centers, indices

            return centers, indices
    ```
    How about this function?
    {prompt}


'''
def fewshot_data_outsider(prompt):
    return f'''
    Here are some examples of how to generate the code.

    Example 1:

    ```python
    def _walsh_hadamard_transform(D: TensorLike, n: Optional[int] = None):
    r"""Compute the Walsh–Hadamard Transform of a one-dimensional array.

    Args:
        D (tensor_like): The array or tensor to be transformed. Must have a length that
            is a power of two.

    """
    orig_shape = qml.math.shape(D)
    n = n or int(qml.math.log2(orig_shape[-1]))
    # Reshape the array so that we may apply the Hadamard transform to each axis individually
    if broadcasted := len(orig_shape) > 1:
        new_shape = (orig_shape[0],) + (2,) * n
    else:
        new_shape = (2,) * n
    D = qml.math.reshape(D, new_shape)
    # Apply Hadamard transform to each axis, shifted by one for broadcasting
    for i in range(broadcasted, n + broadcasted):
        D = qml.math.tensordot(_walsh_hadamard_matrix, D, axes=[[1], [i]])
    # The axes are in reverted order after all matrix multiplications, so we need to transpose;
    # If D was broadcasted, this moves the broadcasting axis to first position as well.
    # Finally, reshape to original shape
    return qml.math.reshape(qml.math.transpose(D), orig_shape)
    ```

    Example 2:

    ```python
   def encode_images(self, image_dir=None, recursive: bool = False, num_enc_workers: int = cpu_count()):
        """
        Generate hashes for all images in a given directory of images.

        Args:
            image_dir: Path to the image directory.
            recursive: Optional, find images recursively in a nested image directory structure, set to False by default.
            num_enc_workers: Optional, number of cpu cores to use for multiprocessing encoding generation, set to number of CPUs in the system by default. 0 disables multiprocessing.

        Returns:
            dictionary: A dictionary that contains a mapping of filenames and corresponding 64 character hash string
                        such as 'Image1.jpg': 'hash_string1', 'Image2.jpg': 'hash_string2', ...

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        mapping = myencoder.encode_images('path/to/directory')
        ```
        """
        if not os.path.isdir(image_dir):
            raise ValueError('Please provide a valid directory path!')

        files = generate_files(image_dir, recursive)

        logger.info(f'Start: Calculating hashes...')

        hashes = parallelise(function=self.encode_image, data=files, verbose=self.verbose, num_workers=num_enc_workers)
        hash_initial_dict = dict(zip(generate_relative_names(image_dir, files), hashes))
        hash_dict = 
            k: v for k, v in hash_initial_dict.items() if v
          # To ignore None (returned if some probelm with image file)

        logger.info(f'End: Calculating hashes!')
        return hash_dict

    ```

    Example 3:

    ```python
   def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(sigmoid_focal_loss)
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': 'reduction \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


    ```
    
    Example 4: 
    ```python
    def calibration_curve(
    y_true,
    y_prob,
    *,
    pos_label=None,
    n_bins=5,
    strategy="uniform",
):
    """Compute true and predicted probabilities for a calibration curve.

    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.

    Calibration curves may also be referred to as reliability diagrams.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.

    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.

    pos_label : int, float, bool or str, default=None
        The label of the positive class.

        .. versionadded:: 1.1

    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.

    strategy : 'uniform', 'quantile', default='uniform'
        Strategy used to define the widths of the bins.

        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.

    Returns
    -------
    prob_true : ndarray of shape (n_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).

    prob_pred : ndarray of shape (n_bins,) or smaller
        The mean predicted probability in each bin.

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.calibration import calibration_curve
    >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9,  1.])
    >>> prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)
    >>> prob_true
    array([0. , 0.5, 1. ])
    >>> prob_pred
    array([0.2  , 0.525, 0.85 ])
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)
    pos_label = _check_pos_label_consistency(pos_label, y_true)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels labels."
        )
    y_true = y_true == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return prob_true, prob_pred

    
    ```

    How about this function?
    {prompt}
    '''
    
def fewshot_using_different_categories(prompt):
    return f'''
    Here are some examples of how to generate the code for deep learning.

    Example 1:
    Here is an example of pre-post processing stage
    def point_mesh_edge_distance(meshes: Meshes, pcls: Pointclouds):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_edge(mesh, pcl) + edge_point(mesh, pcl)`

    `point_edge(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest edge segment in mesh and averages across all points in pcl
    `edge_point(mesh, pcl)`: Computes the squared distance of each edge segment in mesh
        to the closest point in pcl and averages across all edges in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        loss: The `point_edge(mesh, pcl) + edge_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for edges
    verts_packed = meshes.verts_packed()
    edges_packed = meshes.edges_packed()
    segms = verts_packed[edges_packed]  # (S, 2, 3)
    segms_first_idx = meshes.mesh_to_edges_packed_first_idx()
    max_segms = meshes.num_edges_per_mesh().max().item()

    # point to edge distance: shape (P,)
    point_to_edge = point_edge_distance(
        points, points_first_idx, segms, segms_first_idx, max_points
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i), )
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    weights_p = 1.0 / weights_p.float()
    point_to_edge = point_to_edge * weights_p
    point_dist = point_to_edge.sum() / N

    # edge to edge distance: shape (S,)
    edge_to_point = edge_point_distance(
        points, points_first_idx, segms, segms_first_idx, max_segms
    )

    # weight each example by the inverse of number of edges in the example
    segm_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(S_n),)
    num_segms_per_mesh = meshes.num_edges_per_mesh()  # (N,)
    weights_s = num_segms_per_mesh.gather(0, segm_to_mesh_idx)
    weights_s = 1.0 / weights_s.float()
    edge_to_point = edge_to_point * weights_s
    edge_dist = edge_to_point.sum() / N

    return point_dist + edge_dist

    Example 2:
    Here is an example of training stage
    def forward(self, x: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)

    Example 3:
    Here is an example for classification task
    
    def train_and_evaluate_classifier():
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=200)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred, target_names=iris.target_names)

    ```
    
    Example 4: 
    Here is an example for image data
    def generalized_box_iou_loss(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-7,
    ) -> torch.Tensor:

        """
        Gradient-friendly IoU loss with an additional penalty that is non-zero when the
        boxes do not overlap and scales with the size of their smallest enclosing box.
        This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

        Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``, and The two boxes should have the
        same dimensions.
        """


        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _log_api_usage_once(generalized_box_iou_loss)

        boxes1 = _upcast_non_float(boxes1)
        boxes2 = _upcast_non_float(boxes2)
        intsctk, unionk = _loss_inter_union(boxes1, boxes2)
        iouk = intsctk / (unionk + eps)

        x1, y1, x2, y2 = boxes1.unbind(dim=-1)
        x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

        # smallest enclosing box
        xc1 = torch.min(x1, x1g)
        yc1 = torch.min(y1, y1g)
        xc2 = torch.max(x2, x2g)
        yc2 = torch.max(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1)
        miouk = iouk - ((area_c - unionk) / (area_c + eps))

        loss = 1 - miouk

        # Check reduction option and return loss accordingly
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': 'reduction \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
    ```

    How about this function?
    {prompt}
    '''
    

def few_shot_using_same_or_different_category(prompt, example1, example2, example3, example4, category, is_same = False):
    base_shots = f'''
        Here are some examples of how to generate the code for deep learning step by step.

        #Example 1:
        {example1}

        Example 2:
        {example2}

        Example 3:
        
        {example3}
        
        Example 4:
        {example4}
        
        
        '''
    if not is_same:
        return base_shots + f'''
            How about this function?
            {prompt}
            '''
    else:
        return base_shots + f'''
            How about this function?
            {prompt}
            '''
            