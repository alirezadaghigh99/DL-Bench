
pre_post_stage = (
    '''
    def create_tiles(
    images: List[np.ndarray],
    grid_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
    single_tile_size: Optional[Tuple[int, int]] = None,
    tile_scaling: Literal["min", "max", "avg"] = "avg",
    tile_padding_color: Tuple[int, int, int] = (0, 0, 0),
    tile_margin: int = 15,
    tile_margin_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    if len(images) == 0:
        raise ValueError("Could not create image tiles from empty list of images.")
    if single_tile_size is None:
        single_tile_size = _aggregate_images_shape(images=images, mode=tile_scaling)
    resized_images = [
        letterbox_image(
            image=i, desired_size=single_tile_size, color=tile_padding_color
        )
        for i in images
    ]
    grid_size = _establish_grid_size(images=images, grid_size=grid_size)
    if len(images) > grid_size[0] * grid_size[1]:
        raise ValueError(f"Grid of size: grid_size cannot fit {len(images)} images.")
    return _generate_tiles(
        images=resized_images,
        grid_size=grid_size,
        single_tile_size=single_tile_size,
        tile_padding_color=tile_padding_color,
        tile_margin=tile_margin,
        tile_margin_color=tile_margin_color,
    )
    ''',
    '''
    
def _iterate_shift_rule_with_multipliers(rule, order, period):
    r"""Helper method to repeat a shift rule that includes multipliers multiple
    times along the same parameter axis for higher-order derivatives."""
    combined_rules = []

    for partial_rules in itertools.product(rule, repeat=order):
        c, m, s = np.stack(partial_rules).T
        cumul_shift = 0.0
        for _m, _s in zip(m, s):
            cumul_shift *= _m
            cumul_shift += _s
        if period is not None:
            cumul_shift = np.mod(cumul_shift + 0.5 * period, period) - 0.5 * period
        combined_rules.append(np.stack([np.prod(c), np.prod(m), cumul_shift]))

    # combine all terms in the linear combination into a single
    # array, with column order (coefficients, multipliers, shifts)
    return qml.math.stack(combined_rules)
    ''',
    '''
    def affine(
    img: Image.Image,
    matrix: List[float],
    interpolation: int = Image.NEAREST,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
) -> Image.Image:

    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    output_size = img.size
    opts = _parse_fill(fill, img)
    return img.transform(output_size, Image.AFFINE, matrix, interpolation, **opts)

    '''
)

training_stage = (
    '''
    def fit(
        self,
        train_loader: DataLoader,
        override: bool = True,
        progress_bar: bool = False,
    ) -> None:
        """Fit the local Laplace approximation at the parameters of the model.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set
        override : bool, default=True
            whether to initialize H, loss, and n_data again; setting to False is useful for
            online learning settings to accumulate a sequential posterior approximation.
        progress_bar: bool, default=False
        """
        if not override:
            raise ValueError(
                "Last-layer Laplace approximations do not support `override=False`."
            )

        self.model.eval()

        if self.model.last_layer is None:
            self.data: tuple[torch.Tensor, torch.Tensor] | MutableMapping = next(
                iter(train_loader)
            )
            self._find_last_layer(self.data)
            params: torch.Tensor = parameters_to_vector(
                self.model.last_layer.parameters()
            ).detach()
            self.n_params: int = len(params)
            self.n_layers: int = len(list(self.model.last_layer.parameters()))
            # here, check the already set prior precision again
            self.prior_precision: float | torch.Tensor = self._prior_precision
            self.prior_mean: float | torch.Tensor = self._prior_mean
            self._init_H()

        super().fit(train_loader, override=override)
        self.mean: torch.Tensor = parameters_to_vector(
            self.model.last_layer.parameters()
        )

        if not self.enable_backprop:
            self.mean = self.mean.detach()
    ''',
    '''
    def dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    r"""During training, randomly zeroes some elements of the input tensor with probability :attr:`p`.

    Uses samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            dropout, (input,), input, p=p, training=training, inplace=inplace
        )
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got p")
    return (
        _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)
    )

    ''', 
    
    '''
    def spatial_gradient(input: Tensor, mode: str = "sobel", order: int = 1, normalized: bool = True) -> Tensor:
    r"""Compute the first order image derivative in both x and y using a Sobel operator.

    .. image:: _static/img/spatial_gradient.png

    Args:
        input: input image tensor with shape :math:`(B, C, H, W)`.
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.
        normalized: whether the output is normalized.

    Return:
        the derivatives of the input feature map. with shape :math:`(B, C, 2, H, W)`.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/filtering_edges.html>`__.

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = spatial_gradient(input)  # 1x3x2x4x4
        >>> output.shape
        torch.Size([1, 3, 2, 4, 4])
    """
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    # allocate kernel
    kernel = get_spatial_gradient_kernel2d(mode, order, device=input.device, dtype=input.dtype)
    if normalized:
        kernel = normalize_kernel2d(kernel)

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...]

    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels: int = 3 if order == 2 else 2
    padded_inp: Tensor = pad(input.reshape(b * c, 1, h, w), spatial_pad, "replicate")
    out = F.conv2d(padded_inp, tmp_kernel, groups=1, padding=0, stride=1)
    return out.reshape(b, c, out_channels, h, w)
    '''
    
    
)

model_stage = (
    '''
    def camera_position_from_spherical_angles(
    distance: float,
    elevation: float,
    azimuth: float,
    degrees: bool = True,
    device: Device = "cpu",
) -> torch.Tensor:
    """
    Calculate the location of the camera based on the distance away from
    the target point, the elevation and azimuth angles.

   
    """
    broadcasted_args = convert_to_tensors_and_broadcast(
        distance, elevation, azimuth, device=device
    )
    dist, elev, azim = broadcasted_args
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    camera_position = torch.stack([x, y, z], dim=1)
    if camera_position.dim() == 0:
        camera_position = camera_position.view(1, -1)  # add batch dim.
    return camera_position.view(-1, 3)
    
    ''',
    '''
    def gauss_kl(
    q_mu: TensorType, q_sqrt: TensorType, K: TensorType = None, *, K_cholesky: TensorType = None
) -> tf.Tensor:
    """
    Compute the KL divergence KL[q || p] between::

          q(x) = N(q_mu, q_sqrt^2)

    and::

          p(x) = N(0, K)    if K is not None
          p(x) = N(0, I)    if K is None
    """

    if (K is not None) and (K_cholesky is not None):
        raise ValueError(
            "Ambiguous arguments: gauss_kl() must only be passed one of `K` or `K_cholesky`."
        )

    is_white = (K is None) and (K_cholesky is None)
    is_diag = len(q_sqrt.shape) == 2

    M, L = tf.shape(q_mu)[0], tf.shape(q_mu)[1]

    if is_white:
        alpha = q_mu  # [M, L]
    else:
        if K is not None:
            Lp = tf.linalg.cholesky(K)  # [L, M, M] or [M, M]
        elif K_cholesky is not None:
            Lp = K_cholesky  # [L, M, M] or [M, M]

        is_batched = len(Lp.shape) == 3

        q_mu = tf.transpose(q_mu)[:, :, None] if is_batched else q_mu  # [L, M, 1] or [M, L]
        alpha = tf.linalg.triangular_solve(Lp, q_mu, lower=True)  # [L, M, 1] or [M, L]

    if is_diag:
        Lq = Lq_diag = q_sqrt
        Lq_full = tf.linalg.diag(tf.transpose(q_sqrt))  # [L, M, M]
    else:
        Lq = Lq_full = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle # [L, M, M]
        Lq_diag = tf.linalg.diag_part(Lq)  # [M, L]

    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = tf.reduce_sum(tf.square(alpha))

    # Constant term: - L * M
    constant = -to_default_float(tf.size(q_mu, out_type=tf.int64))

    # Log-determinant of the covariance of q(x):
    logdet_qcov = tf.reduce_sum(tf.math.log(tf.square(Lq_diag)))

    # Trace term: tr(Σp⁻¹ Σq)
    if is_white:
        trace = tf.reduce_sum(tf.square(Lq))
    else:
        if is_diag and not is_batched:
            # K is [M, M] and q_sqrt is [M, L]: fast specialisation
            LpT = tf.transpose(Lp)  # [M, M]
            Lp_inv = tf.linalg.triangular_solve(
                Lp, tf.eye(M, dtype=default_float()), lower=True
            )  # [M, M]
            K_inv = tf.linalg.diag_part(tf.linalg.triangular_solve(LpT, Lp_inv, lower=False))[
                :, None
            ]  # [M, M] -> [M, 1]
            trace = tf.reduce_sum(K_inv * tf.square(q_sqrt))
        else:
            if is_batched or Version(tf.__version__) >= Version("2.2"):
                Lp_full = Lp
            else:
                # workaround for segfaults when broadcasting in TensorFlow<2.2
                Lp_full = tf.tile(tf.expand_dims(Lp, 0), [L, 1, 1])
            LpiLq = tf.linalg.triangular_solve(Lp_full, Lq_full, lower=True)
            trace = tf.reduce_sum(tf.square(LpiLq))

    twoKL = mahalanobis + constant - logdet_qcov + trace

    # Log-determinant of the covariance of p(x):
    if not is_white:
        log_sqdiag_Lp = tf.math.log(tf.square(tf.linalg.diag_part(Lp)))
        sum_log_sqdiag_Lp = tf.reduce_sum(log_sqdiag_Lp)
        # If K is [L, M, M], num_latent_gps is no longer implicit, no need to multiply the single kernel logdet
        scale = 1.0 if is_batched else to_default_float(L)
        twoKL += scale * sum_log_sqdiag_Lp

    return 0.5 * twoKL

    ''',
    '''
    def create_knn_graph_and_index(
    features: Optional[FeatureArray],
    *,
    n_neighbors: Optional[int] = None,
    metric: Optional[Metric] = None,
    correct_exact_duplicates: bool = True,
    **sklearn_knn_kwargs,
) -> Tuple[csr_matrix, NearestNeighbors]:
    """Calculate the KNN graph from the features if it is not provided in the kwargs.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.internal.neighbor.knn_graph import create_knn_graph_and_index
    >>> features = np.array([
    ...    [0.701, 0.701],
    ...    [0.900, 0.436],
    ...    [0.000, 1.000],
    ... ])
    >>> knn_graph, knn = create_knn_graph_and_index(features, n_neighbors=1)
    >>> knn_graph.toarray()  # For demonstration purposes only. It is generally a bad idea to transform to dense matrix for large graphs.
    array([[0.        , 0.33140006, 0.        ],
           [0.33140006, 0.        , 0.        ],
           [0.76210367, 0.        , 0.        ]])
    >>> knn
    NearestNeighbors(metric=<function euclidean at ...>, n_neighbors=1)  # For demonstration purposes only. The actual metric may vary.
    """
    # Construct NearestNeighbors object
    knn = features_to_knn(features, n_neighbors=n_neighbors, metric=metric, **sklearn_knn_kwargs)
    # Build graph from NearestNeighbors object
    knn_graph = construct_knn_graph_from_index(knn)

    # Ensure that exact duplicates found with np.unique aren't accidentally missed in the KNN graph
    if correct_exact_duplicates:
        assert features is not None
        knn_graph = correct_knn_graph(features, knn_graph)
    return knn_graph, knn''',
    
        
    
    
)

infer_stage = (
    '''
    def find_fundamental(
    points1: Tensor, points2: Tensor, weights: Optional[Tensor] = None, method: Literal["8POINT", "7POINT"] = "8POINT"
) -> Tensor:
    r"""
    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.
        method: The method to use for computing the fundamental matrix. Supported methods are "7POINT" and "8POINT".

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3*m, 3)`, where `m` number of fundamental matrix.

    Raises:
        ValueError: If an invalid method is provided.

    """
    if method.upper() == "7POINT":
        result = run_7point(points1, points2)
    elif method.upper() == "8POINT":
        result = run_8point(points1, points2, weights)
    else:
        raise ValueError(f"Invalid method: method. Supported methods are '7POINT' and '8POINT'.")
    return result
''',
'''
def predict_proba(self, *args, **kwargs) -> np.ndarray:
        """Predict class probabilities ``P(true label=k)`` using your wrapped classifier `clf`.
        Works just like ``clf.predict_proba()``.

        Parameters
        ----------
        X : np.ndarray or DatasetLike
          Test data in the same format expected by your wrapped classifier.

        Returns
        -------
        pred_probs : np.ndarray
          ``(N x K)`` array of predicted class probabilities, one row for each test example.
        """
        if self._default_clf:
            if args:
                X = args[0]
            elif "X" in kwargs:
                X = kwargs["X"]
                del kwargs["X"]
            else:
                raise ValueError("No input provided to predict, please provide X.")
            X = force_two_dimensions(X)
            new_args = (X,) + args[1:]
            return self.clf.predict_proba(*new_args, **kwargs)
        else:
            return self.clf.predict_proba(*args, **kwargs)
            ''',
            
    '''
    def forward(
        self, X: Tensor, prior: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        x = self._get_embeddings(X)
        steps_output, M_loss = self.encoder(x, prior)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        return (res, M_loss)

    '''
    
)

eval_stage = (
    
   
    '''
     def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate whether constraints are violated or satisfied at a set of x locations

        :param x: Array of shape (n_points x n_dims) containing input locations to evaluate constraint at
        :return: Numpy array of shape (n_points, ) where an element will be 1 if the corresponding input satisfies all
                 constraints and zero if any constraint is violated
        """
        if self.constraint_matrix.shape[1] != x.shape[1]:
            raise ValueError(
                "Dimension mismatch between constraint matrix (second dim )"
                + " and input x (second dim )".format(self.constraint_matrix.shape[1], x.shape[1])
            )

        # Transpose here is needed to handle input dimensions
        # that is, A is (n_const, n_dims) and x is (n_points, n_dims)
        ax = self.constraint_matrix.dot(x.T).T
        return np.all((ax >= self.lower_bound) & (ax <= self.upper_bound), axis=1)'''
    ,
    '''
    def classification_metrics(ground_truth: Dict, retrieved: Dict) -> np.ndarray:
    """
    Given ground truth dictionary and retrieved dictionary, return per class precision, recall and f1 score. Class 1 is
    assigned to duplicate file pairs while class 0 is for non-duplicate file pairs.

    Args:
        ground_truth: A dictionary representing ground truth with filenames as key and a list of duplicate filenames
        as value.
        retrieved: A dictionary representing retrieved duplicates with filenames as key and a list of retrieved
        duplicate filenames as value.

    Returns:
        Dictionary of precision, recall and f1 score for both classes.
    """
    all_pairs = _make_all_unique_possible_pairs(ground_truth)
    ground_truth_duplicate_pairs, retrieved_duplicate_pairs = _make_positive_duplicate_pairs(
        ground_truth, retrieved
    )
    y_true, y_pred = _prepare_labels(
        all_pairs, ground_truth_duplicate_pairs, retrieved_duplicate_pairs
    )
    logger.info(classification_report(y_true, y_pred))
    prec_rec_fscore_support = dict(
        zip(
            ('precision', 'recall', 'f1_score', 'support'),
            precision_recall_fscore_support(y_true, y_pred),
        )
    )
    return prec_rec_fscore_support
    ''',
    
    '''
    def kl_div_loss_2d(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:

        return _reduce_loss(_js_div_2d(target, pred), reduction)
    '''
)
