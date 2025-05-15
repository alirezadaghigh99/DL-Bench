pre_post = (
'''
# ========================================================================
# 1) create_tiles
# ========================================================================
def create_tiles(
    images: List[np.ndarray],
    grid_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
    single_tile_size: Optional[Tuple[int, int]] = None,
    tile_scaling: Literal["min", "max", "avg"] = "avg",
    tile_padding_color: Tuple[int, int, int] = (0, 0, 0),
    tile_margin: int = 15,
    tile_margin_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    A function to combine multiple images into a grid (tiled) image.
    
    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Input Validation:
        - If there are no images given (images list is empty), we cannot create any tiled image.
          We immediately raise a ValueError to inform the user about the issue.
        
    (2) Determine Single Tile Size:
        - If single_tile_size is not explicitly provided, we compute a common size for each image
          by looking at the shapes of all images and aggregating them (e.g., via 'avg', 'min', 'max').
          This step ensures that when we place images on the grid, they have a consistent tile size.
        
    (3) Resize Images with Letterboxing:
        - We loop over each image and resize it to the chosen single_tile_size.
        - 'letterbox_image' pads (or "letterboxes") each image so that it fits the desired dimensions
          while preserving its aspect ratio. The extra space is filled with 'tile_padding_color'.
        
    (4) Establish Grid Dimensions:
        - If grid_size (a tuple of (rows, columns)) is not provided, we attempt to calculate it in a way
          that fits all images nicely (e.g., based on the total number of images).
        - This ensures we have a suitable grid layout, for instance (3 rows, 4 columns).
        
    (5) Grid Capacity Check:
        - We verify that the product of grid rows and columns can accommodate all images in 'images'.
          If there are too many images for the chosen grid, raise a ValueError to avoid truncation
          or overflow.
        
    (6) Generate Tiled Image:
        - Finally, we pass the list of resized images, along with metadata about margins, padding,
          and colors, to an internal function '_generate_tiles' which actually assembles the final
          large image. That returned result is then returned to the caller.
    -----------------------------------------------------------------------------
    """

    # 1. Check for empty images list
    if len(images) == 0:
        raise ValueError("Could not create image tiles from empty list of images.")

    # 2. Determine the tile size if not provided
    if single_tile_size is None:
        single_tile_size = _aggregate_images_shape(images=images, mode=tile_scaling)

    # 3. Resize each image to match the single tile size, adding padding as needed
    resized_images = [
        letterbox_image(
            image=i,
            desired_size=single_tile_size,
            color=tile_padding_color
        )
        for i in images
    ]

    # 4. Decide the grid size (rows, columns) if it wasn't explicitly set
    grid_size = _establish_grid_size(images=images, grid_size=grid_size)

    # 5. Ensure the chosen grid can contain all images
    if len(images) > grid_size[0] * grid_size[1]:
        raise ValueError(f"Grid of size {grid_size} cannot fit {len(images)} images.")

    # 6. Assemble the tiled image and return it
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

# ========================================================================
# 2) _iterate_shift_rule_with_multipliers
# ========================================================================
def _iterate_shift_rule_with_multipliers(rule, order, period):
    r"""
    A helper method to repeat a "shift rule" multiple times for higher-order derivatives,
    taking into account multipliers and shifts in each step.

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Initialize a list to store the combined rules (each element will have
        [coefficient_product, multiplier_product, cumulative_shift]).
        
    (2) Generate All Combinations:
        - We want every possible combination of the base 'rule' repeated 'order' times.
        - We use itertools.product(rule, repeat=order) to create the Cartesian product.
          For example, if rule = [[c1, m1, s1], [c2, m2, s2]] and order=2, we get
          combinations like [[c1, m1, s1], [c1, m1, s1]], [[c1, m1, s1], [c2, m2, s2]], etc.
        
    (3) Extract Coeffs, Multipliers, and Shifts:
        - Each partial_rules item is a tuple of 'order' arrays (c, m, s). We stack them
          and separate them into three arrays (for coefficients, multipliers, shifts).
        
    (4) Compute Cumulative Shift:
        - For each combination, we iterate through the multipliers and shifts, updating
          a 'cumul_shift' value. The logic is:
            cumul_shift *= (multiplier of the current step)
            cumul_shift += (shift of the current step)
          If 'period' is provided, we apply modular arithmetic to keep the shift within
          [-period/2, period/2].
        
    (5) Combine and Store:
        - We compute the product of the coefficients (np.prod(c)) and the product of
          the multipliers (np.prod(m)) to represent how everything scales up.
        - We store [coefficient_product, multiplier_product, cumul_shift] as one entry.
        
    (6) After processing all combinations, we stack them into a single array and return.
    -----------------------------------------------------------------------------
    """

    # 1. Initialize list for storing results
    combined_rules = []

    # 2. Generate all possible sequences of rule repeated 'order' times
    for partial_rules in itertools.product(rule, repeat=order):
        # 'partial_rules' might look like ([c1, m1, s1], [c2, m2, s2], ...)
        # Stack them vertically, then transpose to get c, m, s separated
        c, m, s = np.stack(partial_rules).T

        # 4. Compute cumulative shift
        cumul_shift = 0.0
        for _m, _s in zip(m, s):
            cumul_shift *= _m
            cumul_shift += _s

        # If a period is provided, keep cumul_shift within the range of [-period/2, period/2]
        if period is not None:
            cumul_shift = np.mod(cumul_shift + 0.5 * period, period) - 0.5 * period

        # 5. Compute products for coefficients and multipliers, store results
        combined_rules.append(np.stack([np.prod(c), np.prod(m), cumul_shift]))

    # 6. Stack everything together and return
    return qml.math.stack(combined_rules)

'''
,
'''

# ========================================================================
# 3) affine
# ========================================================================
def affine(
    img: Image.Image,
    matrix: List[float],
    interpolation: int = Image.NEAREST,
    fill: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
) -> Image.Image:
    """
    Applies an affine transformation to a PIL Image using a 3x3 (in practice a 2x3 for PIL) matrix.

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Input Type Check:
        - First, we confirm that 'img' is indeed a PIL Image.
          If not, we raise a TypeError to ensure we don't call transform on a non-PIL object.
        
    (2) Determine Output Size:
        - We want the transformed image to maintain the same size as the input (unless specified otherwise).
          So we set 'output_size' = img.size.
        
    (3) Parse Fill Parameter:
        - The fill option can be an integer, float, or a sequence of these types. 
          We convert this into an appropriate format that PIL expects, which may involve
          distinguishing between single-channel, RGB, or RGBA images.
        
    (4) Perform the Affine Transformation:
        - We call 'img.transform' with the AFFINE mode. 
          The provided 'matrix' is typically in the format [a, b, c, d, e, f], corresponding to:
              x' = a*x + b*y + c
              y' = d*x + e*y + f
          'interpolation' can be NEAREST, BILINEAR, BICUBIC, etc., controlling how pixel 
          values are sampled.
        
    (5) Return the Transformed Image:
        - The 'transform' call yields a new PIL Image object that we return to the caller.
    -----------------------------------------------------------------------------
    """

    # 1. Ensure 'img' is a valid PIL image
    if not _is_pil_image(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    # 2. Use the current size of the image as the output size
    output_size = img.size

    # 3. Convert 'fill' into a dictionary understood by PIL (if necessary)
    opts = _parse_fill(fill, img)

    # 4. Call the PIL 'transform' function with the specified matrix and interpolation
    return img.transform(output_size, Image.AFFINE, matrix, interpolation, **opts)

'''

)


training = (
    '''
    # ========================================================================
# 1) fit
# ========================================================================
def fit(
    self,
    train_loader: DataLoader,
    override: bool = True,
    progress_bar: bool = False,
) -> None:
    """
    Fit the local Laplace approximation at the parameters of the model.

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Parameter Checking and Constraints:
        - The argument `override` here must be True for this implementation,
          otherwise a ValueError is raised. The reason is that this particular
          last-layer Laplace approximation does not support cumulative/online
          updates from previous fits (`override=False` would imply that).

    (2) Model Evaluation Mode:
        - We set `self.model.eval()` to ensure the model is in inference mode
          (i.e., batchnorm/dropout layers behave consistently for approximation).

    (3) Last-Layer Initialization (If Needed):
        - If the model's last layer is not yet identified (i.e., `self.model.last_layer is None`),
          we fetch a single batch from `train_loader` to determine the shape or structure needed.
        - We call `self._find_last_layer(self.data)` internally to detect the layer we want to approximate.
        - Then, we get the parameters from the last layer using `parameters_to_vector`.
        - We record how many parameters (`self.n_params`) and layers (`self.n_layers`) there are.
        - We retrieve and set the prior precision and mean, which govern the Laplace approximation's prior.
        - Finally, we call `self._init_H()` to initialize the Hessian or related approximations.

    (4) Call Parent Class Fit:
        - Once everything is set, we call `super().fit(train_loader, override=override)`.
          This presumably carries out additional steps in the parent class (like accumulating curvature).

    (5) Store the Mean of the Last-Layer:
        - After fitting, we retrieve the last-layer parameters into `self.mean`.
        - If `self.enable_backprop` is False, we detach `self.mean` from the computation graph,
          ensuring no gradient computations will flow from it.

    (6) Summary:
        - This process sets up or updates the Laplace approximation for the last layer of the model.
          The final approximate posterior mean is stored in `self.mean`.
    -----------------------------------------------------------------------------

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Iterates over training batches (X, y). We rely on it also for dataset size (N).
    override : bool, default=True
        Whether to re-initialize Hessian (H), loss trackers, etc. Must be True here.
    progress_bar : bool, default=False
        Toggles a progress bar display (if implemented in the parent class).
    """

    # 1. Enforce that override must be True for this last-layer approach
    if not override:
        raise ValueError(
            "Last-layer Laplace approximations do not support `override=False`."
        )

    # 2. Switch the model to evaluation mode
    self.model.eval()

    # 3. Initialize last layer if it's missing
    if self.model.last_layer is None:
        # fetch a batch from train_loader for dimensional analysis
        self.data: tuple[torch.Tensor, torch.Tensor] | MutableMapping = next(
            iter(train_loader)
        )
        self._find_last_layer(self.data)
        # convert last-layer params into a single vector
        params: torch.Tensor = parameters_to_vector(self.model.last_layer.parameters()).detach()
        self.n_params: int = len(params)
        self.n_layers: int = len(list(self.model.last_layer.parameters()))
        # set prior precision and mean
        self.prior_precision: float | torch.Tensor = self._prior_precision
        self.prior_mean: float | torch.Tensor = self._prior_mean
        self._init_H()

    # 4. Call parent fit routine for potential curvature accumulation
    super().fit(train_loader, override=override)

    # 5. Store final last-layer mean; detach if we don't need autograd
    self.mean: torch.Tensor = parameters_to_vector(self.model.last_layer.parameters())
    if not self.enable_backprop:
        self.mean = self.mean.detach()
    
    ''',
    
    ''' 
    
# ========================================================================
# 2) dropout
# ========================================================================
def dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    r"""
    During training, randomly zeroes some elements of the input tensor with probability `p`.

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Torch Function Check:
        - We first check whether `input` might be a "TorchFunction" or carry special
          dispatch logic. If so, we hand off to `handle_torch_function`.

    (2) Probability Validation:
        - The probability `p` must be between 0.0 and 1.0. If it's outside this range,
          we raise a ValueError.

    (3) Forward Logic:
        - If `training` is True, we apply dropout by zeroing out elements of `input`
          with probability `p`. If `training` is False, the function returns the input
          unmodified (effectively no dropout).
        - In-place vs. out-of-place:
          - If `inplace=True`, the operation modifies `input` directly.
          - Otherwise, it creates a new tensor.

    (4) Return the Updated Tensor:
        - The operation uses a Bernoulli sampling approach under the hood. The result
          is an output tensor where elements are zeroed out randomly, scaled appropriately
          to maintain expected values during training.
    -----------------------------------------------------------------------------

    Args:
        input (Tensor): The input tensor to which dropout will be applied.
        p (float, optional): Probability of an element to be zeroed. Default is 0.5.
        training (bool, optional): If True, applies dropout; if False, pass data as-is.
        inplace (bool, optional): If True, do the operation in-place on `input`.
    """

    # 1. Special Torch function check
    if has_torch_function_unary(input):
        return handle_torch_function(
            dropout, (input,), input, p=p, training=training, inplace=inplace
        )
    # 2. Validate probability p
    if p < 0.0 or p > 1.0:
        raise ValueError(
            "dropout probability has to be between 0 and 1, but got p"
        )

    # 3. Forward logic: apply dropout conditionally
    return (
        _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)
    )

    '''
    ,
    '''
    
# ========================================================================
# 3) spatial_gradient
# ========================================================================
def spatial_gradient(
    input: Tensor,
    mode: str = "sobel",
    order: int = 1,
    normalized: bool = True
) -> Tensor:
    r"""
    Compute the image derivatives (in both x and y directions) using a Sobel or diff operator.

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Basic Checks:
        - Verify `input` is a Tensor: `KORNIA_CHECK_IS_TENSOR(input)`
        - Check that the shape of `input` matches (B, C, H, W).

    (2) Get or Construct a Gradient Kernel:
        - Depending on `mode` ('sobel' or 'diff') and `order`, retrieve a convolution kernel
          via `get_spatial_gradient_kernel2d(...)`.
        - If `normalized` is True, we scale or normalize the kernel so that sums of weights
          match expected values.

    (3) Reshape and Pad:
        - We flatten the batch and channel dimensions into one (b * c, 1, h, w) for the
          convolution operation.
        - We apply a "replicate" pad around the spatial dims to ensure valid convolution
          without reducing image size. The padding size depends on the kernel shape.

    (4) Apply Convolution (conv2d):
        - We convolve the padded input with the kernel(s) in `tmp_kernel`.
        - For a Sobel operator, we might produce 2 channels (dx, dy) if order=1,
          or 3 channels if order=2 (including cross-derivatives or second-order derivatives).

    (5) Reshape and Return:
        - We then reshape the output back to (B, C, out_channels, H, W). The `out_channels`
          dimension is typically 2 (for dx and dy) or possibly 3 if higher order.
    -----------------------------------------------------------------------------

    Args:
        input (Tensor): Image tensor with shape (B, C, H, W).
        mode (str, optional): Derivative mode, 'sobel' or 'diff'. Default is 'sobel'.
        order (int, optional): The order of derivatives to compute. Default is 1.
        normalized (bool, optional): Whether to normalize the kernel for consistent amplitude.
                                     Default is True.

    Returns:
        Tensor: The spatial derivatives, shape (B, C, 2, H, W) for first order,
        or (B, C, 3, H, W) if order=2.

    Example:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = spatial_gradient(input)  # 1x3x2x4x4
        >>> output.shape
        torch.Size([1, 3, 2, 4, 4])
    """

    # 1. Basic input shape checks
    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    # 2. Construct or retrieve gradient kernel
    kernel = get_spatial_gradient_kernel2d(mode, order, device=input.device, dtype=input.dtype)
    if normalized:
        kernel = normalize_kernel2d(kernel)

    # 3. Reshape input & pad
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...]  # shape to match conv2d expectations
    spatial_pad = [
        kernel.size(1) // 2,
        kernel.size(1) // 2,
        kernel.size(2) // 2,
        kernel.size(2) // 2,
    ]
    out_channels: int = 3 if order == 2 else 2
    padded_inp: Tensor = pad(input.reshape(b * c, 1, h, w), spatial_pad, "replicate")

    # 4. Convolution
    out = F.conv2d(padded_inp, tmp_kernel, groups=1, padding=0, stride=1)

    # 5. Reshape back to original batch, channel, derivative dims
    return out.reshape(b, c, out_channels, h, w)
    '''
    
)

model = (
    ''' 
    # ========================================================================
# 1) camera_position_from_spherical_angles
# ========================================================================
def camera_position_from_spherical_angles(
    distance: float,
    elevation: float,
    azimuth: float,
    degrees: bool = True,
    device: Device = "cpu",
) -> torch.Tensor:
    """
    Calculate the location of the camera based on the distance from
    the target point and the spherical angles (elevation, azimuth).

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Input Broadcasting:
        - We call `convert_to_tensors_and_broadcast(distance, elevation, azimuth, device=device)`
          to ensure that all inputs are tensors and that they share the same shape if possible.
          This step allows for vectorized operations over multiple distances/elevations/azimuths.

    (2) Degrees to Radians Conversion:
        - If `degrees=True`, we convert both `elev` and `azim` from degrees to radians:
          radians = (π / 180) * degrees.

    (3) Calculate Cartesian Coordinates:
        - Assume a spherical coordinate system:
            x = r * cos(elev) * sin(azim)
            y = r * sin(elev)
            z = r * cos(elev) * cos(azim)
          Here, r = distance, elev = elevation angle, azim = azimuth angle.

    (4) Stack Resulting Coordinates:
        - We combine (x, y, z) into a single tensor of shape (*, 3). If the broadcasting
          produced multiple values, each row in the final tensor corresponds to one camera position.

    (5) Ensure Batch Dimension:
        - If the result is a zero-dimensional tensor, reshape it to (1, -1).
          Otherwise, finalize the shape to (-1, 3) where -1 is inferred from the current batch size.

    (6) Return:
        - The final tensor of shape (N, 3), representing camera positions in 3D space.
    -----------------------------------------------------------------------------
    """

    # 1. Broadcast inputs
    broadcasted_args = convert_to_tensors_and_broadcast(
        distance, elevation, azimuth, device=device
    )
    dist, elev, azim = broadcasted_args

    # 2. Convert angles to radians if needed
    if degrees:
        elev = math.pi / 180.0 * elev
        azim = math.pi / 180.0 * azim

    # 3. Compute Cartesian coordinates based on spherical geometry
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)

    # 4. Stack the coordinates
    camera_position = torch.stack([x, y, z], dim=1)

    # 5. Ensure we have a proper batch dimension
    if camera_position.dim() == 0:
        camera_position = camera_position.view(1, -1)
    return camera_position.view(-1, 3)

    ''',
    ''' 
    

# ========================================================================
# 2) gauss_kl
# ========================================================================
def gauss_kl(
    q_mu: TensorType, 
    q_sqrt: TensorType, 
    K: TensorType = None, 
    *, 
    K_cholesky: TensorType = None
) -> tf.Tensor:
    r"""
    Compute the KL divergence KL[q(x) || p(x)] where

      q(x) = N(q_mu, q_sqrt^2)
    
    and
    
      p(x) = N(0, K)  if K is not None
             N(0, I)  if K is None.

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Check Ambiguity of K vs K_cholesky:
        - We ensure that the caller does not provide both K and K_cholesky. 
          If both are given, we raise a ValueError because it's unclear which one to use.

    (2) Determine White vs Non-White Case:
        - If both K and K_cholesky are None, we assume p(x) = N(0, I) ("white" case).
        - Otherwise, we use the provided covariance or its Cholesky factor.

    (3) Extract Dimensionality:
        - We look at q_mu's shape: (M, L) where M is batch size or dimension, 
          and L might be the number of latent processes or something similar.
        - We also check if q_sqrt is diagonal or full-rank by checking its dimensions.

    (4) Compute alpha = Σp^-1 q_mu:
        - If p(x) = N(0, I) (white case), then Σp^-1 = Identity, so alpha = q_mu.
        - If we have K or K_cholesky, we compute Lp = Cholesky(K) if needed, 
          and solve alpha = Lp^-1 * q_mu, effectively applying Σp^-1 to q_mu.

    (5) Collect Terms Needed for KL:
        - Mahalanobis term: q_muᵀ Σp^-1 q_mu = sum of squares(alpha).
        - logdet_qcov: the log-determinant of q(x)'s covariance, derived from q_sqrt.
        - trace term: tr(Σp^-1 Σq). This requires additional triangular solves if p is not I.

    (6) Combine Terms into twoKL (2 × KL):
        - twoKL = mahalanobis + constant - logdet_qcov + trace
        - constant = - L * M (related to dimensionality)
        - If p is not I, we also add log|K| to complete the KL formula.

    (7) Return 0.5 * twoKL:
        - The final KL is half of this sum, matching the standard formula for Gaussian KL.

    (8) This function is typically used in Bayesian or GP contexts to measure how
        the distribution q differs from the prior p.
    -----------------------------------------------------------------------------
    """

    # 1. Ensure only one of K or K_cholesky is used
    if (K is not None) and (K_cholesky is not None):
        raise ValueError(
            "Ambiguous arguments: gauss_kl() must only be passed one of `K` or `K_cholesky`."
        )

    # 2. Distinguish cases
    is_white = (K is None) and (K_cholesky is None)
    is_diag = len(q_sqrt.shape) == 2

    M, L = tf.shape(q_mu)[0], tf.shape(q_mu)[1]

    # 4. Compute alpha = Σp^-1 q_mu
    if is_white:
        # p(x) = N(0, I), so alpha is just q_mu
        alpha = q_mu
    else:
        # p(x) = N(0, K), use Cholesky factor to solve
        if K is not None:
            Lp = tf.linalg.cholesky(K)
        elif K_cholesky is not None:
            Lp = K_cholesky

        is_batched = len(Lp.shape) == 3  # might have multiple latent GPs
        q_mu = tf.transpose(q_mu)[:, :, None] if is_batched else q_mu
        alpha = tf.linalg.triangular_solve(Lp, q_mu, lower=True)

    # 5. Distinguish diagonal vs. full-rank q_sqrt
    if is_diag:
        Lq = Lq_diag = q_sqrt
        Lq_full = tf.linalg.diag(tf.transpose(q_sqrt))
    else:
        Lq = Lq_full = tf.linalg.band_part(q_sqrt, -1, 0)
        Lq_diag = tf.linalg.diag_part(Lq)

    # Mahalanobis term
    mahalanobis = tf.reduce_sum(tf.square(alpha))

    # Constant term
    constant = -to_default_float(tf.size(q_mu, out_type=tf.int64))

    # log-determinant of q(x)
    logdet_qcov = tf.reduce_sum(tf.math.log(tf.square(Lq_diag)))

    # Trace term
    if is_white:
        trace = tf.reduce_sum(tf.square(Lq))
    else:
        if is_diag and not is_batched:
            # fast path for K is [M, M] and q_sqrt is [M, L]
            LpT = tf.transpose(Lp)
            Lp_inv = tf.linalg.triangular_solve(Lp, tf.eye(M, dtype=default_float()), lower=True)
            K_inv = tf.linalg.diag_part(
                tf.linalg.triangular_solve(LpT, Lp_inv, lower=False)
            )[:, None]
            trace = tf.reduce_sum(K_inv * tf.square(q_sqrt))
        else:
            # handle batched or full-rank scenarios
            if is_batched or Version(tf.__version__) >= Version("2.2"):
                Lp_full = Lp
            else:
                # fallback for older TF versions
                Lp_full = tf.tile(tf.expand_dims(Lp, 0), [L, 1, 1])
            LpiLq = tf.linalg.triangular_solve(Lp_full, Lq_full, lower=True)
            trace = tf.reduce_sum(tf.square(LpiLq))

    # Combine
    twoKL = mahalanobis + constant - logdet_qcov + trace

    # If not white, add log|K|
    if not is_white:
        log_sqdiag_Lp = tf.math.log(tf.square(tf.linalg.diag_part(Lp)))
        sum_log_sqdiag_Lp = tf.reduce_sum(log_sqdiag_Lp)
        scale = 1.0 if is_batched else to_default_float(L)
        twoKL += scale * sum_log_sqdiag_Lp

    return 0.5 * twoKL


    ''',
    '''
    
# ========================================================================
# 3) create_knn_graph_and_index
# ========================================================================
def create_knn_graph_and_index(
    features: Optional[FeatureArray],
    *,
    n_neighbors: Optional[int] = None,
    metric: Optional[Metric] = None,
    correct_exact_duplicates: bool = True,
    **sklearn_knn_kwargs,
) -> Tuple[csr_matrix, NearestNeighbors]:
    """
    Calculate the KNN graph from the features if not already provided, and create
    the corresponding NearestNeighbors index.

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Instantiate NearestNeighbors:
        - We call `features_to_knn(...)` with the given `features`, `n_neighbors`, `metric`,
          and any additional kwargs (`sklearn_knn_kwargs`). This returns a configured 
          NearestNeighbors object `knn`.

    (2) Build the KNN Graph:
        - We then invoke `construct_knn_graph_from_index(knn)`, which uses the
          NearestNeighbors model to produce a sparse adjacency matrix (`csr_matrix`)
          where entry (i, j) holds the distance from sample i to its j-th neighbor.
        - Typically, only the nearest neighbors are stored, making this matrix sparse.

    (3) Correct for Exact Duplicates (Optionally):
        - If `correct_exact_duplicates=True`, we call `correct_knn_graph(...)` to ensure that
          identical points in `features` do not get improperly excluded or mishandled 
          in the KNN adjacency. This helps maintain consistency when exact duplicates exist.

    (4) Return Graph and Index:
        - We return both the constructed KNN graph (sparse matrix) and the NearestNeighbors
          object. These can be used for further queries or distance lookups.

    (5) Example Use Cases:
        - This function is helpful for clustering, nearest-neighbor classification,
          or other graph-based operations where we need a consistent adjacency structure
          derived from feature vectors.
    -----------------------------------------------------------------------------

    Returns
    -------
    (knn_graph, knn) : Tuple[csr_matrix, NearestNeighbors]
        - knn_graph: A sparse matrix of shape (n_samples, n_samples) containing distances
          to the k nearest neighbors. Off-diagonal zero entries indicate no direct
          neighbor link.
        - knn: The trained NearestNeighbors object that can be used for further queries.

    Examples
    --------
    >>> import numpy as np
    >>> from cleanlab.internal.neighbor.knn_graph import create_knn_graph_and_index
    >>> features = np.array([
    ...     [0.701, 0.701],
    ...     [0.900, 0.436],
    ...     [0.000, 1.000],
    ... ])
    >>> knn_graph, knn = create_knn_graph_and_index(features, n_neighbors=1)
    >>> knn_graph.toarray()
    array([[0.        , 0.33140006, 0.        ],
           [0.33140006, 0.        , 0.        ],
           [0.76210367, 0.        , 0.        ]])
    >>> knn
    NearestNeighbors(metric=<function euclidean ...>, n_neighbors=1)
    """

    # 1. Construct NearestNeighbors object
    knn = features_to_knn(features, n_neighbors=n_neighbors, metric=metric, **sklearn_knn_kwargs)

    # 2. Build graph from NearestNeighbors object
    knn_graph = construct_knn_graph_from_index(knn)

    # 3. Optionally correct for exact duplicates
    if correct_exact_duplicates:
        assert features is not None
        knn_graph = correct_knn_graph(features, knn_graph)

    # 4. Return the adjacency matrix and the KNN model
    return knn_graph, knn
    '''
)

infer = (
    '''# ========================================================================
# 3) forward
# ========================================================================
def forward(
    self,
    X: Tensor,
    prior: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    A forward pass through the model, returning a transformed embedding and a loss term.

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Embedding Extraction:
        - We call `self._get_embeddings(X)` to transform the raw input X into 
          an internal representation (embedding). The shape or meaning of 'x'
          depends on the model architecture.

    (2) Encoder Processing:
        - `steps_output, M_loss = self.encoder(x, prior)` suggests that `encoder`
          may produce multiple intermediate outputs (e.g., from different steps
          in a sequential or iterative process) and also a loss term `M_loss`.
        - `steps_output` could be a list or tuple of intermediate embeddings.

    (3) Summation:
        - We sum the stack of `steps_output` along dimension 0 (`torch.sum(torch.stack(steps_output, dim=0), dim=0)`),
          thus aggregating outputs from each step into a single tensor `res`.

    (4) Return Outputs:
        - We return a tuple `(res, M_loss)`, where `res` is the final aggregated embedding,
          and `M_loss` is presumably a measure of how well the encoding matched some prior
          or some loss function computed inside `encoder`.

    (5) Usage:
        - This method is typically called in a training or inference loop to get both 
          the model output and the relevant loss for backpropagation (if training).
    -----------------------------------------------------------------------------
    """

    # 1. Extract embeddings
    x = self._get_embeddings(X)

    # 2. Pass through encoder with optional prior
    steps_output, M_loss = self.encoder(x, prior)

    # 3. Sum all step outputs
    res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

    # 4. Return final aggregated output and loss
    return (res, M_loss)''',
    
    '''# ========================================================================
# 2) predict_proba
# ========================================================================
def predict_proba(self, *args, **kwargs) -> np.ndarray:
    """
    Predict class probabilities using the wrapped classifier `clf`.

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Default vs. Custom Classifier:
        - The instance may have a `_default_clf` indicating if we should handle
          input data X in a specific manner (like enforcing 2D shape).
        - If `_default_clf` is True, we parse X from either `args` or `kwargs`
          (i.e., `X = args[0]` or `X = kwargs["X"]`). If not provided, raise ValueError.

    (2) Data Reshaping:
        - Once we have X, we call `force_two_dimensions(X)` to ensure X is at least 2D
          (i.e., shape `(N, M)`).

    (3) Classifier’s `predict_proba`:
        - If `_default_clf` is True, we create a new argument tuple with the properly
          shaped X and pass it to `self.clf.predict_proba(...)`.
        - Otherwise, we directly call `self.clf.predict_proba(*args, **kwargs)`.

    (4) Output:
        - Returns an `(N, K)` numpy array of probabilities, where N is the number of samples
          and K is the number of classes. 
    -----------------------------------------------------------------------------
    """

    # 1. Check if we have a default_clf scenario
    if self._default_clf:
        # We parse X from args or kwargs
        if args:
            X = args[0]
        elif "X" in kwargs:
            X = kwargs["X"]
            del kwargs["X"]  # remove from kwargs after extraction
        else:
            raise ValueError("No input provided to predict_proba. Please provide X.")

        # 2. Enforce two-dimensional input
        X = force_two_dimensions(X)
        new_args = (X,) + args[1:]

        # 3. Run classifier predict_proba
        return self.clf.predict_proba(*new_args, **kwargs)
    else:
        # If not default, pass everything directly
        return self.clf.predict_proba(*args, **kwargs)
''',

'''# ========================================================================
# 1) find_fundamental
# ========================================================================
def find_fundamental(
    points1: Tensor,
    points2: Tensor,
    weights: Optional[Tensor] = None,
    method: Literal["8POINT", "7POINT"] = "8POINT",
) -> Tensor:
    r"""
    Calculates the fundamental matrix (or matrices) from two sets of corresponding points
    in two images, potentially with per-correspondence weights.

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Input Points Validation:
        - We assume points1, points2 each have shape (B, N, 2) where B is the batch dimension,
          N is the number of point correspondences. N must be >= 7 (for 7-point) or >= 8 (for 8-point).
        - Optionally, weights can be provided (shape (B, N)) to influence the 8-point algorithm.

    (2) Method Selection:
        - If method is "7POINT", we call `run_7point(...)`.
          This algorithm finds the fundamental matrix satisfying the epipolar constraints with 7 points.
        - If method is "8POINT", we call `run_8point(...)`, possibly using the provided `weights`.
          This algorithm typically requires 8 or more points for a linear solution.

    (3) Return Format:
        - The output shape is (B, 3*m, 3), where m is the number of solutions found.
          (In the 7-point case, there can be up to three possible solutions.)

    (4) Error Checking:
        - If an unsupported method is supplied, we raise a ValueError.

    (5) Summary:
        - This function acts as a high-level dispatcher to either the 7- or 8-point fundamental
          matrix computation, returning a batched set of fundamental matrix solutions.
    -----------------------------------------------------------------------------
    """

    # 2. Choose algorithm based on method
    if method.upper() == "7POINT":
        result = run_7point(points1, points2)
    elif method.upper() == "8POINT":
        result = run_8point(points1, points2, weights)
    else:
        # 4. Raise error if invalid method
        raise ValueError("Invalid method. Supported methods are '7POINT' and '8POINT'.")
    return result
'''
)

eval = (
    '''# ========================================================================
# 1) evaluate
# ========================================================================
def evaluate(self, x: np.ndarray) -> np.ndarray:
    """
    Evaluate whether constraints are violated or satisfied at a set of x locations.

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Dimension Consistency Check:
        - Verify that the number of columns (dimensions) in `self.constraint_matrix`
          matches the second dimension of `x`.
        - If there's a mismatch, raise a ValueError.

    (2) Matrix Multiplication:
        - We compute `ax = self.constraint_matrix.dot(x.T).T`.
        - Here, `self.constraint_matrix` has shape (n_constraints, n_dims), while
          `x.T` has shape (n_dims, n_points). 
        - The transpose operations ensure correct alignment so the result `ax`
          ends up with shape (n_points, n_constraints).

    (3) Compare Against Bounds:
        - We check element-wise whether `ax >= self.lower_bound` and `ax <= self.upper_bound`.
        - Both `self.lower_bound` and `self.upper_bound` are broadcast over each constraint.

    (4) Overall Constraint Satisfaction:
        - We apply `np.all(..., axis=1)`, meaning for each row in `ax` (i.e., for each point),
          we determine if *all* constraints are satisfied.
        - The function returns an array of shape (n_points,) with boolean-like values:
            1 if all constraints are met, 0 otherwise (when cast or interpreted appropriately).
    -----------------------------------------------------------------------------

    :param x: Array of shape (n_points, n_dims) containing input locations to evaluate.
    :return: A numpy array of shape (n_points,) with 1 indicating constraints satisfied,
             and 0 otherwise.
    """

    # (1) Check dimension consistency
    if self.constraint_matrix.shape[1] != x.shape[1]:
        raise ValueError(
            "Dimension mismatch between constraint matrix (shape[1]) "
            "and input x (shape[1]). "
            f"Expected {self.constraint_matrix.shape[1], got {x.shape[1]."
        )

    # (2) Matrix multiplication: A (n_constraints, n_dims) * x^T (n_dims, n_points)
    ax = self.constraint_matrix.dot(x.T).T  # shape: (n_points, n_constraints)

    # (3) Check bounds
    within_lower = ax >= self.lower_bound
    within_upper = ax <= self.upper_bound

    # (4) Return boolean array of overall satisfaction, cast to True/False or 1/0
    return np.all(within_lower & within_upper, axis=1)
''',
'''
# ========================================================================
# 2) classification_metrics
# ========================================================================
def classification_metrics(ground_truth: Dict, retrieved: Dict) -> np.ndarray:
    """
    Compute precision, recall, and F1 score for duplicate vs. non-duplicate classification.

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Gather All Unique Pairs:
        - `_make_all_unique_possible_pairs(ground_truth)` enumerates every unique 
          (fileA, fileB) pair across the dataset.

    (2) Identify Duplicate Pairs:
        - `ground_truth_duplicate_pairs` is the set of true duplicate pairs from the ground truth dictionary.
        - `retrieved_duplicate_pairs` is the set of predicted duplicate pairs from the retrieved dictionary.

    (3) Prepare Labels for Classification:
        - `_prepare_labels(...)` returns two arrays `y_true` and `y_pred`, each indicating class
          labels for all pairs:
            - `y_true[i] = 1` if pair i is a true duplicate, 0 otherwise
            - `y_pred[i] = 1` if the pair i is predicted as duplicate, 0 otherwise

    (4) Evaluate & Log Metrics:
        - We use `classification_report(y_true, y_pred)` to log a summary of precision, recall, F1.
        - Additionally, we compute `precision_recall_fscore_support(y_true, y_pred)` to get
          metric details for each class (0 and 1).
        - We build a dictionary containing `'precision'`, `'recall'`, `'f1_score'`, `'support'` 
          for each class.

    (5) Return:
        - A dictionary `prec_rec_fscore_support` with the computed metrics for both classes (duplicate vs non-duplicate).
    -----------------------------------------------------------------------------

    Args:
        ground_truth: Dict with filenames as keys and lists of duplicate filenames as values.
        retrieved: Dict with filenames as keys and lists of retrieved duplicate filenames as values.

    Returns:
        A dictionary containing arrays for precision, recall, F1 score, and support for each class.
    """

    # (1) Make all possible unique pairs
    all_pairs = _make_all_unique_possible_pairs(ground_truth)

    # (2) Get duplicate pairs from ground truth and retrieved sets
    ground_truth_duplicate_pairs, retrieved_duplicate_pairs = _make_positive_duplicate_pairs(
        ground_truth, retrieved
    )

    # (3) Prepare true labels and predicted labels
    y_true, y_pred = _prepare_labels(
        all_pairs, ground_truth_duplicate_pairs, retrieved_duplicate_pairs
    )

    # (4) Log classification metrics
    logger.info(classification_report(y_true, y_pred))

    prec_rec_fscore_support = dict(
        zip(
            ('precision', 'recall', 'f1_score', 'support'),
            precision_recall_fscore_support(y_true, y_pred),
        )
    )
    return prec_rec_fscore_support''',
    '''
# ========================================================================
# 3) kl_div_loss_2d
# ========================================================================
def kl_div_loss_2d(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """
    Compute the Kullback-Leibler divergence (KL-div) in 2D, then reduce the result (e.g., mean).

    Comprehensive Explanation (Step by Step):
    -----------------------------------------------------------------------------
    (1) Underlying JS or KL Computation:
        - This function internally calls `_js_div_2d(target, pred)`, which presumably
          calculates the Jensen-Shannon divergence or a variant of KL across 2D distributions.

    (2) Reduction:
        - We then pass this divergence score to `_reduce_loss(..., reduction)`.
        - The `reduction` argument can be `"mean"`, `"sum"`, or possibly `"none"`.
          - `"mean"` aggregates the loss over all elements.
          - `"sum"` sums over all elements.
          - `"none"` returns the per-element or per-batch loss.

    (3) Return Final Loss Value:
        - The final scalar or tensor (depending on reduction) is returned as the KL divergence measure.
    -----------------------------------------------------------------------------

    Args:
        pred: Predicted distribution/tensor, shape depends on the 2D structure.
        target: Target distribution/tensor, matching shape of `pred`.
        reduction (str): Specifies the reduction to apply to the final output. 
                         Default is 'mean'.

    Returns:
        A scalar or tensor containing the KL divergence, depending on the chosen reduction.
    """

    # (1) Compute JS-div or KL-div in 2D
    div_score = _js_div_2d(target, pred)

    # (2) Apply reduction
    return _reduce_loss(div_score, reduction)'''
)