import numpy as np

def normalize_points(points):
    """ Normalize a set of points so that the centroid is at the origin and the average distance to the origin is sqrt(2). """
    B, N, _ = points.shape
    mean = np.mean(points, axis=1, keepdims=True)
    std = np.std(points, axis=1, keepdims=True)
    scale = np.sqrt(2) / std
    T = np.zeros((B, 3, 3))
    T[:, 0, 0] = scale[:, 0, 0]
    T[:, 1, 1] = scale[:, 0, 1]
    T[:, 0, 2] = -scale[:, 0, 0] * mean[:, 0, 0]
    T[:, 1, 2] = -scale[:, 0, 1] * mean[:, 0, 1]
    T[:, 2, 2] = 1
    points_h = np.concatenate([points, np.ones((B, N, 1))], axis=-1)
    normalized_points = np.einsum('bij,bkj->bki', T, points_h)
    return normalized_points, T

def construct_matrix(points1, points2):
    """ Construct the matrix A used in the 7-point algorithm. """
    B, N, _ = points1.shape
    A = np.zeros((B, N, 9))
    A[:, :, 0] = points1[:, :, 0] * points2[:, :, 0]
    A[:, :, 1] = points1[:, :, 1] * points2[:, :, 0]
    A[:, :, 2] = points2[:, :, 0]
    A[:, :, 3] = points1[:, :, 0] * points2[:, :, 1]
    A[:, :, 4] = points1[:, :, 1] * points2[:, :, 1]
    A[:, :, 5] = points2[:, :, 1]
    A[:, :, 6] = points1[:, :, 0]
    A[:, :, 7] = points1[:, :, 1]
    A[:, :, 8] = 1
    return A

def solve_fundamental_matrix(A):
    """ Solve for the fundamental matrix using SVD. """
    B = A.shape[0]
    F_matrices = []
    for i in range(B):
        _, _, Vt = np.linalg.svd(A[i])
        F1 = Vt[-1].reshape(3, 3)
        F2 = Vt[-2].reshape(3, 3)
        F_matrices.append((F1, F2))
    return F_matrices

def compute_polynomial(F1, F2):
    """ Compute the coefficients of the cubic polynomial. """
    detF1 = np.linalg.det(F1)
    detF2 = np.linalg.det(F2)
    F1F2 = np.linalg.det(F1 + F2)
    a0 = detF1
    a1 = 3 * (detF2 - detF1)
    a2 = 3 * (F1F2 - 2 * detF2)
    a3 = np.linalg.det(F2)
    return [a3, a2, a1, a0]

def solve_cubic(coeffs):
    """ Solve the cubic polynomial for real roots. """
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    return real_roots

def compute_fundamental_matrices(F1, F2, roots):
    """ Compute the fundamental matrices from the roots. """
    F_matrices = []
    for alpha in roots:
        F = alpha * F1 + (1 - alpha) * F2
        F_matrices.append(F)
    return F_matrices

def run_7point(points1, points2):
    assert points1.shape == points2.shape, "Input point sets must have the same shape."
    B, N, D = points1.shape
    assert N == 7 and D == 2, "Each batch must contain exactly 7 points with 2D coordinates."

    # Normalize points
    norm_points1, T1 = normalize_points(points1)
    norm_points2, T2 = normalize_points(points2)

    # Construct matrix A
    A = construct_matrix(norm_points1, norm_points2)

    # Solve for fundamental matrices
    F_candidates = solve_fundamental_matrix(A)

    # Compute potential fundamental matrices
    all_F_matrices = []
    for i in range(B):
        F1, F2 = F_candidates[i]
        coeffs = compute_polynomial(F1, F2)
        roots = solve_cubic(coeffs)
        F_matrices = compute_fundamental_matrices(F1, F2, roots)
        
        # Denormalize the fundamental matrices
        for F in F_matrices:
            F = np.dot(T2[i].T, np.dot(F, T1[i]))
            all_F_matrices.append(F)

    # Reshape the output
    m = len(all_F_matrices) // B
    all_F_matrices = np.array(all_F_matrices).reshape(B, m, 3, 3)
    return all_F_matrices

