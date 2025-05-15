import numpy as np
import cv2

def motion_from_essential_choose_solution(E_mat, K1, K2, x1, x2, mask=None):
    # Validate input shapes
    if E_mat.shape != (3, 3):
        raise ValueError("E_mat must be a 3x3 matrix.")
    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise ValueError("K1 and K2 must be 3x3 matrices.")
    if x1.shape[1] != 2 or x2.shape[1] != 2:
        raise ValueError("x1 and x2 must have shape (N, 2).")
    if x1.shape[0] != x2.shape[0]:
        raise ValueError("x1 and x2 must have the same number of points.")
    
    # Ensure points are in homogeneous coordinates
    x1_h = np.hstack((x1, np.ones((x1.shape[0], 1))))
    x2_h = np.hstack((x2, np.ones((x2.shape[0], 1))))
    
    # Compute the possible rotations and translations from the essential matrix
    R1, R2, t = cv2.decomposeEssentialMat(E_mat)
    
    # Four possible solutions
    solutions = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]
    
    # Prepare the camera matrices
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    best_solution = None
    max_positive_depth = -1
    best_3d_points = None
    
    for R, t in solutions:
        P2 = K2 @ np.hstack((R, t))
        
        # Triangulate points
        points_4d_hom = cv2.triangulatePoints(P1, P2, x1_h.T, x2_h.T)
        points_3d = points_4d_hom[:3] / points_4d_hom[3]
        
        # Check the number of points with positive depth
        if mask is not None:
            valid_mask = mask.flatten().astype(bool)
            points_3d = points_3d[:, valid_mask]
        
        num_positive_depth = np.sum(points_3d[2, :] > 0)
        
        if num_positive_depth > max_positive_depth:
            max_positive_depth = num_positive_depth
            best_solution = (R, t)
            best_3d_points = points_3d.T
    
    if best_solution is None:
        raise RuntimeError("No valid solution found.")
    
    R_best, t_best = best_solution
    return R_best, t_best, best_3d_points

