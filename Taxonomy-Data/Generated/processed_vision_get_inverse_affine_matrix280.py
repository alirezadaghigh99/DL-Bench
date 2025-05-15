import math
from typing import List

def _get_inverse_affine_matrix(center: List[float], angle: float, translate: List[float], 
                               scale: float, shear: List[float], inverted: bool) -> List[float]:
    # Validate inputs
    if not (isinstance(center, list) and len(center) == 2 and all(isinstance(c, (int, float)) for c in center)):
        raise ValueError("Center must be a list of two float values.")
    if not isinstance(angle, (int, float)):
        raise ValueError("Angle must be a float value.")
    if not (isinstance(translate, list) and len(translate) == 2 and all(isinstance(t, (int, float)) for t in translate)):
        raise ValueError("Translate must be a list of two float values.")
    if not isinstance(scale, (int, float)) or scale == 0:
        raise ValueError("Scale must be a non-zero float value.")
    if not (isinstance(shear, list) and len(shear) == 2 and all(isinstance(s, (int, float)) for s in shear)):
        raise ValueError("Shear must be a list of two float values.")
    if not isinstance(inverted, bool):
        raise ValueError("Inverted must be a boolean value.")

    # Convert angle from degrees to radians
    angle_rad = math.radians(angle)
    sx_rad = math.radians(shear[0])
    sy_rad = math.radians(shear[1])

    # Calculate rotation matrix components
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Calculate shear matrix components
    tan_sx = math.tan(sx_rad)
    tan_sy = math.tan(sy_rad)

    # Calculate the affine transformation matrix
    a = scale * (cos_a - tan_sy * sin_a)
    b = scale * (sin_a + tan_sy * cos_a)
    c = scale * (tan_sx * cos_a - sin_a)
    d = scale * (tan_sx * sin_a + cos_a)

    # Calculate the translation components
    tx = translate[0] + center[0] - a * center[0] - c * center[1]
    ty = translate[1] + center[1] - b * center[0] - d * center[1]

    # Affine transformation matrix
    matrix = [a, b, tx, c, d, ty]

    if inverted:
        # Calculate the determinant
        det = a * d - b * c
        if det == 0:
            raise ValueError("The affine transformation matrix is not invertible.")

        # Calculate the inverse matrix
        inv_a = d / det
        inv_b = -b / det
        inv_c = -c / det
        inv_d = a / det
        inv_tx = (b * ty - d * tx) / det
        inv_ty = (c * tx - a * ty) / det

        # Inverse affine transformation matrix
        matrix = [inv_a, inv_b, inv_tx, inv_c, inv_d, inv_ty]

    return matrix

