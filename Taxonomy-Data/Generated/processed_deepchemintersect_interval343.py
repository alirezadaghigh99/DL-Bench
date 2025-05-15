from typing import Tuple

def intersect_interval(interval1: Tuple[float, float], interval2: Tuple[float, float]) -> Tuple[float, float]:
    # Unpack the intervals
    x1_min, x1_max = interval1
    x2_min, x2_max = interval2
    
    # Calculate the intersection
    intersect_min = max(x1_min, x2_min)
    intersect_max = min(x1_max, x2_max)
    
    # Check if the intersection is valid
    if intersect_min <= intersect_max:
        return (intersect_min, intersect_max)
    else:
        return (0, 0)

