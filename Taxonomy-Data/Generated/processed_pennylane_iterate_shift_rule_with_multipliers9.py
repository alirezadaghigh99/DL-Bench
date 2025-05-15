import numpy as np
import itertools

def _iterate_shift_rule_with_multipliers(rule, order, period=None):
    """
    Apply a shift rule with multipliers repeatedly along the same parameter axis.

    Args:
        rule (np.ndarray): A 2D array where each row contains [coefficient, multiplier, shift].
        order (int): The number of times to apply the shift rule.
        period (float, optional): The period for handling periodic boundary conditions.

    Returns:
        np.ndarray: A stacked array of combined rules with columns corresponding to
                    coefficients, multipliers, and cumulative shifts.
    """
    # Extract coefficients, multipliers, and shifts from the rule
    coefficients, multipliers, shifts = rule[:, 0], rule[:, 1], rule[:, 2]

    # Generate all combinations of applying the rule `order` times
    combinations = itertools.product(range(len(rule)), repeat=order)

    combined_rules = []

    for combination in combinations:
        # Initialize cumulative values
        cumulative_coefficient = 1.0
        cumulative_multiplier = 1.0
        cumulative_shift = 0.0

        # Apply each rule in the combination
        for index in combination:
            cumulative_coefficient *= coefficients[index]
            cumulative_multiplier *= multipliers[index]
            cumulative_shift += shifts[index]

        # Handle periodic boundary conditions if a period is specified
        if period is not None:
            cumulative_shift = cumulative_shift % period

        # Append the combined rule
        combined_rules.append([cumulative_coefficient, cumulative_multiplier, cumulative_shift])

    return np.array(combined_rules)

