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