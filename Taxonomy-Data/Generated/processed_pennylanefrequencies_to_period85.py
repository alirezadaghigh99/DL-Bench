import math
from functools import reduce
from fractions import Fraction

def gcd(a, b):
    """Compute the greatest common divisor of a and b."""
    while b:
        a, b = b, a % b
    return a

def gcd_multiple(numbers):
    """Compute the gcd of a list of numbers."""
    return reduce(gcd, numbers)

def frequencies_to_period(frequencies, decimals=0):
    """
    Calculate the period of a Fourier series based on a set of frequencies.
    
    Parameters:
    frequencies (tuple): A tuple of frequencies.
    decimals (int, optional): The number of decimal places to round to. Default is 0.
    
    Returns:
    float: The period of the Fourier series.
    
    Example:
    frequencies = (0.5, 1.0)
    frequencies_to_period(frequencies)
    # Expected output: 12.566370614359172
    """
    # Round frequencies to the specified number of decimal places
    rounded_frequencies = [round(f, decimals) for f in frequencies]
    
    # Convert frequencies to integers by finding a common denominator
    fractions = [Fraction(f).limit_denominator() for f in rounded_frequencies]
    denominators = [frac.denominator for frac in fractions]
    common_denominator = reduce(lambda x, y: x * y // gcd(x, y), denominators)
    
    # Convert all frequencies to a common integer representation
    integer_frequencies = [int(f * common_denominator) for f in rounded_frequencies]
    
    # Calculate the gcd of the integer frequencies
    gcd_value = gcd_multiple(integer_frequencies)
    
    # Calculate the period
    period = 2 * math.pi * common_denominator / gcd_value
    
    return period

