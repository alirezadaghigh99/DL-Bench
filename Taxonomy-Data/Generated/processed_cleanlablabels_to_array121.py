import numpy as np
import pandas as pd
from typing import Union

LabelLike = Union[list, np.ndarray, pd.Series, pd.DataFrame]

def labels_to_array(y: Union[LabelLike, np.generic]) -> np.ndarray:
    # Check if the input is a pandas DataFrame
    if isinstance(y, pd.DataFrame):
        # Raise an error if the DataFrame has more than one column
        if y.shape[1] != 1:
            raise ValueError("DataFrame input must have exactly one column.")
        # Convert the single column DataFrame to a Series
        y = y.iloc[:, 0]
    
    # Convert the input to a NumPy array
    y_array = np.asarray(y)
    
    # Check if the resulting array is 1D
    if y_array.ndim != 1:
        raise ValueError("Input cannot be converted to a 1D NumPy array.")
    
    return y_array

