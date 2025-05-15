import pandas as pd
import numpy as np
import torch
from typing import Union, Tuple

class GroupNormalizer:
    def transform(
        self, y: pd.Series, X: pd.DataFrame = None, return_norm: bool = False, target_scale: torch.Tensor = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        
        if X is None:
            raise ValueError("X must be provided to determine groups for normalization.")
        
        # Ensure y is aligned with X
        if len(y) != len(X):
            raise ValueError("Length of y must match the number of rows in X.")
        
        # Identify group columns
        group_columns = X.columns.tolist()
        
        # Initialize containers for results
        scaled_data = np.empty_like(y, dtype=np.float64)
        norm_params = {}
        
        # Group by the specified columns
        grouped = X.groupby(group_columns)
        
        for group_keys, indices in grouped.groups.items():
            group_y = y.iloc[indices]
            
            # Calculate normalization parameters
            if target_scale is not None:
                # Use provided target_scale if available
                mean, std = target_scale[group_keys]
            else:
                mean = group_y.mean()
                std = group_y.std()
            
            # Avoid division by zero
            if std == 0:
                std = 1
            
            # Scale the data
            scaled_data[indices] = (group_y - mean) / std
            
            # Store normalization parameters
            norm_params[group_keys] = (mean, std)
        
        if return_norm:
            return scaled_data, norm_params
        else:
            return scaled_data

