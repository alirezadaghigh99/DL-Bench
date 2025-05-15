import pandas as pd
import numpy as np
from typing import Union, Tuple

class TorchNormalizer:
    # Assuming TorchNormalizer is a base class with some functionality
    pass

class GroupNormalizer(TorchNormalizer):
    def fit_transform(
        self, y: pd.Series, X: pd.DataFrame, return_norm: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # Ensure the group columns are present in X
        if 'group' not in X.columns:
            raise ValueError("X must contain a 'group' column for group-specific normalization.")
        
        # Initialize dictionaries to store normalization parameters
        group_means = {}
        group_stds = {}
        
        # Prepare an array to store the scaled data
        scaled_data = np.empty_like(y, dtype=float)
        
        # Iterate over each group
        for group, group_data in X.groupby('group'):
            # Extract the corresponding y values for the current group
            y_group = y[group_data.index]
            
            # Calculate mean and std for the current group
            mean = y_group.mean()
            std = y_group.std()
            
            # Store the parameters
            group_means[group] = mean
            group_stds[group] = std
            
            # Scale the data for the current group
            if std != 0:
                scaled_data[group_data.index] = (y_group - mean) / std
            else:
                # Handle the case where std is zero to avoid division by zero
                scaled_data[group_data.index] = y_group - mean
        
        # If return_norm is True, return the scaled data and normalization parameters
        if return_norm:
            normalization_params = (group_means, group_stds)
            return scaled_data, normalization_params
        
        # Otherwise, just return the scaled data
        return scaled_data

