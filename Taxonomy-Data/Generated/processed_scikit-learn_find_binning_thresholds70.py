import numpy as np

def _find_binning_thresholds(col_data, max_bins):
    # Convert col_data to a numpy array and remove missing values
    col_data = np.array(col_data)
    col_data = col_data[~np.isnan(col_data)]
    
    # Sort the data and find distinct values
    sorted_data = np.sort(col_data)
    distinct_values = np.unique(sorted_data)
    
    n_unique_values = len(distinct_values)
    
    # If the number of distinct values is less than or equal to max_bins
    if n_unique_values <= max_bins:
        # Calculate midpoints between consecutive distinct values
        midpoints = (distinct_values[:-1] + distinct_values[1:]) / 2.0
    else:
        # Calculate approximate midpoints using percentiles
        percentiles = np.linspace(0, 100, min(max_bins, n_unique_values) + 1)
        bin_edges = np.percentile(sorted_data, percentiles)
        midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # Ensure no +inf values are included
    midpoints = midpoints[np.isfinite(midpoints)]
    
    return midpoints

