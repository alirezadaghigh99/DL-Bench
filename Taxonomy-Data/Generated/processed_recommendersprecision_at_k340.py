import pandas as pd

def precision_at_k(rating_true, rating_pred, col_user, col_item, col_prediction, relevancy_method='threshold', k=10, threshold=3.5):
    """
    Calculate precision at K for a recommendation system.
    
    Parameters:
    - rating_true: pandas DataFrame, true ratings with columns [user, item, rating]
    - rating_pred: pandas DataFrame, predicted ratings with columns [user, item, prediction]
    - col_user: str, column name for user
    - col_item: str, column name for item
    - col_prediction: str, column name for prediction
    - relevancy_method: str, method for determining relevancy ('threshold' or other methods)
    - k: int, number of top K items per user
    - threshold: float, threshold for determining relevancy
    
    Returns:
    - float, precision at K
    """
    
    # Ensure the dataframes are sorted by user and prediction
    rating_pred = rating_pred.sort_values(by=[col_user, col_prediction], ascending=[True, False])
    
    # Get top K predictions for each user
    top_k_pred = rating_pred.groupby(col_user).head(k)
    
    # Determine relevancy in the true ratings
    if relevancy_method == 'threshold':
        rating_true['relevant'] = rating_true[col_prediction] >= threshold
    else:
        raise ValueError("Unsupported relevancy method")
    
    # Merge the top K predictions with the true ratings
    merged = pd.merge(top_k_pred, rating_true, on=[col_user, col_item], how='left', suffixes=('_pred', '_true'))
    
    # Fill NaN values in 'relevant' with False (items not in true ratings are not relevant)
    merged['relevant'] = merged['relevant'].fillna(False)
    
    # Calculate precision at K
    precision_sum = merged.groupby(col_user)['relevant'].sum()
    precision_count = top_k_pred.groupby(col_user).size()
    
    # Handle cases where the number of items for a user in the predicted ratings is less than K
    precision_count = precision_count.clip(upper=k)
    
    precision_at_k = (precision_sum / precision_count).mean()
    
    return precision_at_k

