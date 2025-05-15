import pandas as pd

def merge_ranking_true_pred(rating_true, rating_pred, col_user, col_item, col_prediction, k):
    # Merge true and predicted ratings on user and item
    merged = pd.merge(rating_true, rating_pred, on=[col_user, col_item], how='left')
    
    # Sort predictions for each user
    merged = merged.sort_values(by=[col_user, col_prediction], ascending=[True, False])
    
    # Group by user and take the top K predictions
    top_k_pred = merged.groupby(col_user).head(k)
    
    return top_k_pred

def recall_at_k(rating_true, rating_pred, col_user, col_item, col_prediction, relevancy_method, k, threshold):
    # Determine relevancy based on the method
    if relevancy_method == 'threshold':
        rating_true['relevant'] = rating_true[col_prediction] >= threshold
    else:
        raise ValueError("Unsupported relevancy method")
    
    # Filter only relevant items
    relevant_items = rating_true[rating_true['relevant']]
    
    # Get top K predicted items
    top_k_pred = merge_ranking_true_pred(rating_true, rating_pred, col_user, col_item, col_prediction, k)
    
    # Calculate recall at K
    recall_sum = 0.0
    user_count = 0
    
    for user, group in relevant_items.groupby(col_user):
        true_items = set(group[col_item])
        pred_items = set(top_k_pred[top_k_pred[col_user] == user][col_item])
        
        if true_items:
            hits = len(true_items & pred_items)
            recall_sum += hits / len(true_items)
            user_count += 1
    
    # If no users have relevant items, return 0.0
    if user_count == 0:
        return 0.0
    
    return recall_sum / user_count

