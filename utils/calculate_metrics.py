import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true_list, y_pred_list):

    # Definition of the dictionary of metrics
    metrics = {
        'rmse':[], 
        'mae':[],
        'r2':[]       
               }

    # Computation of all the metrics
    metrics['rmse'].append(root_mean_squared_error(y_true=y_true_list, y_pred=y_pred_list))
    metrics['mae'].append(mean_absolute_error(y_true=y_true_list, y_pred=y_pred_list))
    metrics['r2'].append(r2_score(y_true=y_true_list, y_pred=y_pred_list))
    
    return metrics