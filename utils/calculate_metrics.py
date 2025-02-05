import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true_list, y_pred_list, metrics):
    y_true_x = [y_true[0] for y_true in y_true_list]
    y_true_y = [y_true[1] for y_true in y_true_list]
    y_true_z = [y_true[2] for y_true in y_true_list]
    y_pred_x = [y_pred[0] for y_pred in y_pred_list]
    y_pred_y = [y_pred[1] for y_pred in y_pred_list]
    y_pred_z = [y_pred[2] for y_pred in y_pred_list]
    y_pred_ref = [0 for _ in y_true_list]

    
  
    # Computation of all the metrics
    metrics['rmse'].append([
                            root_mean_squared_error(y_true=y_true_x, y_pred=y_pred_x),
                            root_mean_squared_error(y_true=y_true_y, y_pred=y_pred_y),
                            root_mean_squared_error(y_true=y_true_z, y_pred=y_pred_z)
                            ])
    metrics['rmse_ref'].append([
                                root_mean_squared_error(y_true=y_true_x, y_pred=y_pred_ref),
                                root_mean_squared_error(y_true=y_true_y, y_pred=y_pred_ref),
                                root_mean_squared_error(y_true=y_true_z, y_pred=y_pred_ref)
                                ])
    metrics['mae'].append([
                        mean_absolute_error(y_true=y_true_x, y_pred=y_pred_x),
                        mean_absolute_error(y_true=y_true_y, y_pred=y_pred_y),
                        mean_absolute_error(y_true=y_true_z, y_pred=y_pred_z)
                        ])
    metrics['mae_ref'].append([
                        mean_absolute_error(y_true=y_true_x, y_pred=y_pred_ref),
                        mean_absolute_error(y_true=y_true_y, y_pred=y_pred_ref),
                        mean_absolute_error(y_true=y_true_z, y_pred=y_pred_ref)
                        ])
    metrics['r2'].append([
                        r2_score(y_true=y_true_x, y_pred=y_pred_x),
                        r2_score(y_true=y_true_y, y_pred=y_pred_y),
                        r2_score(y_true=y_true_z, y_pred=y_pred_z)
                        ])
    metrics['r2_ref'].append([
                        r2_score(y_true=y_true_x, y_pred=y_pred_ref),
                        r2_score(y_true=y_true_y, y_pred=y_pred_ref),
                        r2_score(y_true=y_true_z, y_pred=y_pred_ref)
                        ])
    
    return metrics