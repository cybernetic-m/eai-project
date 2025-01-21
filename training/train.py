import os
import sys
import json
# Get the absolute paths of the directories containing the utils functions and train_one_epoch
training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../training'))
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
# Add these directories to sys.path
sys.path.append(training_path)
sys.path.append(utils_path)

from train_one_epoch import train_one_epoch
from calculate_metrics import calculate_metrics

import numpy as np
import torch

def train(num_epochs, loss_fn, model, optimizer, training_dataloader, validation_dataloader, hyperparams): 

    best_vloss = 1000000000

    # Definition of the dictionary of metrics
    train_metrics = {
        'rmse':[], 
        'mae':[],
        'r2':[]       
               }
    valid_metrics = {
        'rmse':[], 
        'mae':[],
        'r2':[]       
               }

    for epoch in range(num_epochs):
        print(f'EPOCH {epoch + 1}/{num_epochs}:')

        # Compute the loss_avg, y_pred and y_true
        loss_avg, y_true_list, y_pred_list = train_one_epoch(model, optimizer, loss_fn, training_dataloader, train=True)
                
        train_metrics = calculate_metrics(y_true_list, y_pred_list, train_metrics)

        # Validation 
        with torch.no_grad():
            vloss_avg, vy_true_list, vy_pred_list = train_one_epoch(model, optimizer, loss_fn, validation_dataloader, train=False)
        
        valid_metrics = calculate_metrics(vy_true_list, vy_pred_list, valid_metrics)

        if vloss_avg < best_vloss:
            best_vloss = vloss_avg
            path = model.save(epoch)
            with open(path+'/hyperparam.json', 'w') as f:
                json.dump(hyperparams, f)
        print("train: LOSS %.4f MAE %.4f R2 %.4f RMSE %.4f --- valid: LOSS %.4f MAE %.4f R2 %.4f RMSE %.4f" % (loss_avg, train_metrics['mae'][epoch], train_metrics['r2'][epoch], train_metrics['rmse'][epoch], vloss_avg, valid_metrics['mae'][epoch], valid_metrics['r2'][epoch], valid_metrics['rmse'][epoch]))
    
    with open(path+'/train_metrics.json', 'w') as f:
        json.dump(train_metrics, f)
