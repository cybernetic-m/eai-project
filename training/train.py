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
from calculate_metrics import calculate_metrics # type: ignore

import numpy as np
import torch

def train(num_epochs, loss_fn, model, optimizer, scheduler, training_dataloader, validation_dataloader, hyperparams, model_dict, complete): 

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
        loss_avg, y_true_list, y_pred_list = train_one_epoch(model, optimizer, loss_fn, training_dataloader, complete, train=True)
                
        train_metrics = calculate_metrics(y_true_list, y_pred_list, train_metrics)

        # Validation 
        with torch.no_grad():
            vloss_avg, vy_true_list, vy_pred_list = train_one_epoch(model, optimizer, loss_fn, validation_dataloader, complete, train=False)
        
        valid_metrics = calculate_metrics(vy_true_list, vy_pred_list, valid_metrics)

        if vloss_avg < best_vloss:
            best_vloss = vloss_avg
            path = model.save()
            
        #scheduler.step() # Update the learning rate as lr^gamma (exponential decay)
        
        print("train: LOSS %.6f MAE X1:%.4f, Y1:%.4f, Z1:%.4f R2 X1:%.4f, Y1:%.4f, Z1:%.4f RMSE X1:%.4f, Y1:%.4f, Z1:%.4f"
              % (loss_avg, train_metrics['mae'][epoch][0], train_metrics['mae'][epoch][1], train_metrics['mae'][epoch][2]
                 ,train_metrics['r2'][epoch][0],train_metrics['r2'][epoch][1],train_metrics['r2'][epoch][2]
                 ,train_metrics['rmse'][epoch][0],train_metrics['rmse'][epoch][1],train_metrics['rmse'][epoch][2]))
        print("valid: LOSS %.6f MAE X1:%.4f, Y1:%.4f, Z1:%.4f R2 X1:%.4f, Y1:%.4f, Z1:%.4f RMSE X1:%.4f, Y1:%.4f, Z1:%.4f" 
              % (vloss_avg, valid_metrics['mae'][epoch][0], valid_metrics['mae'][epoch][1], valid_metrics['mae'][epoch][2]
                 ,valid_metrics['r2'][epoch][0],valid_metrics['r2'][epoch][1],valid_metrics['r2'][epoch][2]
                 ,valid_metrics['rmse'][epoch][0],valid_metrics['rmse'][epoch][1],valid_metrics['rmse'][epoch][2]))
    
    # Save in JSON files the hyperparameter list, the structure of the ensemble model (if you are using it), the metrics of train and validation
    with open(path+'/hyperparam.json', 'w') as f:
        json.dump(hyperparams, f)
    if complete==True:
        with open(path+'/ensemble.json', 'w') as f:
            json.dump(model_dict, f)
    with open(path+'/train_metrics.json', 'w') as f:
        json.dump(train_metrics, f)
    with open(path+'/valid_metrics.json', 'w') as f:
        json.dump(valid_metrics, f)
