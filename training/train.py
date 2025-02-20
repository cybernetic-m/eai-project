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

def train(num_epochs, loss_fn, model, optimizer, scheduler, training_dataloader, validation_dataloader, hyperparams, model_dict, autoencoder_dict, complete): 

    best_vloss = 1000000000

    # Definition of the dictionary of metrics
    train_model_metrics = {
        'rmse':[], 
        'mae':[],
        'r2':[],
        'loss':[]       
               }
    train_autoencoder_metrics = {
        'rmse':[], 
        'mae':[],
        'r2':[],
        'loss':[]       
               }
    valid_model_metrics = {
        'rmse':[], 
        'mae':[],
        'r2':[],
        'loss':[]     
               }
    valid_autoencoder_metrics = {
        'rmse':[], 
        'mae':[],
        'r2':[],
        'loss':[]       
               }
    
    if autoencoder_dict != None:
        autoencoder_ = True

    for epoch in range(num_epochs):
        print(f'EPOCH {epoch + 1}/{num_epochs}:')

        # Compute the loss_avg, y_pred and y_true
        loss_avg, loss_avg_autoencoder, y_true_list, y_pred_list, x_true_list, x_pred_list = train_one_epoch(model, optimizer, loss_fn, training_dataloader, complete, autoencoder_, train=True)
                
        train_model_metrics = calculate_metrics(y_true_list, y_pred_list, train_model_metrics)
        if autoencoder_:
            train_autoencoder_metrics = calculate_metrics(x_true_list, x_pred_list, train_autoencoder_metrics)

        train_model_metrics['loss'].append(loss_avg)
        if autoencoder_:
            train_autoencoder_metrics['loss'].append(loss_avg_autoencoder)

        # Validation 
        with torch.no_grad():
            vloss_avg, vloss_avg_autoencoder, vy_true_list, vy_pred_list, vx_true_list, vx_pred_list = train_one_epoch(model, optimizer, loss_fn, validation_dataloader, complete, autoencoder_, train=False)
        
        valid_model_metrics = calculate_metrics(vy_true_list, vy_pred_list, valid_model_metrics)
        if autoencoder_:
            valid_autoencoder_metrics = calculate_metrics(vx_true_list, vx_pred_list, valid_autoencoder_metrics)

        valid_model_metrics['loss'].append(vloss_avg)
        if autoencoder_:
            valid_autoencoder_metrics['loss'].append(vloss_avg_autoencoder)

        if vloss_avg < best_vloss:
            best_vloss = vloss_avg
            path = model.save()
        
        if complete:
            for schedul in scheduler:
                schedul.step() # Update the learning rate as lr^gamma (exponential decay)
            print(model.ensemble.weights)
            
        else:
            scheduler.step() # Update the learning rate as lr^gamma (exponential decay)
        
        print("train MODEL: LOSS %.12f MAE X1:%.4f, Y1:%.4f, Z1:%.4f R2 X1:%.4f, Y1:%.4f, Z1:%.4f RMSE X1:%.6f, Y1:%.6f, Z1:%.6f"
              % (loss_avg, train_model_metrics['mae'][epoch][0], train_model_metrics['mae'][epoch][1], train_model_metrics['mae'][epoch][2]
                 ,train_model_metrics['r2'][epoch][0],train_model_metrics['r2'][epoch][1],train_model_metrics['r2'][epoch][2]
                 ,train_model_metrics['rmse'][epoch][0],train_model_metrics['rmse'][epoch][1],train_model_metrics['rmse'][epoch][2]))
        if autoencoder_:
            print("train AUTOENCODER: LOSS %.12f MAE X1:%.4f, Y1:%.4f, Z1:%.4f R2 X1:%.4f, Y1:%.4f, Z1:%.4f RMSE X1:%.6f, Y1:%.6f, Z1:%.6f"
                % (loss_avg_autoencoder, train_autoencoder_metrics['mae'][epoch][0], train_autoencoder_metrics['mae'][epoch][1], train_autoencoder_metrics['mae'][epoch][2]
                    ,train_autoencoder_metrics['r2'][epoch][0],train_autoencoder_metrics['r2'][epoch][1],train_autoencoder_metrics['r2'][epoch][2]
                    ,train_autoencoder_metrics['rmse'][epoch][0],train_autoencoder_metrics['rmse'][epoch][1],train_autoencoder_metrics['rmse'][epoch][2]))
        print("valid MODEL: LOSS %.12f MAE X1:%.4f, Y1:%.4f, Z1:%.4f R2 X1:%.4f, Y1:%.4f, Z1:%.4f RMSE X1:%.6f, Y1:%.6f, Z1:%.6f" 
              % (vloss_avg, valid_model_metrics['mae'][epoch][0], valid_model_metrics['mae'][epoch][1], valid_model_metrics['mae'][epoch][2]
                 ,valid_model_metrics['r2'][epoch][0],valid_model_metrics['r2'][epoch][1],valid_model_metrics['r2'][epoch][2]
                 ,valid_model_metrics['rmse'][epoch][0],valid_model_metrics['rmse'][epoch][1],valid_model_metrics['rmse'][epoch][2]))
        if autoencoder_:
            print("valid AUTOENCODER: LOSS %.12f MAE X1:%.4f, Y1:%.4f, Z1:%.4f R2 X1:%.4f, Y1:%.4f, Z1:%.4f RMSE X1:%.6f, Y1:%.6f, Z1:%.6f" 
                % (vloss_avg_autoencoder, valid_autoencoder_metrics['mae'][epoch][0], valid_autoencoder_metrics['mae'][epoch][1], valid_autoencoder_metrics['mae'][epoch][2]
                    ,valid_autoencoder_metrics['r2'][epoch][0],valid_autoencoder_metrics['r2'][epoch][1],valid_autoencoder_metrics['r2'][epoch][2]
                    ,valid_autoencoder_metrics['rmse'][epoch][0],valid_autoencoder_metrics['rmse'][epoch][1],valid_autoencoder_metrics['rmse'][epoch][2]))
        
    # Save in JSON files the hyperparameter list, the structure of the ensemble model (if you are using it), the metrics of train and validation
    with open(path+'/hyperparam.json', 'w') as f:
        json.dump(hyperparams, f)
    if complete==True:
        with open(path+'/ensemble.json', 'w') as f:
            json.dump(model_dict, f)
    if autoencoder_:
        with open(path+'/autoencoder.json', 'w') as f:
            json.dump(autoencoder_dict, f)
    with open(path+'/train_model_metrics.json', 'w') as f:
        json.dump(train_model_metrics, f)
    with open(path+'/valid_model_metrics.json', 'w') as f:
        json.dump(valid_model_metrics, f)
    if autoencoder_:
        with open(path+'/train_autoencoder_metrics.json', 'w') as f:
            json.dump(train_autoencoder_metrics, f)
        with open(path+'/valid_autoencoder_metrics.json', 'w') as f:
            json.dump(valid_autoencoder_metrics, f)


