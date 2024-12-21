import os
import sys
# Get the absolute paths of the directories containing the utils functions and train_one_epoch
training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../training'))
# Add these directories to sys.path
sys.path.append(training_path)

from train_one_epoch import train_one_epoch

import numpy as np
import torch

def train(num_epochs, loss_fn, model, optimizer, training_dataloader, validation_dataloader): 

    best_vloss = 1000000000


    for epoch in range(num_epochs):
        print(f'EPOCH {epoch + 1}/{num_epochs}:')

        # Compute the loss_avg, y_pred and y_true
        loss_avg, y_true_list, y_pred_list = train_one_epoch(model, optimizer, loss_fn, training_dataloader, train=True)

        # Calculate the Root Mean Square Error metrics
        train_se = 0 # Initialize the Mean Square Error
 
        train_list_len = len(y_pred_list)
        n_ts = train_list_len * len(y_pred_list[0])

        for i in range(train_list_len):
            y_pred = np.array(y_pred_list[i])
            y_true = np.array(y_true_list[i])
            train_se += np.sum((y_pred-y_true)**2)
        train_rmse = np.sqrt(train_se / n_ts)

        # Validation 
        with torch.no_grad():
            vloss_avg, vy_true_list, vy_pred_list = train_one_epoch(model, optimizer, loss_fn, validation_dataloader, train=False)
        
        # Calculate the Root Mean Square Error metrics
        valid_se = 0 # Initialize the Mean Square Error

        valid_list_len = len(vy_pred_list)
        n_vs = valid_list_len * len(vy_pred_list[0])

        for i in range(valid_list_len):
            vy_pred = np.array(vy_pred_list[i])
            vy_true = np.array(vy_true_list[i])
            valid_se += np.sum((vy_pred-vy_true)**2)
        valid_rmse = np.sqrt(valid_se / n_vs)

        if vloss_avg < best_vloss:
            best_vloss = vloss_avg
            model.save(epoch)
        
        print("train LOSS %.4f valid LOSS %.4f train RMSE %.4f valid RMSE %.4f" % (loss_avg, vloss_avg, train_rmse, valid_rmse))