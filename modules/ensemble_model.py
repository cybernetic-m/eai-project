import sys
import os
arima_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ARIMA'))
sys.path.append(arima_path)

import torch
import torch.nn as nn
from TimeSeries import ARIMA # type: ignore
from blocks import *


class ensemble_model(nn.Module):

    def __init__(self, model_dict, device, mode):
        super(ensemble_model,self).__init__()
        # model_dict is something like -> {'block_1': {'param_1': [10,20,30,3]}, 'block_2': {'param_1': [10,10], 'param_2': [20, 30]}, ...}

        self.models = nn.ModuleList() # List of models
        self.mode = mode # Modality of voting
    
        for model_name, model_param in model_dict.items():
            if model_name == 'ARIMA':
                arima_block = ARIMA(**model_param, device=device)
                self.models.append(arima_block)
            elif model_name == 'linear_regressor':
                linear_block = linear(**model_param).to(device)
                self.models.append(linear_block)
            elif model_name == 'mlp':
                mlp_block = mlp(**model_param).to(device)
                self.models.append(mlp_block)
        
        self.n_models = len(self.models) # Number of models in the ensemble

        # Learnable Voting layers
        # (3*len(models) because in our problem each model give three predictions gradients (X1, Y1, Z1)
        self.voting_mlp = mlp([3*self.n_models, 4*self.n_models, 3]) # MLP layer 
        self.voting_linear = nn.Linear(in_features=3*self.n_models, out_features=3) # Linear layer 

        # Weights initialization for auto weighted average system
        self.weights = torch.ones(self.n_models).to(device) / self.n_models # In this way we have a tensor of uniform weighting [1/3, 1/3, 1/3] for example

        print("Ensemble Model Summary:", self.models)

    def voting(self, prediction_list):
        #print(prediction_list)
        prediction_tensor = torch.stack(prediction_list) # Transform a list of tensor in a tensor [torch.tensor[8,1,3], torch.tensor[8,1,3], torch.tensor[8,1,3]] -> torch.tensor[3, 8, 1, 3]
        #print(prediction_tensor.shape)
        if self.mode == 'average':
            y = torch.mean(prediction_tensor, dim=0) # Do the average on dim=0 because it's the model dimension [3, 8 , 1, 3] where the 3 is the number of models
        if self.mode == 'mlp':
            prediction_tensor = prediction_tensor.view(prediction_tensor.shape[1], prediction_tensor.shape[2], -1) # Transform torch.tensor[3, 8, 1, 3] -> [8, 1, 9]
            y = self.voting_mlp(prediction_tensor)
        if self.mode == 'linear':
            prediction_tensor = prediction_tensor.view(prediction_tensor.shape[1], prediction_tensor.shape[2], -1) # Transform torch.tensor[3, 8, 1, 3] -> [8, 1, 9]
            y = self.voting_linear(prediction_tensor)
        if self.mode == 'auto-weighted':
            weights = self.weights.view(3,1,1,1) # Broadcasting of weights adding 3 dimension for multiplication to [3, 8, 1, 3]
            y = (weights*prediction_tensor).sum(dim=0) # Do the weighted average (summing in dim=0 -> number of models)
        return y
    
    def update_weights(self, y_pred, y_true):
        model_losses = []
        with torch.no_grad():
            for n in range(self.n_models):
                
                print(y_pred[n].shape)
                print(y_true.shape)
                loss_n = (y_pred[n] - y_true)**2 # Dimension [8, 1, 3]
                print(loss_n.shape)
                # Do the average among 8 samples in the batch (dim=0)
                loss_n_avg_batch = torch.mean(loss_n, dim=0)
                # Do the average among X1, Y1, Z1 losses (they are 3 losses, one for each axis gradients)
                loss_n_avg_batch_gradients = torch.mean(loss_n_avg_batch, dim=1)
                model_losses.append(loss_n_avg_batch_gradients) # Append the average loss among the batch
                #print(model_losses[n].shape)
                #print(self.weights.shape)
                self.weights[n] = 1 / model_losses[n]
                #print(self.weights)
                # Do the softmax of the tensor weights ([3]) because we want to normalize all the weights
                # such that their sum to one (dim=0 because the tensor shape is simply [3])
                self.weights = torch.softmax(self.weights, dim=0) 

    def forward(self, x, y_true):
     y_pred = [model(x) for model in self.models] # Create a list of tensor of predictions [[8,1,3], [8,1,3], ...]
     #print(y_pred[0].shape)
     if self.mode == 'auto-weighted':
        self.update_weights(y_pred, y_true) # Update the weights for the autoweighted voting
     y = self.voting(y_pred) # Apply the voting among the predictions y = [8,1,3]
     return y
 
'''
if __name__ == '__main__' :

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model_dict = {'mlp': {'layer_dim_list': [3,40,50,3]},  
                'ARIMA': {'p': 2, 'd': 0, 'q': 2, 'ps': 1, 'ds': 1, 'qs': 1, 's': 1},
                'linear_regressor': {'in_features': 3, 'out_features': 3}
                }
    model = ensemble_model(model_dict, device, mode='auto-weighted')
    arima = ARIMA(p= 2, d= 0, q= 2, ps= 1, ds= 1, qs= 1, s= 1).to(device)
    t = torch.rand(8, 1, 3).to(device) # Remember to pass from 4 values of temperature to 3 values of features and from 5 -> 1
    y_true = torch.rand(8, 1, 3).to(device)
    out = arima(t)
    print(out.shape)
'''



            
