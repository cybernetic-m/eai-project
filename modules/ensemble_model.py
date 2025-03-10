import sys
import os
arima_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ARIMA'))
sys.path.append(arima_path)

import torch
import torch.nn as nn
from TimeSeries import ARIMA # type: ignore
from blocks import *
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

class ensemble_model(nn.Module):

    def __init__(self, model_dict, device, mode, heterogeneous = False, gamma=0.9):
        super(ensemble_model,self).__init__()
        # model_dict is something like -> {'block_1': {'param_1': [10,20,30,3]}, 'block_2': {'param_1': [10,10], 'param_2': [20, 30]}, ...}

        self.mode = mode # Modality of voting
        self.models = nn.ModuleList() # List of models
        self.arima_models = nn.ModuleList()
        self.rnn_models = nn.ModuleList()
        for model_name, model_param_list in model_dict.items():
            if model_name == 'ARIMA':
                for model_param in model_param_list:
                    arima_block = ARIMA(**model_param, device=device)
                    self.arima_models.append(arima_block)
            elif model_name == 'linear_regressor':
                for model_param in model_param_list:
                    linear_block = linear(**model_param).to(device)
                    self.models.append(linear_block)
            elif model_name == 'mlp':
                for model_param in model_param_list:
                    mlp_block = mlp(**model_param).to(device)
                    self.models.append(mlp_block)
            elif model_name == 'rnn':
                for model_param in model_param_list:
                    rnn_block = rnn_regressor(**model_param).to(device)
                    self.rnn_models.append(rnn_block)
            elif model_name == 'lstm':
                for model_param in model_param_list:
                    lstm_block = lstm_regressor(**model_param).to(device)
                    self.rnn_models.append(lstm_block)
        
        self.n_models = len(self.models)+len(self.arima_models)+len(self.rnn_models) # Number of models in the ensemble

        # Boolean that permits to send to the model an heterogeneous vector of features [temperature_features (extracted from autoencoder), gradient]
        self.heterogeneous = heterogeneous 

        # Weights initialization for auto weighted average system
        self.weights = torch.ones(self.n_models).to(device) / self.n_models # In this way we have a tensor of uniform weighting [1/3, 1/3, 1/3] for example

        self.gamma = gamma
        
        print("Ensemble Model Summary:", self.models, self.arima_models, self.rnn_models)
        print("Mode of voting", self.mode)
        
    def get_models(self):
        return self.models, self.rnn_models, self.arima_models

    def voting(self, prediction_list):
        #print(prediction_list)
        #print("Mode:", self.mode)

        #print("Predictions model 0:", prediction_list[0][0].detach().cpu().numpy())
        #print("Predictions model 1:", prediction_list[1][0].detach().cpu().numpy())
        #print("Predictions model 2:", prediction_list[2][0].detach().cpu().numpy())
        prediction_tensor = torch.stack(prediction_list) # Transform a list of tensor in a tensor [torch.tensor[8,1,3], torch.tensor[8,1,3], torch.tensor[8,1,3]] -> torch.tensor[3, 8, 1, 3]
        #print(prediction_tensor.shape)
        if self.mode == 'average':
            y = torch.mean(prediction_tensor, dim=0)  # Do the average on dim=0 because it's the model dimension [3, 8 , 1, 3] where the 3 is the number of models
            #print("Average y:", y[0].detach().cpu().numpy())
        if self.mode == 'median':
            y = torch.median(prediction_tensor, dim=0).values  # Do the average on dim=0 because it's the model dimension [3, 8 , 1, 3] where the 3 is the number of models
            #print("Median y:", y[0].detach().cpu().numpy())
        if self.mode == 'auto-weighted':
            weights = self.weights.reshape(self.n_models,1,1,1) # Broadcasting of weights adding 3 dimension for multiplication to [3, 8, 1, 3]
            y = (weights*prediction_tensor).sum(dim=0) # Do the weighted average (summing in dim=0 -> number of models)
            #print("AW y:", y[0].detach().cpu().numpy())
        return y
    
    def update_weights(self, y_pred, y_true):
        model_losses = []
        with torch.no_grad():
            for n in range(self.n_models):
                
                #print(y_pred[n].shape)
                #print(y_true.shape)
                loss_n = root_mean_squared_error(y_pred[n].detach().cpu().squeeze(1), y_true[:,1,:].detach().cpu()) # Dimension [8, 1, 3]
                #print(n, ": ", loss_n)
                #print(loss_n.shape)
                # Do the average among 8 samples in the batch (dim=0)
                #loss_n_avg_batch = torch.mean(loss_n, dim=0)
                # Do the average among X1, Y1, Z1 losses (they are 3 losses, one for each axis gradients)
                #loss_n_avg_batch_gradients = torch.mean(loss_n_avg_batch, dim=1)
                #model_losses.append(loss_n_avg_batch_gradients) # Append the average loss among the batch
                model_losses.append(loss_n)
                #print(model_losses[n].shape)
                #print(self.weights.shape) 
                weight = 1 / (model_losses[n] + 1e-6)
                self.weights[n] = self.gamma * weight + (1-self.gamma) * self.weights[n]
                #print(self.weights)
            # Do the softmax of the tensor weights ([3]) because we want to normalize all the weights
            # such that their sum to one (dim=0 because the tensor shape is simply [3])
            self.weights = F.softmax(self.weights/max(self.weights), dim=0) 

    def forward(self, x, y_true_norm, y_true):
        y_pred = []
        if self.heterogeneous:
            #print(y_true_norm.shape)
            #print(x.shape)
            if x.shape[1] != 1:
                x = x.reshape(x.shape[0], 1, -1)
            x = torch.concat((x, y_true_norm), dim=2)
            
        if self.models:
            y_pred += [model(x) for model in self.models] # Create a list of tensor of predictions [[8,1,3], [8,1,3], ...]
        #print(y_true[:,0,:].shape)
        if self.arima_models:
            y_pred += [arima(y_true[:,0,:].unsqueeze(1)) for arima in self.arima_models] # [8,1,3]
        if self.rnn_models:
            #print(x.shape)
            y_pred += [model(x) for model in self.rnn_models]
        if self.mode == 'auto-weighted':
            self.update_weights(y_pred, y_true) # Update the weights for the autoweighted voting
        y = self.voting(y_pred) # Apply the voting among the predictions y = [8,1,3]
        return y, y_pred
    
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



            
