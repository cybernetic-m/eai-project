import sys
import os
arima_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ARIMA'))
sys.path.append(arima_path)

import torch
import torch.nn as nn
from TimeSeries import ARIMA
from blocks import *


class ensemble_model(nn.Module):

    def __init__(self, model_dict):
        super(ensemble_model)
        # model_dict is something like -> {'block_1': {'param_1': [10,20,30,3]}, 'block_2': {'param_1': [10,10], 'param_2': [20, 30]}, ...}

        self.models = [] # List of models
    
        for model_name, model_param in model_dict.items():
            if model_name == 'ARIMA':
                arima_block = ARIMA(**model_param)
                self.models.append(arima_block)
            elif model_name == 'linear_regressor':
                linear_block = nn.Linear(**model_param)
                self.models.append(linear_block)
            elif model_name == 'mlp':
                mlp_block = mlp(**model_param)
                self.models.append(mlp_block)
        
        self.voting_mlp = mlp([3*len(self.models), 4*len(self.models), 3])
        print("Ensemble Model Summary:", self.models)

    def voting(self, prediction_list, mode):

        if mode == 'average':
            y = sum(prediction_list) / len(prediction_list)
        if mode == 'mlp':
            prediction_tensor = torch.tensor(prediction_list)
            y = self.voting_mlp(prediction_tensor)


'''
model_dict = {'mlp': {'layer_dim_list': [10,20,30,3]},  
              'ARIMA': {'p': 2, 'd': 1, 'q': 1, 'ps': 1, 'ds': 1, 'qs': 1, 's': 1},
              'linear_regressor': {'in_features': 10, 'out_features': 3}
              }
model = ensemble_model(model_dict)
'''


            
