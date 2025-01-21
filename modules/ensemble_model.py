import torch
import torch.nn as nn
from ARIMA.TimeSeries import ARIMA
from modules import MLP


class ensemble_model(nn.Module):

    def __init__(self, model_dict):
        super(ensemble_model)

        models = [] # List of models
        
        for model_name, model_param in model_dict.items():
            if model_name == 'ARIMA':
                arima_block = ARIMA(**model_param)
                models.append(arima_block)
            elif model_name == 'linear_regressor':
                linear_block = nn.Linear(**model_param)
                models.append(linear_block)
            elif model_name == 'rnn':
                rnn_block = nn.RNN(**model_param)
                models.append(rnn_block)
            elif model_name == 'mlp':
                mlp_block = MLP(**model_param)
                models.append(mlp_block)




            
