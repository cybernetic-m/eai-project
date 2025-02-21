import os
import sys
import torch
import time
import json

# Get the absolute paths of the directories containing the utils functions 
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))

# Add these directories to sys.path
sys.path.append(utils_path)

# Import section
from calculate_metrics import calculate_metrics

def test(model, model_path, test_dataloader, loss_fn, complete, autoencoder):

    # Load the weights
    model.load(model_path)

    # Set the model in evaluation mode
    model.eval()

    loss_model = 0 # value of the test loss for the model
    loss_autoencoder = 0 #  value of the loss for the autoencoder
    # Lists of predictions and true labels
    y_pred_list = []
    y_true_list = []
    x_pred_list = []
    x_true_list = []
    # List of inference time
    inference_time_list = []

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):

            # Take input and label from the list [input, label]
            #print("Data: ", data)
            x, y_true = data       # tuple of tensor -> data = ([8,5,4], [8,3]) where 8 is batch size
            #print("x: ", x.shape)
            #print("y_true: ", y_true.shape)
               
            # Counting the time for inference
            # Start counter
            start_time = time.time()

            # Make predictions
            # If we use ensemble model, then the model forward method should have y_true for the voting system
            # Else (LSTM only) the forward need only the input x
            if complete:
                y_true = y_true.detach() 
                if autoencoder:
                    (y_pred, y_pred_models), x_pred = model(x, y_true)
                else:
                    y_pred, y_pred_models = model(x, y_true)  
                #print("y_pred:", y_pred.shape)
            else:
                y_pred = model(x)
                #print("y_pred:", y_pred.shape)

            # Stop counter
            end_time = time.time()
            inference_time = end_time - start_time # Compute the inference time 
            inference_time_list.append(inference_time) # Append the inference time for each batch in the list

            # Compute the loss
            loss_model += loss_fn(y_pred, y_true[:,1,:].unsqueeze(1)).detach().item()

            if autoencoder:
                x = x.permute(0,2,1)
                loss_autoencoder += loss_fn(x_pred, x).detach().item()

            # Create the list of the y_true and y_pred
            # Transform the tensor(, device='cuda:0') in a list [] and summing the lists y_true_list = [ ...]
            y_true_list += y_true[:,1,:].cpu().tolist()  # y_tru[:,1,:] because y_true[:,0,:] is the present value sended to ARIMA, while y_true[:,1,:] is the future value to be predicted
            y_pred_list += y_pred.squeeze(1).cpu().tolist()
            if autoencoder:
                x_true_list += x.cpu().tolist()
                x_pred_list += x_pred.cpu().tolist()

            #print("List of y_true:", y_true_list)
            #print("List of y_pred:", y_pred_list)
            
    # Average Loss in testing
    loss_model_avg = loss_model / len(test_dataloader)   # tensor(value, device = 'cuda:0')
    loss_autoencoder_avg = loss_autoencoder / len(test_dataloader)   # tensor(value, device = 'cuda:0')

    # Average of the training time
    total_inference_time = sum(inference_time_list)
    inference_time_avg = total_inference_time / len(inference_time_list) 
    
    # Definition of the test metrics dictionary for the model
    test_model_metrics = {
    'rmse':[],
    'rmse_ref':[],
    'mae':[],
    'mae_ref':[],
    'r2':[],
    'r2_ref':[],
    'loss_avg': loss_model_avg,
    'total_inference_time': total_inference_time,
    'inference_time_avg': inference_time_avg
    }

    # Definition of the test metrics dictionary for the autoencoder
    test_autoencoder_metrics = {
    'rmse':[],
    'rmse_ref':[],
    'mae':[],
    'mae_ref':[],
    'r2':[],
    'r2_ref':[],
    'loss_avg': loss_autoencoder_avg,
    }
    
    # Calculate the metrics
    test_model_metrics = calculate_metrics(y_true_list, y_pred_list, test_model_metrics, train=False)
    if autoencoder:
        test_autoencoder_metrics = calculate_metrics(x_true_list ,x_pred_list, test_autoencoder_metrics, autoencoder = True, train=False)

    # We take all the path without "model.pt" 
    # (Ex. "./results/training_2025-01-28_11-11/model.pt" -> "./results/training_2025-01-28_11-11/")
    results_path = model_path[:-8]

    with open(results_path +'/test_model_metrics.json', 'w') as f:
        json.dump(test_model_metrics, f)
    if autoencoder:
        with open(results_path +'/test_autoencoder_metrics.json', 'w') as f:
            json.dump(test_autoencoder_metrics, f)
   
    return test_model_metrics, test_autoencoder_metrics, loss_model_avg, loss_autoencoder_avg, total_inference_time, inference_time_avg, y_true_list, y_pred_list, x_true_list, x_pred_list