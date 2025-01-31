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

def test(model, model_path, test_dataloader, loss_fn, complete):

    # Definition of the test metrics dictionary
    test_metrics = {
        'rmse':[], 
        'mae':[],
        'r2':[]       
               }
  
    # Load the weights
    model.load(model_path)

    # Set the model in evaluation mode
    model.eval()

    loss = 0  # value of the test loss
    # Lists of predictions and true labels
    y_pred_list = []
    y_true_list = []
    # List of inference time
    inference_time_list = []

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):

            # Take input and label from the list [input, label]
            print("Data: ", data)
            x, y_true = data       # tuple of tensor -> data = ([8,5,4], [8,3]) where 8 is batch size
            print("x: ", x.shape)
            print("y_true: ", y_true.shape)
               
            # Counting the time for inference
            # Start counter
            start_time = time.time()

            # Make predictions
            # If we use ensemble model, then the model forward method should have y_true for the voting system
            # Else (LSTM only) the forward need only the input x
            if complete:
                y_true = y_true.detach() # 
                y_pred = model(x, y_true)  # y_pred = []
                print("y_pred:", y_pred.shape)
            else:
                y_pred = model(x)
                print("y_pred:", y_pred.shape)

            # Stop counter
            end_time = time.time()
            inference_time = end_time - start_time # Compute the inference time 
            inference_time_list.append(inference_time) # Append the inference time for each batch in the list

            # Compute the loss
            loss_value = loss_fn(y_pred, y_true[:,1,:].unsqueeze(1))
            loss += loss_value.detach().item() # Incremental value for the average

            # Create the list of the y_true and y_pred
            # Transform the tensor(, device='cuda:0') in a list [] and summing the lists y_true_list = [ ...]
            y_true_list += y_true[:,1,:].cpu().tolist()
            y_pred_list += y_pred.squeeze(1).cpu().tolist()

            print("List of y_true:", y_true_list)
            print("List of y_pred:", y_pred_list)
            
    # Average Loss in testing
    loss_avg = loss / len(test_dataloader)   # tensor(value, device = 'cuda:0')
    # Average of the training time
    total_inference_time = sum(inference_time_list)
    inference_time_avg = total_inference_time / len(inference_time_list) 
    
    # Calculate the metrics
    test_metrics = calculate_metrics(y_true_list, y_pred_list, test_metrics)

    # We take all the path without "model.pt" 
    # (Ex. "./results/training_2025-01-28_11-11/model.pt" -> "./results/training_2025-01-28_11-11/")
    results_path = model_path[:-8]

    with open(results_path +'/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f)
   
    return test_metrics, loss_avg, total_inference_time, inference_time_avg, y_true_list, y_pred_list