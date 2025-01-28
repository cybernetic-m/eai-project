import os
import sys
import torch
import time

# Get the absolute paths of the directories containing the utils functions and testing directory
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
testing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../testing'))

# Add these directories to sys.path
sys.path.append(utils_path)

# Import section
from utils import calculate_metrics, save_metrics

def test(model, model_path, test_dataloader, test_metrics_dict, loss_fn):

    # Definition of the test metrics dictionary
    test_metrics = {
        'rmse':[], 
        'mae':[],
        'r2':[]       
               }

    # Set the model in evaluation mode
    model.eval()
    
    # Load the weights
    model.load(model_path)

    loss = 0  # value of the test loss
    # Lists of predictions and true labels
    y_pred_list = []
    y_true_list = []
    inference_time_list = []

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):

            # Take input and label from the list [input, label]
            x, y_true = data       # data = [ tensor([[[waveform]]], [sample_rate]),   tensor([3], device='cuda:0')]
            
            # Take only the waveform for the input tuple (waveform, sample_rate)
            x = x[0]       # x[0] = tensor([[[waveform]]], [sample_rate])
            
            # Counting the time for inference
            # Start counter
            start_time = time.time()

            # Make predictions (Tensor with 7 values of probability, 1 for each class)
            y_pred = model(x)  # y_pred = tensor([[0.1254, 0.1152, 0.1396, 0.1600, 0.1058, 0.1394, 0.2146]],device='mps:0')

            # Stop counter
            end_time = time.time()
            inference_time = end_time - start_time
            inference_time_list.append(inference_time)
            
            # Compute the loss
            loss_value = loss_fn(y_pred, y_true)
            loss += loss_value.detach().item() # Incremental value for the average

            # Create the list of the y_true and y_pred
            # Transform the tensor([3], device='cuda:0') in a list [6] and summing the lists y_true_list = [6, 1, 2, 4 ...]
            y_true_list += y_true.tolist()  
            # Argmax take the index (position) of the maximum value in y_pred => torch.argmax(y_pred) = tensor(6, device='mps:0') and item() take only the value 6
            y_pred_list += torch.argmax(y_pred.detach(), dim=1).cpu().tolist()
            
    # Average Loss in testing
    loss_avg = loss / len(test_dataloader)   # tensor(value, device = 'cuda:0')

    calculate_metrics(y_true_list, y_pred_list, test_metrics_dict, loss_avg, i, inference_time_list, test=True)
       
    results_path = model_path[:-8]

    name_test_metrics = f'/{model.model_name}_test_metrics.json'
   
    save_metrics(test_metrics_dict, results_path + name_test_metrics)

    return inference_time_list