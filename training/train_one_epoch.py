import numpy as np
import gc
import torch

def train_one_epoch(model, optimizer, loss_fn, dataloader, complete, train):

    # Initialize the lists of predictions and true values
    y_pred_list = []
    y_true_list = []

    loss_epoch = 0 # Initialize the loss
    if train:
        model.train() # Set the model in Training mode
    else:
        model.eval() # Set the model in Evaluation mode
    
    for X, y_true in dataloader:
        
        # Set the device
        device = X.device

        if train:
            # Put the gradient to zero
            optimizer.zero_grad()
        
        # Make predictions
        if complete:
            y_true = y_true.unsqueeze(1).detach()
            y_pred = model(X, y_true)
        else:
            y_pred = model(X)
        #print(y_pred)

        # Compute the loss
        loss = loss_fn(y_pred, y_true)

        if train:
            # Compute the gradient
            loss.backward()

            # Update the weights
            optimizer.step()
        
        # Accumulate the loss in this epoch
        loss_epoch += loss.detach().item()

        # Add predictions and true values to the lists
        if complete:
            y_true_list += y_true.squeeze(1).cpu().tolist()
            y_pred_list += y_pred.squeeze(1).cpu().tolist()
        else:
            y_true_list += y_true.cpu().tolist()
            y_pred_list += y_pred.cpu().tolist()

        del X, y_true, y_pred, loss
        gc.collect()

    if device == 'cuda':
        torch.cuda.empty_cache()
        
    # Compute the average loss
    loss_avg = loss_epoch / len(dataloader)

    return loss_avg, y_true_list, y_pred_list