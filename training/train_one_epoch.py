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
            if complete:
                for optim in optimizer:
                    optim.zero_grad()
            else:
                optimizer.zero_grad()
        
        # Make predictions
        if complete:
            y_true = y_true.detach()
            y_pred, y_pred_models = model(X, y_true)
        else:
            y_pred = model(X)
        #print(y_pred)

        # Compute the loss
        if complete:
            loss = [loss_fn(y_pred, y_true[:,1,:].unsqueeze(1)) for y_pred in y_pred_models]
        else:
            loss = loss_fn(y_pred, y_true[:,1,:].unsqueeze(1))

        if train:
            # Compute the gradient
            if complete:
                for l in loss:
                    l.backward(retain_graph=True)
            else:
                loss.backward()

            # Update the weights
            if complete:
                for optim in optimizer:
                    optim.step()
            else:
                optimizer.step()
            
        
        # Accumulate the loss in this epoch
        if complete:
            loss_epoch += torch.mean(torch.tensor(loss)).detach().item()
        else:
            loss_epoch += loss.detach().item()

        # Add predictions and true values to the lists
        y_true_list += y_true[:,1,:].cpu().tolist()
        y_pred_list += y_pred.squeeze(1).cpu().tolist()

        del X, y_true, y_pred, loss
        gc.collect()

    if device == 'cuda':
        torch.cuda.empty_cache()
        
    # Compute the average loss
    loss_avg = loss_epoch / len(dataloader)

    return loss_avg, y_true_list, y_pred_list