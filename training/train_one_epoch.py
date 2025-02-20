import numpy as np
import gc
import torch

def train_one_epoch(model, optimizer, loss_fn, dataloader, complete, autoencoder, train):

    # Initialize the lists of predictions and true values
    y_pred_list = []
    y_true_list = []

    loss_epoch = 0 # Initialize the loss for the model
    loss_epoch_autoencoder = 0 # Initialize the loss of the autoencoder

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
            if autoencoder:
                y_pred, y_pred_models, x_pred = model(X, y_true) # In this case the model return also the x_pred for the autoencoder update! 
            else:
                y_pred, y_pred_models = model(X, y_true)
        else:
            y_pred = model(X)
        #print(y_pred)

        # Compute the loss
        if complete:
            loss = [loss_fn(y_pred, y_true[:,1,:].unsqueeze(1)) for y_pred in y_pred_models]
            if autoencoder:
                loss.append(loss_fn(x_pred, X)) # Loss computation of Autoencoder compare X and x_pred because the autoencoder reconstruct the input!
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
            if autoencoder:
                loss_epoch += torch.mean(torch.tensor(loss[:-1])).detach().item()
                loss_epoch_autoencoder += torch.tensor(loss[-1]).detach().item()
            else:
                loss_epoch += torch.mean(torch.tensor(loss)).detach().item()
        else:
            loss_epoch += loss.detach().item()

        # Add predictions and true values to the lists
        y_true_list += y_true[:,1,:].cpu().tolist()
        y_pred_list += y_pred.squeeze(1).cpu().tolist()
        if autoencoder:
            x_true_list += X.cpu().tolist()
            x_pred_list += x_pred.cpu().tolist()

        del X, y_true, y_pred, loss
        gc.collect()

    if device == 'cuda':
        torch.cuda.empty_cache()
        
    # Compute the average loss
    loss_avg = loss_epoch / len(dataloader)
    loss_avg_autoencoder = loss_avg_autoencoder / len(dataloader)

    return loss_avg, loss_avg_autoencoder, y_true_list, y_pred_list, x_true_list, x_pred_list 