# training.py

import torch
import torch.optim as optim
from torchmetrics.regression import R2Score, MeanAbsoluteError
import torch.optim.lr_scheduler as lr_scheduler
from typing import Tuple

def train_model(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module,
                optimizer: optim.Optimizer, scheduler: lr_scheduler,checkpoint_path:str, num_epochs: int,device: torch.device, patience=30) -> None:
    """
    Trains the model on the training dataset for the specified number of epochs.
    """
    r2_score = R2Score().to(device)
    mae = MeanAbsoluteError().to(device)
    best_train_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        r2_score.reset()
        mae.reset()
        
        for batch_X, dist_ids, prop_ids, batch_y in train_loader:
            batch_X, dist_ids, prop_ids, batch_y = batch_X.to(device), dist_ids.to(device), prop_ids.to(device), batch_y.to(device).unsqueeze(1)
            
            outputs = model(batch_X, dist_ids, prop_ids)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            r2_score.update(outputs, batch_y)
            mae.update(outputs, batch_y)
        
        avg_loss = epoch_loss / len(train_loader)
        avg_r2 = r2_score.compute().item()
        avg_mae = mae.compute().item()
        scheduler.step(avg_loss)
       
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, R2: {avg_r2:.4f}")
        
        # Early Stopping
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            no_improve_count = 0
            # Save model checkpoint
            checkpoint = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_train_loss': best_train_loss,
                'mae': mae,
                'r2_score': avg_r2,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"New best model at epoch {epoch+1}, Loss: {best_train_loss:.4f}")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("Early stopping triggered.")
                break
