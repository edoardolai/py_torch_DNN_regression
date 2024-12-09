# evaluation.py

import torch
from torchmetrics.regression import R2Score, MeanAbsoluteError
from typing import Tuple
import os 

def evaluate_model(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module,
                   device: torch.device,checkpoint_path: str) -> Tuple[float, float, float]:
    """
    Evaluates the model on the test dataset and returns the loss, MAE, and R² score.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
    print("Best model loaded.")
    model.eval()
    epoch_loss = 0
    r2_score = R2Score().to(device)
    mae = MeanAbsoluteError().to(device)
    
    with torch.no_grad():
        for batch_X, dist_ids, prop_ids, batch_y in test_loader:
            batch_X, dist_ids, prop_ids, batch_y = batch_X.to(device), dist_ids.to(device), prop_ids.to(device), batch_y.to(device).unsqueeze(1)
            
            outputs = model(batch_X, dist_ids, prop_ids)
            loss = criterion(outputs, batch_y)
            
            epoch_loss += loss.item()
            r2_score.update(outputs, batch_y)
            mae.update(outputs, batch_y)
    
    avg_loss = epoch_loss / len(test_loader)
    avg_r2 = r2_score.compute().item()
    avg_mae = mae.compute().item()

    return avg_loss, avg_mae, avg_r2