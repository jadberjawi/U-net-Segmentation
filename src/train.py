import os
import torch
import yaml
import wandb
import argparse
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.transforms import AsDiscrete

from src.data_loader import get_dataloaders
from src.model import get_model
from src.utils import get_device

def train(config_path):

    # 1. Setup WandB
    wandb.init(
        project=config['project_name'], 
        name=f"{config['experiment_name']}_fold{config['data']['train_fold']}",
        config=config
    )
    
    # Set deterministic training for reproducibility
    set_determinism(seed=42)
    device = get_device('auto')

    # 2. Data & Model
    train_loader, val_loader = get_dataloaders(config)
    model = get_model(config).to(device)
    
    loss_function = DiceCELoss(
        to_onehot_y=True, 
        softmax=True, 
        lambda_dice=1.0,  # Weight for Dice score
        lambda_ce=1.0     # Weight for Cross Entropy (The "Force" fix)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # 3. Training Loop
    best_metric = -1
    best_metric_epoch = -1
    
    for epoch in range(config['training']['max_epochs']):
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= step
        wandb.log({"train/loss": epoch_loss, "epoch": epoch})
        print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}")

        # 4. Validation Loop
        if (epoch + 1) % config['training']['val_interval'] == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = model(val_inputs)
                    
                    # Post-processing for metric calculation
                    # Convert raw output logits -> Discrete (0 or 1)
                    val_outputs = AsDiscrete(argmax=True, to_onehot=2)(val_outputs)
                    val_labels = AsDiscrete(to_onehot=2)(val_labels)
                    
                    dice_metric(y_pred=val_outputs, y=val_labels)
                
                metric = dice_metric.aggregate().item()
                dice_metric.reset()

                wandb.log({"val/dice_score": metric, "epoch": epoch})
                
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    save_path = os.path.join("models", f"best_model_fold{config['data']['train_fold']}.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f"New Best Metric: {metric:.4f} at epoch {best_metric_epoch}")

    print(f"Training completed. Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--fold", type=int, default=None, help="Override fold number from config")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override fold if provided in command line
    if args.fold is not None:
        print(f"Overriding config fold with: {args.fold}")
        config['data']['train_fold'] = args.fold
        
    # Pass the updated config dictionary to train()
    # Note: You need to modify train() to accept a dict, not a path, 
    # OR just re-save the dict temporarily. 
    # A cleaner way is to pass the loaded config dict directly to train functions.
    
    # Let's adjust the train function signature in your mind:
    # def train(config): ... instead of def train(config_path):
    
    train(config)