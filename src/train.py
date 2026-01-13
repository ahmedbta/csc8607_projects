"""
Entraînement principal.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import torchmetrics

from src.utils import set_seed, get_device, count_parameters, save_config_snapshot
from src.data_loading import get_dataloaders
from src.model import build_model

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, max_steps=None):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [train]")
    
    for i, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (i + 1))
        
        if writer:
            writer.add_scalar('train/loss_step', loss.item(), epoch * len(dataloader) + i)

        if max_steps and i + 1 >= max_steps:
            break
            
    avg_loss = total_loss / len(dataloader)
    if writer:
        writer.add_scalar('train/loss', avg_loss, epoch)

    return avg_loss

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, writer, num_classes):
    model.eval()
    total_loss = 0
    
    # Using torchmetrics for accuracy
    accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes).to(device)
    f1_score = torchmetrics.F1Score(task="multilabel", num_labels=num_classes).to(device)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [val]")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        # Update metrics
        preds = torch.sigmoid(outputs)
        accuracy.update(preds, targets.int())
        f1_score.update(preds, targets.int())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy.compute()
    f1 = f1_score.compute()

    if writer:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/accuracy', acc, epoch)
        writer.add_scalar('val/f1', f1, epoch)
        
    print(f"Validation Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}, F1-Score={f1:.4f}")

    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--seed", type=int, default=None, help="Manual seed for reproducibility.")
    parser.add_argument("--overfit_small", action="store_true", help="Overfit on a small subset of the data.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override number of steps for training.")
    parser.add_argument("--run_name", type=str, default=None, help="Name for the TensorBoard run.")
    args = parser.parse_args()

    # --- 1. Load configuration ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with CLI args if provided
    if args.seed:
        config['train']['seed'] = args.seed
    if args.max_epochs:
        config['train']['epochs'] = args.max_epochs
    if args.max_steps and 'train' in config:
        config['train']['max_steps'] = args.max_steps
    if args.overfit_small:
        config['train']['overfit_small'] = True

    # --- 2. Setup environment ---
    set_seed(config['train']['seed'])
    device = get_device(config['train']['device'])
    print(f"DEBUG: Device selected by get_device(): {device}")
    print(f"Using device: {device}")

    # --- 3. Load data ---
    print("Loading data...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    if config['train'].get('overfit_small'):
        print("Overfitting on a small batch...")
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        train_loader = [train_batch] * 10 # Repeat the batch
        val_loader = [val_batch]

    # --- 4. Build model ---
    print("Building model...")
    model = build_model(config)
    model.to(device)
    print(f"Model built. Trainable parameters: {count_parameters(model):,}")
    
    criterion = nn.BCEWithLogitsLoss()
    
    optim_config = config['train']['optimizer']
    if optim_config['name'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer: {optim_config['name']}")

    # --- 5. TensorBoard Setup ---
    run_name = args.run_name or f"train_{os.path.basename(args.config).split('.')[0]}"
    writer = SummaryWriter(log_dir=os.path.join(config['paths']['runs_dir'], run_name))
    save_config_snapshot(config, os.path.join(config['paths']['runs_dir'], run_name))
    
    # --- 6. Training Loop ---
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    epochs = config['train']['epochs']
    
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, config['train'].get('max_steps'))
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer, meta['num_classes'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(config['paths']['artifacts_dir'], exist_ok=True)
            checkpoint_path = os.path.join(config['paths']['artifacts_dir'], 'best.ckpt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    writer.close()
    print("--- Training Complete ---")


if __name__ == "__main__":
    main()
