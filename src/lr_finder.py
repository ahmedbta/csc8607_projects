"""
Recherche de taux d'apprentissage (LR finder). 
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import math

from src.utils import set_seed, get_device
from src.data_loading import get_dataloaders
from src.model import build_model

def find_lr(model, train_loader, criterion, optimizer, device, start_lr, end_lr, num_iter, writer):
    model.train()
    optimizer.param_groups[0]['lr'] = start_lr
    lr_lambda = lambda i: (end_lr / start_lr) ** (i / (num_iter - 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    losses = []
    lrs = []
    
    iterator = iter(train_loader)
    progress_bar = tqdm(range(num_iter), desc="LR Finder")
    
    for i in progress_bar:
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, targets = next(iterator)
            
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(loss.item())

        if writer:
            writer.add_scalar('lr_finder/loss', loss.item(), i)
            writer.add_scalar('lr_finder/lr', current_lr, i)
        
        progress_bar.set_postfix(lr=current_lr, loss=loss.item())
        
        if loss.item() > 4 * (min(losses) if losses else 1.0) and i > 10:
            print("Loss exploded. Stopping LR finder.")
            break

    # Plot to console
    print("\n--- LR Finder Results ---")
    for lr, loss in zip(lrs, losses):
        print(f"LR: {lr:.6f}, Loss: {loss:.4f}")
    print("--- End of LR Finder ---")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--start_lr", type=float, default=1e-7, help="Start learning rate.")
    parser.add_argument("--end_lr", type=float, default=1.0, help="End learning rate.")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of iterations.")
    parser.add_argument("--run_name", type=str, default="lr_finder", help="Name for the TensorBoard run.")
    args = parser.parse_args()

    # --- 1. Load configuration ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- 2. Setup environment ---
    set_seed(config['train']['seed'])
    device = get_device(config['train']['device'])
    print(f"Using device: {device}")

    # --- 3. Load data ---
    print("Loading data...")
    # We only need the training loader for the LR finder
    train_loader, _, _, _ = get_dataloaders(config)
    print(f"Data loaded. Train batches: {len(train_loader)}")

    # --- 4. Build model ---
    print("Building model...")
    model = build_model(config)
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer: we use the one from the config, but we will override the LR
    optim_config = config['train']['optimizer']
    if optim_config['name'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr, weight_decay=optim_config['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer: {optim_config['name']}")

    # --- 5. TensorBoard Setup ---
    writer = SummaryWriter(log_dir=os.path.join(config['paths']['runs_dir'], args.run_name))
    
    # --- 6. Run LR Finder ---
    print("\n--- Starting LR Finder ---")
    find_lr(model, train_loader, criterion, optimizer, device, args.start_lr, args.end_lr, args.num_iter, writer)

    writer.close()
    print("--- LR Finder Complete ---")


if __name__ == "__main__":
    main()
