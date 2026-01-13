"""
Évaluation du modèle sur le jeu de test.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import torchmetrics

from src.utils import set_seed, get_device
from src.data_loading import get_dataloaders
from src.model import build_model

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes):
    """Évalue le modèle sur un jeu de données."""
    model.eval()
    total_loss = 0
    
    # Métriques
    accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_classes).to(device)
    f1_score = torchmetrics.F1Score(task="multilabel", num_labels=num_classes).to(device)

    progress_bar = tqdm(dataloader, desc="Evaluating")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        # Mettre à jour les métriques
        preds = torch.sigmoid(outputs)
        accuracy.update(preds, targets.int())
        f1_score.update(preds, targets.int())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy.compute()
    f1 = f1_score.compute()

    print(f"\n--- Test Results ---")
    print(f"Loss:     {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"--------------------")

    return avg_loss, acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    args = parser.parse_args()

    # --- 1. Charger la configuration ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- 2. Setup de l'environnement ---
    set_seed(config['train']['seed'])
    device = get_device(config['train']['device'])
    print(f"Using device: {device}")

    # --- 3. Charger les données de test ---
    print("Loading test data...")
    _, _, test_loader, meta = get_dataloaders(config)
    print(f"Data loaded. Test batches: {len(test_loader)}")

    # --- 4. Construire le modèle ---
    print("Building model...")
    model = build_model(config)
    
    # --- 5. Charger le checkpoint ---
    print(f"Loading checkpoint from {args.checkpoint}...")
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    model.to(device)

    # --- 6. Évaluation ---
    criterion = nn.BCEWithLogitsLoss()
    evaluate(model, test_loader, criterion, device, meta['num_classes'])

if __name__ == "__main__":
    main()
