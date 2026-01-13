"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""
import os
import random
import yaml
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed: int) -> None:
    """Initialise les seeds (numpy/torch/python)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Les deux lignes suivantes sont parfois nécessaires pour une reproductibilité parfaite sur GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(prefer: str | None = "auto") -> str:
    """Retourne 'cpu' ou 'cuda' (ou choix basé sur 'auto')."""
    if prefer == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        else:
            raise RuntimeError("CUDA is not available, but 'cuda' device was explicitly preferred.")
    elif prefer == "cpu":
        return "cpu"
    
    # Mode "auto" ou cas par défaut
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model: nn.Module) -> int:
    """Retourne le nombre de paramètres entraînables du modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config_snapshot(config: dict, out_dir: str) -> None:
    """Sauvegarde une copie de la config (ex: YAML) dans out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    snapshot_path = os.path.join(out_dir, "config_snapshot.yaml")
    with open(snapshot_path, "w") as f:
        yaml.dump(config, f)