"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""
import torch
import os
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import Compose

from src.preprocessing import get_preprocess_transforms
from src.augmentation import get_augmentation_transforms

class AttributeSelector:
    def __init__(self, attr_indices):
        self.attr_indices = attr_indices

    def __call__(self, target):
        return torch.tensor([target[i] for i in self.attr_indices], dtype=torch.float32)

def get_dataloaders(config: dict):
    """
    Crée et retourne les DataLoaders d'entraînement/validation/test et des métadonnées.
    """
    dataset_config = config['dataset']
    train_config = config['train']

    # Transformations
    preprocess_transforms = get_preprocess_transforms(config)
    augment_transforms = get_augmentation_transforms(config)

    train_transform = Compose([
        augment_transforms,
        preprocess_transforms
    ]) if augment_transforms else preprocess_transforms
    
    eval_transform = preprocess_transforms

    # Création d'une instance pour récupérer les noms d'attributs
    # C'est un peu lourd, mais c'est une façon robuste de le faire sans hardcoder.
    try:
        temp_dataset = CelebA(root=dataset_config['root'], split='all', download=dataset_config['download'])
    except RuntimeError as e:
        if "already exists" in str(e):
            # This can happen in multiprocessing contexts, it's safe to ignore
            temp_dataset = CelebA(root=dataset_config['root'], split='all', download=False)
        else:
            raise e

    all_attr_names = temp_dataset.attr_names
    
    # Sélection des attributs
    attributes_to_select = dataset_config['attributes']
    attr_indices = [all_attr_names.index(attr) for attr in attributes_to_select]

    target_transform = AttributeSelector(attr_indices)

    # Création des datasets
    train_dataset = CelebA(
        root=dataset_config['root'],
        split='train',
        transform=train_transform,
        target_transform=target_transform,
        download=dataset_config['download']
    )

    val_dataset = CelebA(
        root=dataset_config['root'],
        split='valid',
        transform=eval_transform,
        target_transform=target_transform,
        download=dataset_config['download']
    )

    test_dataset = CelebA(
        root=dataset_config['root'],
        split='test',
        transform=eval_transform,
        target_transform=target_transform,
        download=dataset_config['download']
    )

    # Creation des DataLoaders
    # num_workers is taken from config. If os.name == 'nt', it might need to be 0
    # due to multiprocessing issues. However, for performance, we will let the config
    # decide. If issues arise, user might need to set it to 0 manually.
    num_workers = dataset_config.get('num_workers', 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=dataset_config['shuffle'],
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Création du dictionnaire de métadonnées
    meta = {
        "num_classes": config['model']['num_classes'],
        "input_shape": tuple(config['model']['input_shape'])
    }

    return train_loader, val_loader, test_loader, meta
