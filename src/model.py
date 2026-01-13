"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""
import torch
import torch.nn as nn

class CNNMultiLabel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_config = config['model']
        in_channels = model_config['input_shape'][0]
        num_classes = model_config['num_classes']
        channels_block3 = model_config['channels_block3']
        dropout_prob = model_config['dropout']

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output shape: (batch, 64, 32, 32)

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Output shape: (batch, 128, 16, 16)

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, channels_block3, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_block3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        )
        # Output shape: (batch, channels_block3, 1, 1)

        # Head
        self.head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(channels_block3, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Flatten the output for the linear layer
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

def build_model(config: dict) -> nn.Module:
    """Construit et retourne un nn.Module selon la config."""
    model_type = config['model'].get('type')
    if model_type == 'cnn_multilabel':
        return CNNMultiLabel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")