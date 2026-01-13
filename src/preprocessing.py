"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""
import torchvision.transforms as T

def get_preprocess_transforms(config: dict):
    """Retourne les transformations de pré-traitement."""
    transforms_list = []

    # Redimensionnement
    if 'resize' in config['preprocess'] and config['preprocess']['resize']:
        transforms_list.append(T.Resize(config['preprocess']['resize']))

    # Conversion en tenseur
    transforms_list.append(T.ToTensor())

    # Normalisation
    if 'normalize' in config['preprocess'] and config['preprocess']['normalize']:
        transforms_list.append(T.Normalize(
            mean=config['preprocess']['normalize']['mean'],
            std=config['preprocess']['normalize']['std']
        ))

    return T.Compose(transforms_list)