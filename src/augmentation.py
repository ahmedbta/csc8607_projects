"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""
import torchvision.transforms as T

def get_augmentation_transforms(config: dict):
    """Retourne les transformations d'augmentation."""
    augment_config = config.get('augment', {})
    transforms_list = []

    if augment_config.get('random_flip'):
        transforms_list.append(T.RandomHorizontalFlip())

    # Add RandomRotation if enabled in config
    if augment_config.get('random_rotation', {}).get('enabled'):
        degrees = augment_config['random_rotation'].get('degrees', 10)
        transforms_list.append(T.RandomRotation(degrees))

    # Add ColorJitter if enabled in config
    if augment_config.get('color_jitter', {}).get('enabled'):
        cj_config = augment_config['color_jitter']
        brightness = cj_config.get('brightness', 0)
        contrast = cj_config.get('contrast', 0)
        saturation = cj_config.get('saturation', 0)
        hue = cj_config.get('hue', 0)
        transforms_list.append(T.ColorJitter(
            brightness=brightness, 
            contrast=contrast, 
            saturation=saturation, 
            hue=hue
        ))

    if not transforms_list:
        return None

    return T.Compose(transforms_list)