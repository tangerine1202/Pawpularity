IN_KAGGLE = False
MODEL_CLASS_NAME = 'SwinTransformerModel'
MODEL_NAME = 'Swin_V2_T'
MODEL_CKPT = 'ckpt.pth'
# Only used in Kaggle
KAGGLE_DIR = 'kaggle-files'

MODEL_WEIGHT = f'/kaggle/input/{KAGGLE_DIR}/{MODEL_CKPT}' if IN_KAGGLE else f'output/{MODEL_CKPT}'
CODE_PATH = f'/kaggle/input/{KAGGLE_DIR}' if IN_KAGGLE else None
DATA_DIR = '/kaggle/input/petfinder-pawpularity-score' if IN_KAGGLE else 'data/'
OUTPUT_DIR = './' if IN_KAGGLE else 'output/'

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import sys
import os
from pathlib import Path

# Set up paths and imports based on environment
if IN_KAGGLE:
    # Add the directory containing your modules to sys.path
    sys.path.append(CODE_PATH)
    from data import load_data, CustomDataset  # type: ignore
    from baseline import *  # type: ignore
    # Import minimal OmegaConf functionality for config compatibility
    from omegaconf import OmegaConf
else:
    from utils.data import load_data, CustomDataset
    from model.baseline import *
    from omegaconf import OmegaConf, DictConfig
    try:
        import hydra
        from hydra.utils import to_absolute_path
        HYDRA_AVAILABLE = True
    except ImportError:
        HYDRA_AVAILABLE = False


def create_config():
    """Create a configuration object compatible with the Hydra setup."""
    # Default configuration that mimics our hydra config structure
    config_dict = {
        "exp_name": os.path.splitext(MODEL_CKPT)[0],
        "model": {
            "class_name": MODEL_CLASS_NAME,
            "name": MODEL_NAME,
            "pretrained": False,
            "freeze_pretrained": False,
            "weights": None  # We don't need weights for inference as we're loading a saved model
        },
        "data": {
            "data_dir": DATA_DIR,
            "augmentation": "TA",
            "val_size": 0.2
        },
        "training": {
            "batch_size": 32,
            "seed": 42
        },
        "wandb": {
            "use_wandb": False
        },
        "inference": {
            "model_weight": MODEL_WEIGHT,
            "output_dir": OUTPUT_DIR
        }
    }

    # Create an OmegaConf object from the dictionary
    return OmegaConf.create(config_dict)


def inference(cfg):
    """Run inference using the provided configuration"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Set random seeds
    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # Model instantiation using config
    model_mapping = {
        "ResNetModel": ResNetModel,
        "SwinTransformerModel": SwinTransformerModel,
    }
    model_class_name = cfg.model.class_name
    if model_class_name not in model_mapping:
        raise ValueError(f"Model class '{model_class_name}' is not recognized.")
    # Extract model args from config, excluding class_name
    model_args = {k: v for k, v in dict(cfg.model).items() if k != 'class_name'}
    model = model_mapping[model_class_name](**model_args).to(device)

    model_weight = cfg.inference.model_weight
    state_dict = torch.load(model_weight, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Load the data
    img_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(224, 224), antialias=True),
    ]

    data_dir = cfg.data.data_dir
    if not IN_KAGGLE and HYDRA_AVAILABLE:
        data_dir = to_absolute_path(data_dir)

    _, test_df = load_data(data_dir, check=False)
    dataset = CustomDataset(test_df, has_label=False, img_transform=v2.Compose(img_transforms))
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size)

    # Run inference
    all_outputs = []
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            images, data = map(lambda x: x.to(device), batch)
            outputs = model(images)
            # Denormalize the labels
            outputs = dataset.denormalize_labels(outputs)
            all_outputs.append(outputs.cpu())
    all_outputs = torch.cat(all_outputs, dim=0)

    # Save predictions to CSV
    output_df = pd.DataFrame({
        'Id': test_df['Id'],
        'Pawpularity': all_outputs.squeeze().numpy()}
    )

    output_path = Path(cfg.inference.output_dir) / 'submission.csv'
    output_df.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    print(output_df.head(8))


if __name__ == '__main__':
    if not IN_KAGGLE and HYDRA_AVAILABLE:
        # Use Hydra for local environment if available
        @hydra.main(version_base=None, config_path="conf", config_name="config")
        def main(cfg: DictConfig):
            # Add inference-specific configurations
            cfg.inference = {
                "model_weight": MODEL_WEIGHT,
                "output_dir": OUTPUT_DIR
            }

            # Override specific model parameters for inference
            cfg.model.class_name = MODEL_CLASS_NAME
            cfg.model.name = MODEL_NAME
            cfg.model.pretrained = False
            cfg.model.freeze_pretrained = False
            cfg.model.weights = None

            inference(cfg)

        main()
    else:
        # Use created config for Kaggle or if Hydra is not available
        cfg = create_config()
        inference(cfg)
