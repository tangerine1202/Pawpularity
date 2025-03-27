import logging
from pathlib import Path
import random
import gc

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.optim import AdamW

from utils.data import load_data, split_train_val, CustomDataset
from model.baseline import TorchVisionModel, ResNetModel, SwinTransformerModel
from utils.debug import log_data_info
from utils.wandb_utils import (
    init_wandb,
    watch_model,
    log_metrics,
    log_model_artifact,
    finish as finish_wandb
)

log = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loss_fn(outputs, labels):
    mse_loss = torch.nn.functional.mse_loss(outputs, labels)
    rmse_loss = torch.sqrt(mse_loss)
    return rmse_loss


def train(model, train_dataloader, optimizer, pbar):
    accumulated_train_loss = 0
    model.train()
    for batch in train_dataloader:
        images, data, labels = map(lambda x: x.to(device), batch)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        accumulated_train_loss += loss.item()
        pbar.update(1)
    training_loss = accumulated_train_loss / len(train_dataloader)
    return training_loss


def evaluate(model, val_dataloader):
    model.eval()
    with torch.inference_mode():
        accumulated_val_loss = 0
        for batch in tqdm(val_dataloader, desc='Validation', leave=False):
            images, data, labels = map(lambda x: x.to(device), batch)

            outputs = model(images)
            val_loss = loss_fn(outputs, labels)
            accumulated_val_loss += val_loss.item()
    val_loss = accumulated_val_loss / len(val_dataloader)
    return val_loss


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    log.info(f'Using device: {device}')
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize wandb
    run = init_wandb(cfg)

    # Set random seeds
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
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

    optimizer = AdamW(model.parameters(), lr=cfg.training.lr)

    # Watch model with wandb if enabled
    watch_model(model, log_freq=cfg.wandb.log_freq)

    # Data
    data_dir = Path(to_absolute_path(cfg.data.data_dir))
    train_df, test_df = load_data(data_dir, check=True)
    train_df, val_df = split_train_val(train_df, val_size=cfg.data.val_size)
    log_data_info(train_df, val_df, test_df)

    img_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(224, 224), antialias=True),
    ]
    if cfg.data.augmentation == 'TA':
        img_transforms += [v2.TrivialAugmentWide()]
    elif cfg.data.augmentation == 'RA':
        img_transforms += [v2.RandAugment()]
    elif cfg.data.augmentation == 'AA_img':
        img_transforms += [v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET)]
    elif cfg.data.augmentation == 'AA_svhn':
        img_transforms += [v2.AutoAugment(v2.AutoAugmentPolicy.SVHN)]
    elif cfg.data.augmentation == 'AA_cifar10':
        img_transforms += [v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)]
    else:
        raise NotImplementedError(f'Augmentation {cfg.data.augmentation} not implemented')

    train_ds, val_ds = map(lambda df: CustomDataset(df, img_transform=v2.Compose(img_transforms)), (train_df, val_df))
    train_dataloader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    with tqdm(total=cfg.training.epochs * len(train_dataloader), desc='Training', leave=False) as pbar:
        for epoch in range(cfg.training.epochs):
            log.info(f'Epoch {epoch + 1}/{cfg.training.epochs}')

            train_loss = train(model, train_dataloader, optimizer, pbar) * 100
            val_loss = evaluate(model, val_dataloader) * 100

            log.info(f'Train Loss: {train_loss:.4f}')
            log.info(f'Val   Loss: {val_loss:.4f}')
            pbar.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})

            # Log metrics
            log_metrics({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })

    # Save the model
    output_dir = Path(to_absolute_path("output")) / f'{cfg.exp_name}_{cfg.model.name}'
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'ckpt.pth'
    torch.save(model.state_dict(), model_path)
    log.info(f'Model saved to {model_path}')

    # Log model as artifact
    log_model_artifact(model_path, f"model-{cfg.exp_name}")

    # Finish wandb run
    finish_wandb()

    # Clean up CUDA memory
    del model
    del train_dataloader
    del val_dataloader
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
