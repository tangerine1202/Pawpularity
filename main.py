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
import torch.multiprocessing as mp

from utils.data import load_data, split_train_val, CustomDataset
from model.baseline import ResNetModel, SwinTransformerModel
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
    accumulated_se = np.array([])

    model.train()
    for batch in train_dataloader:
        images, data, labels = map(lambda x: x.to(device), batch)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        accumulated_train_loss += loss.item()
        accumulated_se = np.append(accumulated_se, ((outputs - labels) ** 2).detach().cpu().numpy())
        pbar.set_postfix({'loss': loss.item(), 'rmse': np.sqrt(accumulated_se.mean())})
        pbar.update(1)

    info = {
        'loss': accumulated_train_loss / len(train_dataloader),
        'rmse': np.sqrt(accumulated_se.mean()).item(),
    }
    return info


def evaluate(model, val_dataloader):
    model.eval()
    with torch.inference_mode():
        accumulated_val_loss = 0
        accumulated_se = np.array([])

        pbar = tqdm(total=len(val_dataloader), desc='Validation')
        for batch in val_dataloader:
            images, data, labels = map(lambda x: x.to(device), batch)

            outputs = model(images)
            val_loss = loss_fn(outputs, labels)

            accumulated_val_loss += val_loss.item()
            accumulated_se = np.append(accumulated_se, ((outputs - labels) ** 2).detach().cpu().numpy())
            pbar.set_postfix({'loss': val_loss.item(), 'rmse': np.sqrt(accumulated_se.mean())})
            pbar.update(1)

    info = {
        'loss': accumulated_val_loss / len(val_dataloader),
        'rmse': np.sqrt(accumulated_se.mean()).item(),
    }
    return info


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

    val_img_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(224, 224), antialias=True),
    ]
    img_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(448, 448), antialias=True),
    ]
    if cfg.data.augmentation is None:
        pass
    elif cfg.data.augmentation == 'TA':
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
    img_transforms += [
        v2.Resize(size=(224, 224), antialias=True),
    ]

    mp.set_start_method('spawn')
    train_ds = CustomDataset(
        train_df,
        img_transform=v2.Compose(img_transforms),
        use_depth=cfg.data.use_depth,
    )
    val_ds = CustomDataset(
        val_df,
        img_transform=v2.Compose(val_img_transforms),
        use_depth=cfg.data.use_depth,
    )
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

    pbar = tqdm(total=cfg.training.epochs * len(train_dataloader), desc='Training')
    for epoch in range(cfg.training.epochs):
        log.info(f'Epoch {epoch + 1}/{cfg.training.epochs}')

        train_info = train(model, train_dataloader, optimizer, pbar)
        val_info = evaluate(model, val_dataloader)

        info = {
            "epoch": epoch + 1,
            **{'train_' + k: v for k, v in train_info.items()},
            **{'val_' + k: v for k, v in val_info.items()},
        }
        msg = '\n'.join([f'{k}: {v:.4f}' for k, v in info.items()])
        log.info(msg)
        log_metrics(info)

    # Save the model
    output_dir = Path(to_absolute_path("output")) / f'{cfg.exp_name}_{cfg.model.name}'
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'ckpt.pth'
    eval_path = output_dir / 'eval.txt'
    torch.save(model.state_dict(), model_path)
    log.info(f'Model saved to {model_path}')
    with open(eval_path, 'w') as f:
        for k, v in info.items():
            f.write(f'{k}: {v:.4f}\n')
        f.write(f'Config: {OmegaConf.to_yaml(cfg)}\n')
    log.info(f'Evaluation saved to {eval_path}')

    # Log model as artifact
    # log_model_artifact(model_path, f"model-{cfg.exp_name}")

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
