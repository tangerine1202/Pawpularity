from argparse import ArgumentParser
import logging
from pathlib import Path
import random
import gc

from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.optim import AdamW

from utils.data import load_data, split_train_val, CustomDataset
from model.baseline import TorchVisionModel, ResNetModel, SwinTransformerModel
from utils.debug import log_data_info

log = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f'Using device: {device}')


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


def main(args):

    # Model
    if args.model.startswith('ResNet'):
        model_name = 'ResNetModel'
        model_args = dict(
            name=args.model,
            weights=f'{args.model}_Weights.DEFAULT',
            pretrained=True,
            freeze_pretrained=True
        )
    elif args.model.startswith('Swin'):
        model_name = 'SwinTransformerModel'
        model_args = dict(
            name=args.model,
            weights=f'{args.model}_Weights.DEFAULT',
            pretrained=True,
            freeze_pretrained=False
        )
    else:
        raise NotImplementedError(f'Model {args.model} not implemented')
    model = eval(model_name)(**model_args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Data
    train_df, test_df = load_data(args.data_dir, check=True)
    train_df, val_df = split_train_val(train_df, val_size=0.2)
    log_data_info(train_df, val_df, test_df)

    img_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(224, 224), antialias=True),
    ]
    if args.augmentation == 'TA':
        img_transforms += [v2.TrivialAugmentWide()]
    elif args.augmentation == 'RA':
        img_transforms += [v2.RandAugment()]
    elif args.augmentation == 'AA_img':
        img_transforms += [v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET)]
    elif args.augmentation == 'AA_svhn':
        img_transforms += [v2.AutoAugment(v2.AutoAugmentPolicy.SVHN)]
    elif args.augmentation == 'AA_cifar10':
        img_transforms += [v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)]
    else:
        raise NotImplementedError(f'Augmentation {args.augmentation} not implemented')
    # CutMix, Mixup https://pytorch.org/vision/0.21/auto_examples/transforms/plot_cutmix_mixup.html#where-to-use-mixup-and-cutmix

    train_ds, val_ds = map(lambda df: CustomDataset(df, img_transform=v2.Compose(img_transforms)), (train_df, val_df))
    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    with tqdm(total=args.epochs * len(train_dataloader), desc='Training', leave=False) as pbar:
        for epoch in range(args.epochs):
            log.info(f'Epoch {epoch + 1}/{args.epochs}')

            train_loss = train(model, train_dataloader, optimizer, pbar) * 100
            val_loss = evaluate(model, val_dataloader) * 100

            log.info(f'Train Loss: {train_loss:.4f}')
            log.info(f'Val   Loss: {val_loss:.4f}')
            pbar.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})

    # Save the model
    output_dir = Path('output') / f'{args.exp_name}_{args.model}'
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'ckpt.pth'
    torch.save(model.state_dict(), model_path)
    log.info(f'Model saved to {model_path}')

    # Clean up CUDA memory
    del model
    del train_dataloader
    del val_dataloader
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('exp_name', type=str, help='Experiment name')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name')
    parser.add_argument('-aug', '--augmentation', type=str, default='TA', help='Data augmentation method')
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('-seed', '--seed', type=int, default=42)
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--data_dir', type=Path, default='data', help='Directory containing the dataset')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] - %(message)s',
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    main(args)
