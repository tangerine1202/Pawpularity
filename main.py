from argparse import ArgumentParser
import logging
from pathlib import Path
import random

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from utils.data import load_data, split_train_val, CustomDataset
from model.baseline import ResNet18Model
from utils.debug import log_data_info

log = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f'Using device: {device}')


def loss_fn(outputs, labels):
    return torch.nn.functional.mse_loss(outputs, labels)


def main(args):
    train_df, test_df = load_data(args.data_dir, check=True)
    train_df, val_df = split_train_val(train_df, val_size=0.2)

    log_data_info(train_df, val_df, test_df)

    transform = None
    train_ds, val_ds = map(lambda df: CustomDataset(df, transform=transform), (train_df, val_df))
    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    model = ResNet18Model(pretrained=True, freeze_pretrained=False)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        log.info(f'Epoch {epoch + 1}/{args.epochs}')

        model.train()
        accumulated_val_loss = 0
        for batch in tqdm(train_dataloader):
            images, data, labels = map(lambda x: x.to(device), batch)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            accumulated_val_loss += loss.item()
            break
        loss = accumulated_val_loss / len(train_dataloader)
        log.info(f'Train Loss: {loss :.4f}')

        model.eval()
        with torch.inference_mode():
            accumulated_val_loss = 0
            for batch in tqdm(val_dataloader):
                images, data, labels = map(lambda x: x.to(device), batch)

                outputs = model(images)
                val_loss = loss_fn(outputs, labels)
                accumulated_val_loss += val_loss.item()
                break
        val_loss = accumulated_val_loss / len(val_dataloader)
        log.info(f'Val Loss: {val_loss :.4f}')

    # Save the model
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f'Saving model to {output_dir}')
    # Save the model state_dict
    torch.save(model.state_dict(), output_dir / 'model.pth')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default='data', help='Directory containing the dataset')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('-seed', '--seed', type=int, default=42)
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] - %(message)s',
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # ensures that CUDA convolution operations produce deterministic results
    torch.backends.cudnn.deterministic = True

    main(args)
