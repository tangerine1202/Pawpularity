import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import sys

IN_KAGGLE = False
MODEL_NAME = 'ResNet18Model'
MODEL_CKPT = 'model.pth'
MODEL_ARGS = dict(
    pretrained=False,
    freeze_pretrained=False
)
MODEL_WEIGHT = f'/kaggle/input/v0-1-1/{MODEL_CKPT}' if IN_KAGGLE else f'output/{MODEL_CKPT}'
DATA_DIR = '/kaggle/input/petfinder-pawpularity-score' if IN_KAGGLE else 'data/'
OUTPUT_DIR = './' if IN_KAGGLE else 'output/'

# Add the directory containing your modules to sys.path
if IN_KAGGLE:
    sys.path.append('/kaggle/input/v0-1-1')
    from data import load_data, CustomDataset
    from baseline import *
else:
    from utils.data import load_data, CustomDataset
    from model.baseline import *


def inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = eval(MODEL_NAME)(**MODEL_ARGS).to(device)
    state_dict = torch.load(MODEL_WEIGHT, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Load the data
    _, test_df = load_data(DATA_DIR, check=False)
    dataset = CustomDataset(test_df, has_label=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)

    all_outputs = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc='Inference'):
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
    output_df.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)

    print(output_df.head(3))


if __name__ == '__main__':
    inference()
