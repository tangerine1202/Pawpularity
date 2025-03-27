# %% [code]
IN_KAGGLE = True
DS_PREFIX = '0327'
MODEL_NAME = 'ResNetModel'
MODEL_CKPT = 'TAaug_ResNet18.pth'
MODEL_ARGS = dict(
    name='ResNet18',
    pretrained=False,
    freeze_pretrained=False,
)
# MODEL_NAME = 'SwinTransformerModel'
# MODEL_CKPT = 'TAaug_Swin_V2_B.pth'
# MODEL_ARGS = dict(
#     name='Swin_V2_B',
#     pretrained=False,
#     freeze_pretrained=False,
# )

CKPT_DS_NAME = f'{DS_PREFIX}-model'
CODE_DS_NAME = f'{DS_PREFIX}-code'
MODEL_WEIGHT = f'/kaggle/input/{CKPT_DS_NAME}/{MODEL_CKPT}' if IN_KAGGLE else f'output/{MODEL_CKPT}'
CODE_PATH = f'/kaggle/input/{CODE_DS_NAME}' if IN_KAGGLE else None
DATA_DIR = '/kaggle/input/petfinder-pawpularity-score' if IN_KAGGLE else 'data/'
OUTPUT_DIR = './' if IN_KAGGLE else 'output/'

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import sys
if IN_KAGGLE:
    # Add the directory containing your modules to sys.path
    sys.path.append(CODE_PATH)
    from data import load_data, CustomDataset  # type: ignore
    from baseline import *  # type: ignore
else:
    from utils.data import load_data, CustomDataset
    from model.baseline import *


def inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load the model
    model = eval(MODEL_NAME)(**MODEL_ARGS).to(device)
    state_dict = torch.load(MODEL_WEIGHT, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    # Load the data
    img_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(224, 224), antialias=True),
    ]
    _, test_df = load_data(DATA_DIR, check=False)
    dataset = CustomDataset(test_df, has_label=False, img_transform=v2.Compose(img_transforms))
    dataloader = DataLoader(dataset, batch_size=32)

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
    output_df.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)

    print(output_df.head(8))


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    inference()
