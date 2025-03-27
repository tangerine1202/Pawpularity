# Pawpularity (Team 14)

This project predicts the "Pawpularity" score of pet images from the [PetFinder.my Pawpularity Contest](https://www.kaggle.com/c/petfinder-pawpularity-score) on Kaggle. The score represents how appealing a pet's photo is to potential adopters.


## Code Structure

- `main.py` - Main training script
- `exp.sh` - Helper script to run multiple training experiments
- `data/` - Contains training and test data:
  - `train/` - Training images
  - `test/` - Test images
  - `train.csv` - Training data
  - `test.csv` - Test data
- `model/` - Neural network model implementations
- `utils/` - Utility scripts:
  - `data.py` - Dataset loading and preprocessing
  - `debug.py` - Logging and debugging utilities
- `inference.py` - Prediction script for submissions


## Setup

### Prepare Data

1. Download the dataset from [Kaggle](https://www.kaggle.com/c/petfinder-pawpularity-score/data).
2. Unzip the dataset to get the `train` and `test` folders, and the `train.csv` and `test.csv` files.
3. Move folders and files to the `data/` directory:
   ```
   data/
   ├── train/
   ├── test/
   ├── train.csv
   └── test.csv
   ```

### Install Dependencies

#### Using `uv` (Recommended)

[`uv`](https://github.com/astral-sh/uv) is a fast, reliable Python package installer and resolver.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

#### Using `pip`

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## Usage

### Training a Model

Run the following command to train a model:

```bash
python main.py \
  <experiment_name> \
  -m <model_name> \
  -ep <epochs> \
  -lr <learning_rate> \
  -bs <batch_size>
```

#### Supported Models

- [Torchvision pre-trained models](https://pytorch.org/vision/main/models.html#classification):
  - ResNet: `ResNet18`, `ResNet34`, `ResNet50`, etc.
  - SwinTransformer: `Swin_V2_B`, `Swin_V2_S`, etc.

### Running Multiple Experiments

Use `exp.sh` to run a series of experiments with different augmentation strategies:

```bash
# Make script executable
chmod +x exp.sh

# Modify exp.sh to change model, epochs, learning rate, or batch size
# Run all experiments
./exp.sh
```

### Test on Inference

Evaluate the inference pipeline using randomly generated test data:

```bash
# Modify the variables in inference.py
python inference.py
```


## Submit to Kaggle

Submission instructions will be added soon.