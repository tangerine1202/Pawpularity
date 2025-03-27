# Pawpularity Score Prediction

This project predicts the popularity score of pet photographs using deep learning models.


## Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Pawpularity.git
cd Pawpularity
```

### Prepare Data

1. Download the dataset from [Kaggle](https://www.kaggle.com/c/petfinder-pawpularity-score/data).
   ```bash
   kaggle competitions download -c petfinder-pawpularity-score
   ```
2. Unzip the dataset to get the `train` and `test` folders, and the `train.csv` and `test.csv` files.
3. Create a `data/` directory in the project root:
   ```bash
   mkdir data
   ```
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

### Training

```bash
# Train with a specific model
python main.py exp_name=my_experiment +model=resnet18

# Customize training parameters
python main.py exp_name=my_experiment +model=resnet18 training.lr=1e-5 training.epochs=20

# Use Wandb logging
python main.py exp_name=my_experiment wandb.use_wandb=true
```

### Inference

Local inference:
```bash
python inference.py
```

For Kaggle:
1. Set `IN_KAGGLE = True` in inference.py
2. Copy-paste the inference.py code into a Kaggle script
3. Upload necessary model files (i.e. `utils/data.py`, `model/baselines.py`) to Kaggle dataset named `kaggle-files`.
4. Run script in Kaggle.


## Configuration

This project uses Hydra for configuration management:

- `conf/config.yaml`: Base configuration
- `conf/model/resnet.yaml`: ResNet model configuration
- `conf/model/swin.yaml`: Swin Transformer configuration

### Project Structure

- `main.py` - Main training script
- `conf/` - Configuration files
- `data/` - Contains training and test data:
  - `train/` - Training images
  - `test/` - Test images
  - `train.csv` - Training data
  - `test.csv` - Test data
- `model/` - Neural network model implementations
- `utils/` - Utility scripts
- `inference.py` - Prediction script for submissions
- `exp.sh` - Helper script to run multiple training experiments
