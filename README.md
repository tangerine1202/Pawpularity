# Pawpularity (Team 14)

This project predicts the "Pawpularity" score of pet images from the [PetFinder.my Pawpularity Contest](https://www.kaggle.com/c/petfinder-pawpularity-score) on Kaggle. The score represents how appealing a pet's photo is to potential adopters.

## Code Structure

- `main.py` - Main training script
- `exp.sh` - Helper script to run multiple training experiments
- `model/` - Neural network model implementations
- `utils/`
  - `data.py` - Dataset loading and preprocessing
  - `debug.py` - Logging and debugging utilities
- `inference.py` - Prediction script for submissions

## Setup

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast, reliable Python package installer and resolver.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

### Using pip

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
python main.py \
  <experiment_name>
  -m <model_name> \
  -ep <epochs> \
  -lr <learning_rate> \
  -bs <batch_size> \
```

#### Supported Models
- [Torchvision pre-trained models](https://pytorch.org/vision/main/models.html#classification)
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

This test evaluates the inference pipeline using randomly generated test data.

```bash
# Modify the variables in inference.py
python inference.py
```

## Submit to Kaggle

```
TBA
```