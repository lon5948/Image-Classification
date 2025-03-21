import torch

# Data paths
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

# Model parameters
NUM_CLASSES = 100
RESNET_VERSION = 50  # Options: 18, 34, 50
USE_PRETRAINED = True

# Training parameters
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Other settings
RANDOM_SEED = 42
NUM_WORKERS = 4

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Output paths
MODEL_SAVE_PATH = "best_model.pth"
TRAIN_CURVES_PATH = "training_curves.png"
