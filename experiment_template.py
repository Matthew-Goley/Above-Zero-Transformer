# This is the main file
# Run this file to run the project
# Use this file to change values

from train import run_experiment

MODEL_CONFIG = {
    "feature_dim": 6,
    "d_model": 32,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.2
}

TRAINING_CONFIG = {
    "window": 24,
    "seq_len": 48,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 0
}

if __name__ == "__main__":
    run_experiment(MODEL_CONFIG, TRAINING_CONFIG)