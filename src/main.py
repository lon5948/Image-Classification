import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    RANDOM_SEED,
    RESNET_VERSION,
    TEST_DIR,
    TRAIN_CURVES_PATH,
    TRAIN_DIR,
    USE_PRETRAINED,
    VAL_DIR,
    WEIGHT_DECAY,
)
from src.dataset import get_data_loaders
from src.model import create_model
from src.test import predict_images
from src.train import train_model
from src.utils import set_seed, setup_directories
from src.visualization import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Biological Image Classification")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "resume", "test"],
        help="Mode to run: train, resume or test",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train when resuming (defaults to original setting)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate when resuming training "
        "(defaults to original setting * 0.1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="prediction.csv",
        help="Output file for predictions (CSV)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(RANDOM_SEED)

    output_dir = setup_directories()
    model_path = os.path.join(output_dir, MODEL_SAVE_PATH)
    curves_path = os.path.join(output_dir, TRAIN_CURVES_PATH)
    prediction_path = os.path.join(output_dir, args.output)

    print(f"Using device: {DEVICE}")

    if args.mode == "train" or args.mode == "resume":
        train_loader, val_loader = get_data_loaders(
            TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS
        )

        model = create_model(NUM_CLASSES, RESNET_VERSION, USE_PRETRAINED, DEVICE)

        if args.mode == "resume":
            print(f"Loading model from {model_path} to resume training")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        epochs_to_train = args.epochs if args.epochs else NUM_EPOCHS

        if args.mode == "resume":
            if args.learning_rate:
                learning_rate = args.learning_rate
            else:
                learning_rate = LEARNING_RATE * 0.1
            print(f"Resuming with learning rate: {learning_rate}")
        else:
            learning_rate = LEARNING_RATE

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
        )

        # For resume mode, use cosine annealing scheduler which works better
        # for fine-tuning
        if args.mode == "resume":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs_to_train, eta_min=learning_rate / 10
            )
        else:
            # Keep the original scheduler for new training
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=3
            )

        print("Starting training...")
        history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            DEVICE,
            epochs_to_train,
            model_path,
        )

        plot_training_curves(history, curves_path)

    elif args.mode == "test":
        model = create_model(NUM_CLASSES, RESNET_VERSION, USE_PRETRAINED, DEVICE)
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        print("Generating predictions...")
        predict_images(
            model, TEST_DIR, prediction_path, DEVICE, BATCH_SIZE, NUM_WORKERS
        )


if __name__ == "__main__":
    main()
