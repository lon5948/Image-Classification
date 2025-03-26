import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    BATCH_SIZE,
    CLASS_ACCURACIES_PATH,
    CONFUSION_MATRIX_PATH,
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
from src.confusion_matrix import create_confusion_matrix, plot_class_accuracies
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
        choices=["train", "resume", "evaluate", "test"],
        help="Mode to run: train, resume, evaluate or test",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train when resuming (defaults to original setting)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate when resuming training (defaults to original setting)",
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
    confusion_matrix_path = os.path.join(output_dir, CONFUSION_MATRIX_PATH)
    class_accuracies_path = os.path.join(output_dir, CLASS_ACCURACIES_PATH)
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

        learning_rate = args.lr if args.lr else LEARNING_RATE

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
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

    elif args.mode == "evaluate":
        # Load the trained model
        model = create_model(NUM_CLASSES, RESNET_VERSION, USE_PRETRAINED, DEVICE)
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        # Get validation data loader
        _, val_loader = get_data_loaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE, NUM_WORKERS)

        # Generate and save confusion matrix
        print("Generating confusion matrix...")
        conf_matrix = create_confusion_matrix(
            model,
            val_loader,
            DEVICE,
            NUM_CLASSES,
            normalize=True,
            save_path=confusion_matrix_path,
        )

        # Plot and save per-class accuracies
        print("Generating per-class accuracy plot...")
        plot_class_accuracies(conf_matrix, save_path=class_accuracies_path)

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
