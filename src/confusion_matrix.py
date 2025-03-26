import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def create_confusion_matrix(
    model, val_loader, device, num_classes=100, normalize=True, save_path=None
):
    """
    Visualize a confusion matrix for the validation set.

    Args:
        model: The trained PyTorch model
        val_loader: DataLoader for validation data
        device: Device to run the model on
        num_classes: Number of classes in the dataset
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the confusion matrix plot

    Returns:
        conf_matrix: The confusion matrix
    """
    # Put model in evaluation mode
    model.eval()

    # Lists to store all predictions and true labels
    all_preds = []
    all_labels = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Gather predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    # Normalize if requested
    if normalize:
        conf_matrix = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        )
        conf_matrix = np.nan_to_num(conf_matrix)  # Replace NaN with 0

    # Plot the confusion matrix
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        conf_matrix,
        annot=False,  # Turn off annotations for better readability with many classes
        cmap="Blues",
        square=True,
        fmt=".2f" if normalize else "d",
        cbar=True,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

    return conf_matrix


def plot_class_accuracies(conf_matrix, save_path=None):
    """
    Plot per-class accuracies from the confusion matrix.

    Args:
        conf_matrix: The confusion matrix (should be normalized)
        save_path: Path to save the class accuracies plot
    """
    # Get diagonal elements (true positives)
    class_accuracies = np.diag(conf_matrix)

    # Plot class accuracies
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(class_accuracies)), class_accuracies)
    plt.xlabel("Class Index")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(range(0, len(class_accuracies), 5))
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path)
        print(f"Class accuracies plot saved to {save_path}")

    return class_accuracies


def plot_top_k_confused_classes(conf_matrix, k=10, save_path=None):
    """
    Plot the top k most confused class pairs.

    Args:
        conf_matrix: The confusion matrix (should be normalized)
        k: Number of most confused pairs to show
        save_path: Path to save the plot
    """
    # Create a copy of the confusion matrix to avoid modifying the original
    cm_copy = conf_matrix.copy()

    # Set diagonal elements to 0 since we're interested in misclassifications
    np.fill_diagonal(cm_copy, 0)

    # Find the top k confused pairs
    confused_pairs = []
    for i in range(k):
        # Find the indices of the maximum value in the matrix
        true_label, pred_label = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
        confusion_value = cm_copy[true_label, pred_label]
        confused_pairs.append((true_label, pred_label, confusion_value))

        # Set this pair to 0 to find the next highest
        cm_copy[true_label, pred_label] = 0

    # Plot the confused pairs
    plt.figure(figsize=(12, 8))

    # Extract data for plotting
    true_labels = [pair[0] for pair in confused_pairs]
    pred_labels = [pair[1] for pair in confused_pairs]
    confusion_values = [pair[2] for pair in confused_pairs]

    # Create labels for the x-axis
    x_labels = [
        f"{true_l} → {pred_l}" for true_l, pred_l in zip(true_labels, pred_labels)
    ]

    # Plot the bars
    plt.bar(range(len(confusion_values)), confusion_values)
    plt.xlabel("True → Predicted Class")
    plt.ylabel("Confusion Rate")
    plt.title(f"Top {k} Most Confused Class Pairs")
    plt.xticks(range(len(confusion_values)), x_labels, rotation=45, ha="right")
    plt.ylim(0, max(confusion_values) * 1.1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Top confused classes plot saved to {save_path}")

    return confused_pairs


def evaluate_model(model, val_loader, device, num_classes=100, output_dir=None):
    """
    Comprehensive evaluation of the model.

    Args:
        model: The trained PyTorch model
        val_loader: DataLoader for validation data
        device: Device to run the model on
        num_classes: Number of classes in the dataset
        output_dir: Directory to save evaluation results
    """
    # Create paths for saving results
    if output_dir:
        import os

        os.makedirs(output_dir, exist_ok=True)
        confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
        class_accuracies_path = os.path.join(output_dir, "class_accuracies.png")
        top_confused_path = os.path.join(output_dir, "top_confused_classes.png")
    else:
        confusion_matrix_path = None
        class_accuracies_path = None
        top_confused_path = None

    # Generate confusion matrix
    print("Generating confusion matrix...")
    conf_matrix = create_confusion_matrix(
        model,
        val_loader,
        device,
        num_classes,
        normalize=True,
        save_path=confusion_matrix_path,
    )

    # Plot class accuracies
    print("Analyzing per-class accuracies...")
    class_accuracies = plot_class_accuracies(
        conf_matrix, save_path=class_accuracies_path
    )

    # Find and plot the most confused class pairs
    print("Finding most confused class pairs...")
    top_confused = plot_top_k_confused_classes(
        conf_matrix, k=10, save_path=top_confused_path
    )

    # Calculate overall accuracy
    overall_accuracy = (
        np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
        if not np.isclose(np.sum(conf_matrix), 0)
        else 0
    )

    # Print summary statistics
    print("\nEvaluation Summary:")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(
        f"Best performing class: {np.argmax(class_accuracies)}"
        "(Accuracy: {np.max(class_accuracies):.4f})"
    )
    print(
        f"Worst performing class: {np.argmin(class_accuracies)}"
        "(Accuracy: {np.min(class_accuracies):.4f})"
    )

    # Print top confused pairs
    print("\nTop confused class pairs (True → Predicted):")
    for i, (true_label, pred_label, confusion_value) in enumerate(top_confused, 1):
        print(f"{i}. Class {true_label} → Class {pred_label}: {confusion_value:.4f}")

    return conf_matrix, class_accuracies, top_confused
