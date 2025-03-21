import csv

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import TestImageDataset


def predict_images(model, test_dir, output_file, device, batch_size=32, num_workers=4):
    """
    Generate predictions for images in the test directory and save to a CSV file.

    Args:
        model: Trained PyTorch model
        test_dir: Directory containing test images (without category folders)
        output_file: Path to save the prediction CSV
        device: Device to run predictions on
        batch_size: Batch size for predictions
        num_workers: Number of workers for data loading
    """

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = TestImageDataset(test_dir, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"Found {len(test_dataset)} test images")

    model.eval()

    predictions = []
    with torch.no_grad():
        for images, img_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for img_name, pred_label in zip(img_names, predicted.cpu().numpy()):
                predictions.append(
                    {
                        "image_name": img_name.split(".")[0],
                        "pred_label": int(pred_label),
                    }
                )

    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["image_name", "pred_label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for pred in predictions:
            writer.writerow(pred)

    print(f"Predictions saved to {output_file}")
    return output_file
