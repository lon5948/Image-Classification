import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class BiologicalImageDataset(Dataset):
    """
    Dataset class for loading biological images from a directory structure
    where each class has its own folder, named 0-99.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Get class directories and make sure we sort numerically (0, 1, 2,
        # ...)
        class_dirs = [
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        ]
        class_dirs = sorted(class_dirs, key=lambda x: int(x))

        # Create class to index mapping - in this case, the folder name itself
        # is the class index
        self.class_to_idx = {class_name: int(class_name) for class_name in class_dirs}
        self.idx_to_class = {int(class_name): class_name for class_name in class_dirs}
        self.class_names = {
            int(class_name): f"Category {class_name}" for class_name in class_dirs
        }

        # Collect image paths and labels
        for class_dir in class_dirs:
            class_idx = int(class_dir)
            class_path = os.path.join(data_dir, class_dir)

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class TestImageDataset(Dataset):
    """
    Dataset class for loading test images that don't have category folders.
    Images are directly in the test folder.
    """

    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = []

        for filename in os.listdir(test_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_files.append(filename)

        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name


def get_data_loaders(train_dir, val_dir, batch_size, num_workers):
    """
    Create and return data loaders for training, validation, and testing.

    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        test_dir: Directory containing test data
        batch_size: Batch size for the data loaders
        num_workers: Number of workers for the data loaders

    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        class_names: List of class names
    """

    train_dataset = BiologicalImageDataset(train_dir, transform=train_transform)
    val_dataset = BiologicalImageDataset(val_dir, transform=test_transform)

    print(
        f"Found {len(train_dataset)} training images "
        f"across {len(train_dataset.class_to_idx)} classes"
    )
    print(f"Found {len(val_dataset)} validation images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
