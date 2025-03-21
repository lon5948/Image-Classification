import torch.nn as nn
from torchvision import models


class ModifiedResNet(nn.Module):
    def __init__(self, num_classes=100, pretrained=True, resnet_version=50):
        super().__init__()

        # Load the specified ResNet version with proper weights parameter
        if resnet_version == 18:
            if pretrained:
                weights = models.ResNet18_Weights.DEFAULT
            else:
                weights = None
            base_model = models.resnet18(weights=weights)
            feature_dim = 512
        elif resnet_version == 34:
            if pretrained:
                weights = models.ResNet34_Weights.DEFAULT
            else:
                weights = None
            base_model = models.resnet34(weights=weights)
            feature_dim = 512
        elif resnet_version == 50:
            if pretrained:
                weights = models.ResNet50_Weights.DEFAULT
            else:
                weights = None
            base_model = models.resnet50(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")

        # Use all layers except the last one (avg pool and fc)
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        # Keep the same structure as the original attention module for
        # compatibility
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 8, feature_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Improved classifier with same structure but better dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

        # Initialize classifier weights properly
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        att = self.attention(x)
        x = x * att
        x = self.avg_pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(num_classes, resnet_version, pretrained, device):
    """
    Create and initialize the model.

    Args:
        num_classes: Number of classes for classification
        resnet_version: ResNet version (18, 34, or 50)
        pretrained: Whether to use pretrained weights
        device: Device to move the model to

    Returns:
        Initialized model
    """
    model = ModifiedResNet(
        num_classes=num_classes,
        pretrained=pretrained,
        resnet_version=resnet_version,
    ).to(device)

    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    assert num_params < 100000000, "Model size exceeds 100M parameters limit"

    return model
