import timm
import torch
import torch.nn as nn
import torchvision


class ModifiedResNet(nn.Module):
    def __init__(self, num_classes=100, pretrained=True, resnet_version=50):
        super().__init__()

        # Load the specified ResNet version with proper weights parameter
        if resnet_version == 18:
            self.base_model = timm.create_model("resnet18", pretrained=pretrained)
            feature_dim = 512
        elif resnet_version == 34:
            self.base_model = timm.create_model("resnet34", pretrained=pretrained)
            feature_dim = 512
        elif resnet_version == 50:
            self.base_model = timm.create_model("resnet50", pretrained=pretrained)
            feature_dim = 2048
        elif resnet_version == 101:
            self.base_model = torchvision.models.resnext101_32x8d(
                weights=torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
            )
            feature_dim = self.base_model.fc.in_features
        elif resnet_version == 152:
            self.base_model = timm.create_model("resnet152", pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")

        for param in self.base_model.parameters():
            param.requires_grad = True

        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.3),  # Start with dropout to prevent overfitting
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        for m in self.base_model.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.base_model(x)


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

    # Clear unused memory
    torch.cuda.empty_cache()

    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    assert num_params < 100000000, "Model size exceeds 100M parameters limit"

    return model
