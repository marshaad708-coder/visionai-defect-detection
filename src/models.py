"""Model definitions for VisionAI Track A (supervised classification).

Two options, selected via config:
  - "compact_cnn": small CNN trained from scratch (~0.9M params). Fast on CPU,
    designed for the 200x200 grayscale NEU images.
  - "resnet18": ImageNet-pretrained ResNet-18 with a replaced classification
    head, fine-tuned on the defect data (grayscale repeated to 3 channels).
"""
import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CompactCNN(nn.Module):
    """~0.9M-parameter CNN for 224x224 grayscale defect images.

    4 conv blocks (32/64/128/256 ch) -> global average pooling -> MLP head.
    The last convolutional layer is named `last_conv` so Grad-CAM can target it.
    """

    def __init__(self, num_classes: int = 6, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(in_channels, 32),    # 224 -> 112
            _ConvBlock(32, 64),             # 112 -> 56
            _ConvBlock(64, 128),            # 56 -> 28
            _ConvBlock(128, 256),           # 28 -> 14
        )
        # Expose the final conv layer for Grad-CAM.
        self.last_conv = self.features[-1].block[-4]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_resnet18(num_classes: int = 6, pretrained: bool = True,
                   freeze_backbone: bool = False) -> nn.Module:
    """ResNet-18 with a fresh classification head; exposes layer4 for Grad-CAM."""
    from torchvision.models import resnet18, ResNet18_Weights

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # Alias used by the Grad-CAM helper.
    model.last_conv = model.layer4[-1].conv2
    return model


def get_model(name: str, num_classes: int = 6, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "compact_cnn":
        return CompactCNN(num_classes=num_classes, **kwargs)
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model '{name}'. Choose 'compact_cnn' or 'resnet18'.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
