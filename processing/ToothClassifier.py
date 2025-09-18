# processing/ToothClassifier.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ToothClassifier(nn.Module):
    """
    ResNet18 برای طبقه‌بندی دوکلاسه (healthy / decayed).
    - ورودی باید 3 کاناله 224x224 باشد (برای تصاویر خاکستری، در ورودی 3 کاناله‌شان می‌کنیم).
    - اگر use_imagenet_weights=True باشد، وزن‌های ImageNet بارگذاری می‌شوند.
    - اگر freeze_backbone=True باشد، بدنه ResNet فریز می‌شود و فقط هد آموزش می‌بیند.
    - اگر dropout>0 باشد، بین backbone و لایه‌ی نهایی Dropout اعمال می‌شود.
    """
    def __init__(
        self,
        num_classes: int = 2,
        use_imagenet_weights: bool = True,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # بارگذاری backbone
        weights = ResNet18_Weights.DEFAULT if use_imagenet_weights else None
        backbone = resnet18(weights=weights)

        # فریز اختیاری بدنه
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        in_features = backbone.fc.in_features

        # Head نهایی
        head_layers = []
        if dropout and dropout > 0:
            head_layers.append(nn.Dropout(p=float(dropout)))
        head_layers.append(nn.Linear(in_features, num_classes))
        backbone.fc = nn.Sequential(*head_layers) if len(head_layers) > 1 else head_layers[0]

        # برای سازگاری با predict/Grad-CAM، کل مدل را زیر self.model نگه می‌داریم
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
