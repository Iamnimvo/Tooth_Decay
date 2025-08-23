import torch
import torch.nn as nn
from torchvision import models

class ToothClassifier(nn.Module):
    def __init__(self):
        super(ToothClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 کلاس: پوسیدگی یا عدم پوسیدگی

    def forward(self, x):
        return self.model(x)