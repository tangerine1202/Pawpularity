import logging

import torch
from torchvision.models import resnet18, ResNet18_Weights

log = logging.getLogger(__name__)


class ResNet18Model(torch.nn.Module):
    def __init__(self, pretrained=True, freeze_pretrained=False):
        if freeze_pretrained and not pretrained:
            raise ValueError("Cannot freeze pretrained model if pretrained is False")

        super(ResNet18Model, self).__init__()
        self.weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.preprocess = self.weights.transforms() if self.weights else None
        self.model = resnet18(weights=self.weights, progress=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        self.model.fc.weight.data.fill_(0.0)
        self.model.fc.bias.data.fill_(0.0)

        if pretrained and freeze_pretrained:
            self.freeze()

    def forward(self, x):
        if self.preprocess:
            x = self.preprocess(x)
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc.requires_grad = True

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
