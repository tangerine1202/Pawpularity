import logging

import torch
from torchvision.models import *

log = logging.getLogger(__name__)


class TorchVisionModel(torch.nn.Module):
    def __init__(self, name, weights=None, pretrained=True, freeze_pretrained=True, **model_kwargs):
        if pretrained and weights is None:
            raise ValueError("Cannot use pretrained without weights")
        if freeze_pretrained and not pretrained:
            raise ValueError("Cannot freeze pretrained model if pretrained is False")

        super(TorchVisionModel, self).__init__()
        self.weights = eval(weights) if weights and pretrained else None
        self.model = eval(name.lower())(weights=weights, progress=True, **model_kwargs)
        self.preprocess = self.weights.transforms() if self.weights else None

        self.head_layer = self.reset_head_weights()

        if pretrained and freeze_pretrained:
            self.freeze_pretrained()

    def reset_head_weights(self, output_dim=1) -> torch.nn.Parameter:
        """Reset the weights of the model head. This method must be implemented by subclasses."""
        raise NotImplementedError(
            "reset_head_weights() must be implemented in subclasses of TorchVisionModel"
        )

    def forward(self, x):
        if self.preprocess:
            x = self.preprocess(x)
        return self.model(x)

    def freeze_pretrained(self):
        for param in self.model.parameters():
            param.requires_grad = False
        if hasattr(self, 'head_layer'):
            if isinstance(self.head_layer, torch.nn.Module):
                for param in self.head_layer.parameters():
                    param.requires_grad = True
            else:
                self.head_layer.requires_grad = True
        else:
            raise ValueError("head_layer is not defined")

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


class ResNetModel(TorchVisionModel):
    def __init__(self, name, weights=None, pretrained=True, freeze_pretrained=True, **model_kwargs):
        assert 'resnet' in name.lower(), f'Model name must contain "resnet" for ResNetModel, Got: {name}'
        super(ResNetModel, self).__init__(
            name,
            weights=weights,
            pretrained=pretrained,
            freeze_pretrained=freeze_pretrained,
            **model_kwargs
        )

    def reset_head_weights(self, output_dim=1) -> torch.nn.Parameter:
        """Reset the weights of the model head."""
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, output_dim)
        self.model.fc.weight.data.fill_(0.0)
        self.model.fc.bias.data.fill_(0.0)
        return self.model.fc


class SwinTransformerModel(TorchVisionModel):
    def __init__(self, name, weights=None, pretrained=True, freeze_pretrained=True, **model_kwargs):
        assert 'swin' in name.lower(), f'Model name must contain "swin", Got: {name}'
        super(SwinTransformerModel, self).__init__(
            name,
            weights=weights,
            pretrained=pretrained,
            freeze_pretrained=freeze_pretrained,
            **model_kwargs
        )

    def reset_head_weights(self, output_dim=1) -> torch.nn.Parameter:
        """Reset the weights of the model head."""
        self.model.head = torch.nn.Linear(self.model.head.in_features, output_dim)
        self.model.head.weight.data.fill_(0.0)
        self.model.head.bias.data.fill_(0.0)
        return self.model.head
