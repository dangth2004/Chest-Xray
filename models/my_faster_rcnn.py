import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers


class MyFasterRCNN(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=15):
        super(MyFasterRCNN, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.model = self._build_model()

    def forward(self, x):
        return self.model(x)

    def _build_model(self):
        return FasterRCNN(backbone=self._get_backbone(), num_classes=self.num_classes,
                          image_mean=[0.456], image_std=[0.224])

    @staticmethod
    def _get_backbone():
        # Using ResNet-50 as backbone
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # It's grayscale image so the first convolutional layer take in_channels=1
        backbone.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7),
                                         stride=(2, 2), padding=(3, 3), bias=False)

        with torch.no_grad():
            pretrained_weights = resnet.conv1.weight
            mean_pretrained_weights = pretrained_weights.mean(dim=1, keepdim=True)
            backbone.conv1.weight.copy_(mean_pretrained_weights)

        # Add FPN model for ResNet-50
        trainable_backbone_layers = _validate_trainable_layers(is_trained=True,
                                                               trainable_backbone_layers=5,
                                                               max_value=5, default_value=3)
        backbone = _resnet_fpn_extractor(backbone, trainable_layers=trainable_backbone_layers)

        return backbone
