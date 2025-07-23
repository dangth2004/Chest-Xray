import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers


def get_backbone(backbone='resnet50'):
    if backbone == 'resnet50':
        # Using ResNet-50 as backbone
        backbone = resnet50(weights=None)
        # It's grayscale image so the first convolutional layer take in_channels=1
        backbone.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7),
                                         stride=(2, 2), padding=(3, 3), bias=False)

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        state_dict = resnet.state_dict()
        del state_dict['conv1.weight']

        backbone.load_state_dict(state_dict, strict=False)

        # Add FPN model for ResNet-50
        trainable_backbone_layers = _validate_trainable_layers(is_trained=True,
                                                               trainable_backbone_layers=5,
                                                               max_value=5, default_value=3)
        backbone = _resnet_fpn_extractor(backbone, trainable_layers=trainable_backbone_layers)

        return backbone

    elif backbone == 'resnet101':
        # Using ResNet-101 as backbone
        backbone = resnet101(weights=None)
        # It's grayscale image so the first convolutional layer take in_channels=1
        backbone.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7),
                                         stride=(2, 2), padding=(3, 3), bias=False)

        resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        state_dict = resnet.state_dict()
        del state_dict['conv1.weight']

        backbone.load_state_dict(state_dict, strict=False)

        # Add FPN model for ResNet-101
        trainable_backbone_layers = _validate_trainable_layers(is_trained=True,
                                                               trainable_backbone_layers=5,
                                                               max_value=5, default_value=3)
        backbone = _resnet_fpn_extractor(backbone, trainable_layers=trainable_backbone_layers)

        return backbone
    else:
        raise NotImplementedError


def model_faster_rcnn(device, backbone='resnet50', num_classes=15):
    backbone = get_backbone(backbone)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes,
                       image_mean=[0.456], image_std=[0.224]).to(device)
    return model
