import torch
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights


class MyResNet(torch.nn.Module):
    def __init__(self, num_classes=2, backbone='resnet50'):
        super(MyResNet, self).__init__()
        if backbone == 'resnet50':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.model = resnet101(weights=ResNet101_Weights.DEFAULT)
            resnet = resnet101(weights=ResNet101_Weights.DEFAULT)

        # It's grayscale image so the first convolutional layer take in_channels=1
        self.model.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7),
                                           stride=(2, 2), padding=(3, 3), bias=False)

        # Edit the weight of the first convolution layer
        with torch.no_grad():
            pretrained_weights = resnet.conv1.weight
            mean_pretrained_weight = pretrained_weights.mean(dim=1, keepdim=True)
            self.model.conv1.weight.copy_(mean_pretrained_weight)

        # Edit the last fully connected layer to predict 2 classes
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        return self.model(x)
