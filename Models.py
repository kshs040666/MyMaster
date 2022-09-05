import torch
import torchvision
from torch.nn import Module
from torch.nn.modules.container import Sequential

class FasterRCNN_Backbone_Resnet101(Module):
    def __init__(self) -> None:
        super().__init__()
        resnet101 = torchvision.models.resnet101()
        self.resnet = Sequential(resnet101.conv1, 
                                resnet101.bn1, 
                                resnet101.relu, 
                                resnet101.maxpool, 
                                resnet101.layer1, 
                                resnet101.layer2, 
                                resnet101.layer3,
                                resnet101.layer4)
        self.out_channels = 2048
    def forward(self, x):
        return self.resnet(x)

class FasterRCNN_Backbone_Googlenet(Module):
    def __init__(self) -> None:
        super().__init__()
        google = torchvision.models.googlenet()
        self.googlenet = Sequential(google.conv1, 
                                    google.maxpool1, 
                                    google.conv2, 
                                    google.conv3, 
                                    google.maxpool2, 
                                    google.inception3a, 
                                    google.inception3b, 
                                    google.maxpool3, 
                                    google.inception4a, 
                                    google.inception4b, 
                                    google.inception4c, 
                                    google.inception4d, 
                                    google.inception4e, 
                                    google.maxpool4, 
                                    google.inception5a, 
                                    google.inception5b
        )
        self.out_channels = 1024
    def forward(self, x):
        return self.googlenet(x)

class FasterRCNN_Backbone_Alexnet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.alexnet = torchvision.models.alexnet().features
        self.out_channels = 256
    def forward(self, x):
        return self.alexnet(x)

class FasterRCNN_Backbone_VGG16(Module):
    def __init__(self) -> None:
        super().__init__()
        self.vgg16 = torchvision.models.vgg16().features
        self.out_channels = 512
    def forward(self, x):
        return self.vgg16(x)

