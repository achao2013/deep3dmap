import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import torchvision.models as tvmodel
from ..builder import BACKBONES



@BACKBONES.register_module()
class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()

        self.featChannel = 512
        self.layer1 = tvmodel.vgg16_bn(pretrained=True).features
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1',  nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))),
            ('bn1',  nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(True)),
            ('pool1',  nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
       
            ('conv2',  nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))),
            ('bn2',  nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(True)),
            ('pool2',  nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
        
            ('conv3',  nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))),
            ('bn3',  nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(True)),
        
            ('conv4', nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))),
            ('bn4',  nn.BatchNorm2d(256)),
            ('relu4', nn.ReLU(True)),
            ('pool3',  nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
        
            ('conv5',  nn.Conv2d(256, 512, (3, 3), (1, 1), 1)),
            ('bn5',  nn.BatchNorm2d(512)),
            ('relu5', nn.ReLU(True)),
            ('pool4',  nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
        
            ('conv6',  nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)),
            ('bn6',  nn.BatchNorm2d(512)),
            ('relu6', nn.ReLU(True)),
        
            ('conv7',  nn.Conv2d(512, 512, (3, 3), (1, 1), 1)),
            ('bn7',  nn.BatchNorm2d(512)),
            ('relu7', nn.ReLU(True)),
            ('pool5',  nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)),
            ]))

        
            


    def forward(self, x):
        x=F.interpolate(x, size=[224,224])
        imga = x[:, 0:3, :, :]
        feata = self.layer1(imga)
        feata = F.avg_pool2d(feata, feata.size()[2:]).view(feata.size(0), feata.size(1))

        return feata


