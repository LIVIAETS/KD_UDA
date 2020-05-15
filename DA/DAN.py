import torch
from torchvision import models
import torch.nn as nn
import DA.mmd

class DANNetVGG16(nn.Module):
    def __init__(self):
        super(DANNetVGG16, self).__init__()
        model = models.vgg16(pretrained=True)  #False

        self.features = model.features


        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.cls_fc = nn.Linear(4096, 31)

    def forward(self, source, target):
        loss = 0
        source = self.features(source)
        source = source.view(source.size(0), -1)
        source = self.classifier(source)
        if self.training == True:
            target = self.features(target)
            target = target.view(target.size(0), -1)
            target = self.classifier(target)
            loss += DA.mmd.mmd_rbf_noaccelerate(source, target)
        source = self.cls_fc(source)
        return source, loss

    def s_forward(self, source):
        source = self.features(source)
        source = source.view(source.size(0), -1)
        source = self.classifier(source)
        source = self.cls_fc(source)
        return source



class DANNetResNet50(nn.Module):
    def __init__(self):
        super(DANNetResNet50, self).__init__()
        model = models.resnet50(pretrained=True)  #False

        self.features = model.features
        self.classifier = nn.Linear(2048, 31)

    def forward(self, source, target):
        loss = 0
        source = self.features(source)
        s_features = source.view(source.size(0), -1)
        if self.training == True:
            target = self.features(target)
            t_features = target.view(target.size(0), -1)
            loss += DA.mmd.mmd_rbf_noaccelerate(s_features, t_features)
        source = self.classifier(s_features)
        return source, loss