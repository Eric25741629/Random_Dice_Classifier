import torch
import torch.nn as nn
import torchvision.models as models

class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        # 換個更小的model
        mobilenet = models.mobilenet_v3_large(weights=True)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(960, 6)
        )
        self.classifier2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(960, 8)  # 7種點數+1種不是點數
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output1 = self.classifier1(x)
        output2 = self.classifier2(x)
        return output1, output2
# class Classifier(nn.Module):
#     def __init__(self, num_classes=64):
#         super(Classifier, self).__init__()
#         #換個更小的model
        
#         mobilenet = models.mobilenet_v3_large(weights=True)
#         self.features = mobilenet.features
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(960, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


# def MobileNet(num_classes=36):
#     return MobileNetClassifier(num_classes)


# class Classifier(nn.Module):
#     def __init__(self, num_classes=36):
#         super(Classifier, self).__init__()
#         vgg = models.vgg16(pretrained=True)
#         self.features = vgg.features
#         self.avgpool = vgg.avgpool
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

