import torch
import torch.nn as nn
import torchvision.models as models

# class Classifier(nn.Module):
#     def __init__(self, num_classes=10):
#         super(Classifier, self).__init__()
#         # 換個更小的model
#         mobilenet = models.mobilenet_v3_large(weights=True)
#         self.features = mobilenet.features
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.classifier1 = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(960, num_classes)
#         )
#         self.classifier2 = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(960, 8)  # 7種點數+1種不是點數
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         output1 = self.classifier1(x)
#         output2 = self.classifier2(x)
#         return output1, output2
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

import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, num_classes_category, num_classes_status):
        super(MyModel, self).__init__()
        self.mobilenetv3_1 = models.mobilenet_v3_large(pretrained=True)
        num_features = self.mobilenetv3_1.classifier[0].in_features
        self.fc_category = nn.Linear(num_features, num_classes_category)
        
        self.mobilenetv3_2 = models.mobilenet_v3_large(pretrained=True)
        num_features2 = self.mobilenetv3_2.classifier[0].in_features  # 修改這裡，使用第二個網路的特徵數
        self.fc_status = nn.Linear(num_features2, num_classes_status)

    def forward(self, x):
        # 第一個網路
        x1 = self.mobilenetv3_1.features(x)
        x1 = F.adaptive_avg_pool2d(x1, (1, 1))
        x1 = x1.view(x1.size(0), -1)
        category_output = self.fc_category(x1)

        # 第二個網路
        x2 = self.mobilenetv3_2.features(x)
        x2 = F.adaptive_avg_pool2d(x2, (1, 1))
        x2 = x2.view(x2.size(0), -1)
        status_output = self.fc_status(x2)

        return category_output, status_output

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, class_num=6,dice_num=7):
        super(Classifier, self).__init__()

        # 第一個分支
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7)
        self.pool2_1 = nn.MaxPool2d(kernel_size=2)
        self.conv3_1 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3)
        self.pool3_1 = nn.MaxPool2d(kernel_size=2)
        self.fc1_1 = nn.Linear(9 * 4 * 4, 9 * 4 * 4*2)
        self.fc2_1 = nn.Linear(9 * 4 * 4*2, class_num)

        # 第二個分支
        self.conv1_2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7)
        self.pool2_2 = nn.MaxPool2d(kernel_size=2)
        self.conv3_2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3)
        self.pool3_2 = nn.MaxPool2d(kernel_size=2)
        self.fc1_2 = nn.Linear(9 * 4 * 4, 9 * 4 * 4*2)
        self.fc2_2 = nn.Linear(9 * 4 * 4*2, dice_num)

    def forward(self, x):
        # 第一個分支
        x1 = self.pool1_1(F.relu(self.conv1_1(x)))
        x1 = self.pool2_1(F.relu(self.conv2_1(x1)))
        x1 = self.pool3_1(F.relu(self.conv3_1(x1)))
        x1 = torch.flatten(x1, 1)
        x1 = F.relu(self.fc1_1(x1))
        output_state1 = self.fc2_1(x1)

        # 第二個分支
        x2 = self.pool1_2(F.relu(self.conv1_2(x)))
        x2 = self.pool2_2(F.relu(self.conv2_2(x2)))
        x2 = self.pool3_2(F.relu(self.conv3_2(x2)))
        x2 = torch.flatten(x2, 1)
        x2 = F.relu(self.fc1_2(x2))
        output_state2 = self.fc2_2(x2)

        return output_state1, output_state2


