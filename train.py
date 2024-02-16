
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from dataload import readfile, ImgDataset
from sklearn.model_selection import train_test_split
import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
# 定義簡單的卷積神經網路模型
from model import Classifier, MyModel


class GaussianBlur(object):
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        if torch.rand(1).item() < 0.2:  # 20% 的概率應用模糊
            img.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(0.1, 2.0)))
        return img


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 資料預處理
transform1 = transforms.Compose([
    transforms.Resize((62, 62)),
    GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.7)], p=0.2),
    # 隨機偏移
    transforms.RandomAffine(
        degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
])
transform2 = transforms.Compose([
    transforms.Resize((62, 62)),
    transforms.ToTensor(),
])
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if __name__ == '__main__':
    # 資料集路徑
    data_path = r'C:\python_project\yinyun_auto_play_randomdice\record/'
    # class_list = ['mimic', 'jocker', 'assassin',
    #               'summon', 'bubble', 'yinyun', 'sup', 'broken_growning', 'growning']
    # ['assassin1', 'assassin2', 'assassin3', 'assassin4', 'assassin5', 'assassin6', 'assassin7', 'background', 'broken_growning1', 'broken_growning2',
    #           'broken_growning3', 'broken_growning4', 'broken_growning5', 'broken_growning6', 'broken_growning7', 'bubble1', 'bubble2', 'bubble3', 'bubble4', 'bubble5', 'bubble6', 'bubble7', 'growning1', 'growning2', 'growning3', 'growning4', 'growning5', 'growning6', 'growning7', 'jocker1', 'jocker2', 'jocker3', 'jocker4', 'jocker5', 'jocker6', 'jocker7', 'mimic1', 'mimic2', 'mimic3', 'mimic4', 'mimic5', 'mimic6', 'mimic7', 'summon1', 'summon2', 'summon3', 'summon4', 'summon5', 'summon6', 'summon7', 'sup1', 'sup2', 'sup3', 'sup4', 'sup5', 'sup6', 'sup7', 'yinyun1', 'yinyun2', 'yinyun3', 'yinyun4', 'yinyun5', 'yinyun6', 'yinyun7']
    class_list = ['assassin', 'broken_growning', 'bubble',
                  'growning', 'jocker', 'mimic', 'summon', 'sup', 'yinyun']
    class_total_num = [1, 2, 3, 4, 5, 6, 7, 8]
    batch_size = 1024
    x, y1, y2 = readfile(data_path, class_list,
                         class_total_num, num_max=7000, background=True)
    # ImageFolder 資料載入
    train_x, test_x, train_y1, test_y1, train_y2, test_y2 = train_test_split(
        x, y1, y2, test_size=0.3, random_state=43)

    # 資料集分割成訓練集和測試集
    train_set = ImgDataset(train_x, train_y1, train_y2, transform1)
    val_set = ImgDataset(test_x, test_y1, test_y2, transform2)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    test_loader = DataLoader(val_set, batch_size=batch_size,
                             shuffle=False, num_workers=1)

    # 初始化模型、損失函數和優化器
    num_classes = len(class_list)+1
    total_num = len(class_total_num)
    model = MyModel(num_classes, total_num).cuda()
    criterion_attr1 = nn.CrossEntropyLoss()
    criterion_attr2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練模型
    num_epochs = 10
    for epoch in range(num_epochs):
        if epoch == 5:
            optimizer = optim.Adam(model.parameters(), lr=0.001/10)

        model.train()
        running_loss_attr1 = 0.0
        running_loss_attr2 = 0.0
        for inputs, (labels_attr1, labels_attr2) in train_loader:
            # 顯示圖片
            # for i in range(len(labels_attr1)):
            #     print(labels_attr1[i],labels_attr2[i])

            #     img = inputs[i].numpy().transpose(1, 2, 0)
            #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #     cv2.imshow('img',img)
            #     cv2.waitKey(0)
            optimizer.zero_grad()
            inputs, (labels_attr1, labels_attr2) = inputs.to(
                device), (labels_attr1.to(device), labels_attr2.to(device))
            output_attr1, output_attr2 = model(inputs)
            loss_attr1 = criterion_attr1(output_attr1, labels_attr1)
            loss_attr2 = criterion_attr2(output_attr2, labels_attr2)
            total_loss = loss_attr1 + loss_attr2
            total_loss.backward()
            optimizer.step()
            running_loss_attr1 += loss_attr1.item()
            running_loss_attr2 += loss_attr2.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss Attr1: {running_loss_attr1 / len(train_loader)}, Loss Attr2: {running_loss_attr2 / len(train_loader)}')
        # 儲存模型
        torch.save(model.state_dict(), '10MyModeltest{}.pth'.format(epoch+1))

        # 測試模型
        model.eval()
        correct_attr1, correct_attr2 = 0, 0
        total = 0
        with torch.no_grad():
            for inputs, (labels_attr1, labels_attr2) in test_loader:
                inputs, (labels_attr1, labels_attr2) = inputs.to(
                    device), (labels_attr1.to(device), labels_attr2.to(device))
                output_attr1, output_attr2 = model(inputs)
                _, predicted_attr1 = torch.max(output_attr1, 1)
                _, predicted_attr2 = torch.max(output_attr2, 1)
                total += labels_attr1.size(0)
                correct_attr1 += (predicted_attr1 == labels_attr1).sum().item()
                correct_attr2 += (predicted_attr2 == labels_attr2).sum().item()

        accuracy_attr1 = correct_attr1 / total
        accuracy_attr2 = correct_attr2 / total

        print(
            f'Accuracy on test set - Attribute 1: {accuracy_attr1 * 100:.2f}%')
        print(
            f'Accuracy on test set - Attribute 2: {accuracy_attr2 * 100:.2f}%')
