# 繪製混淆矩陣
from torchvision import transforms
from dataload import readfile, ImgDataset
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from dataload import readfile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import cv2
from model import MyModel
if __name__ == '__main__':
    mod = MyModel(10, 8)
    mod.load_state_dict(torch.load(r'10MyModeltest10.pth'))
    mod.eval().cuda()
    data_path = r'C:\python_project\yinyun_auto_play_randomdice\record/'
    class_list = ['assassin', 'broken_growning', 'bubble',
                  'growning', 'jocker', 'mimic', 'summon', 'sup', 'yinyun']

    class_total_num = [1, 2, 3, 4, 5, 6, 7, 8]
    batch_size = 1024
    x, y1, y2 = readfile(data_path, class_list,
                         class_total_num, num_max=70, background=True)
    # ImageFolder 資料載入
    train_x, test_x, train_y1, test_y1, train_y2, test_y2 = train_test_split(
        x, y1, y2, test_size=0.5, random_state=43)

    transform2 = transforms.Compose([
        transforms.Resize((62, 62)),
        transforms.ToTensor(),
    ])
    # 資料集分割成訓練集和測試集

    val_set = ImgDataset(test_x, test_y1, test_y2, transform2)

    test_loader = DataLoader(val_set, batch_size=batch_size,
                             shuffle=False, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mod.to(device)
    y_true_attr1 = []
    y_pred_attr1 = []
    y_true_attr2 = []
    y_pred_attr2 = []
    with torch.no_grad():
        for inputs, (labels_attr1, labels_attr2) in test_loader:
            inputs, (labels_attr1, labels_attr2) = inputs.to(
                device), (labels_attr1.to(device), labels_attr2.to(device))
            output_attr1, output_attr2 = model(inputs)
            _, predicted_attr1 = torch.max(output_attr1, 1)
            _, predicted_attr2 = torch.max(output_attr2, 1)
            y_true_attr1.extend(labels_attr1.cpu().numpy())
            y_pred_attr1.extend(predicted_attr1.cpu().numpy())
            y_true_attr2.extend(labels_attr2.cpu().numpy())
            y_pred_attr2.extend(predicted_attr2.cpu().numpy())
    cm_attr1 = confusion_matrix(y_true_attr1, y_pred_attr1)
    cm_attr2 = confusion_matrix(y_true_attr2, y_pred_attr2)
    # 分成兩張圖
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_attr1, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix Attr1')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_attr2, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix Attr2')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    # 將預測錯誤的圖片顯示出來
    for i in range(len(y_true_attr1)):
        if y_true_attr1[i] != y_pred_attr1[i]:
            # Convert PIL Image to NumPy array
            img = np.array(test_x[i].convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            # print(y_true_attr1[i], y_pred_attr1[i])
            print(class_list[y_true_attr1[i]], class_list[y_pred_attr1[i]])
    for i in range(len(y_true_attr2)):
        if y_true_attr2[i] != y_pred_attr2[i]:
            # Convert PIL Image to NumPy array
            img = np.array(test_x[i].convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            print(y_true_attr2[i],
                  y_pred_attr2[i])
            # 印出圖片檔名

            cv2.imshow('img', img)
            cv2.waitKey(0)
            # print(y_true_attr2[i], y_pred_attr2[i])
