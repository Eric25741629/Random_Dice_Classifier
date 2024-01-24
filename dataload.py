from PIL import Image
import re
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ImgDataset(Dataset):
    def __init__(self, x, y1=None, y2=None, transform=None):
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.transform = transform
        if y1 is not None:
            self.y1 = torch.LongTensor(y1)
        if y2 is not None:
            self.y2 = torch.LongTensor(y2)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]

        if self.transform is not None:
            X = self.transform(X)

        if self.y1 is not None and self.y2 is not None:
            Y1 = self.y1[index]
            Y2 = self.y2[index]
            return X, (Y1, Y2)  # 返回一個tuple，包含Y1和Y2



# 讀取資料夾下的所有圖片檔


def read_files_path(path):
    image_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                image_files.append(file_path)
    return image_files


def readfile(path, class_list, class_total_num, background=False):
    data_path = []
    x = []
    y1 = []
    y2 = []
    for i in class_list:
        for num in class_total_num:
            data_path.append(path + i + str(num))
    if background:
        data_path.append(path + 'background')
    for i in data_path:
        class_ = os.path.basename(i)
        if class_ != 'background':
            class_part = re.search(r'[a-z]+', class_).group()
            class_index = class_list.index(class_part)
            number_part = int(re.search(r'\d+', class_).group())
        else:
            number_part = 0
            class_index = len(class_list)
        img_path_list = read_files_path(i)
        for j in img_path_list:
            #使用PIL讀取圖片
            img = Image.open(j)
            x.append(img)
            y1.append(class_index)
            y2.append(number_part)
    return x, y1, y2


if __name__ == '__main__':
    workspace_dir = r'C:\python_dev\yinyun_auto_play_randomdice\record\train_data/'
    class_list = ['mimic', 'jocker', 'assassin', 'summon', 'bubble']
    class_total_num = [1, 2, 3, 4, 5, 6, 7]
    x, y1, y2 = readfile(workspace_dir, class_list,
                         class_total_num, background=True)
    print(y1, y2)
    print(len(x), len(y1), len(y2))
