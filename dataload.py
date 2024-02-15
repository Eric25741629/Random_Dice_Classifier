from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import re
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random


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


def read_files_path(path):
    image_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                image_files.append(file_path)
    return image_files


def readfile_worker(params):
    i, class_list, num_max, background = params
    x = []
    y1 = []
    y2 = []

    class_ = os.path.basename(i)

    match = re.match(r'([a-zA-Z_]+)(\d+)', class_)

    if match:
        class_index = class_list.index(match.group(1))
        number_part = int(match.group(2))
    # else:
    #     raise ValueError('Class name should contain a number')

    if class_ == 'background':
        number_part = 0
        class_index = len(class_list)
    print(class_, class_index, number_part)
    data_num = 0
    img_path_list = read_files_path(i)

    if len(img_path_list) == 0:
        return x, y1, y2

    random.shuffle(img_path_list)
    random_float = num_max / len(img_path_list)
    if random_float > 1:
        random_float = 1
    if random_float < 0.2:
        random_float = 0.2
    print(random_float)

    # 提前分配空間
    x = [Image.open(j).copy() for j in img_path_list[:num_max]]
    y1 = [class_index] * len(x)
    y2 = [number_part] * len(x)

    return x, y1, y2


def readfile(path, class_list, class_total_num, num_max=800, background=False):
    data_path = []

    for i in class_list:
        for num in class_total_num:
            data_path.append(path + str(i) + str(num))

    if background:
        data_path.append(path + 'background')

    params_list = [(i, class_list, num_max, background) for i in data_path]

    x = []
    y1 = []
    y2 = []

    with ThreadPoolExecutor(max_workers=100) as executor:
        results = list(executor.map(readfile_worker, params_list))

    for result in results:
        x.extend(result[0])
        y1.extend(result[1])
        y2.extend(result[2])

    return x, y1, y2


if __name__ == '__main__':
    workspace_dir = r'C:\python_project\yinyun_auto_play_randomdice\record/'
    class_list = ['mimic', 'jocker', 'assassin', 'summon', 'bubble']
    class_total_num = [1, 2, 3, 4, 5, 6, 7]
    x, y1, y2 = readfile(workspace_dir, class_list,
                         class_total_num, background=True)
    print(y1, y2)
    print(len(x), len(y1), len(y2))
