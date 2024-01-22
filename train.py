import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
from sklearn.model_selection import train_test_split

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def readfile(path, label):
    ansX = []
    ansY1 = []  # 存儲種類
    ansY2 = []  # 存儲點數
    label_list = ['mimic', 'jocker', 'assassin',
                  'summon', 'bubble', 'background']
    # label_list = ['growning', 'yinyun', 'jocker', 'sup', 'broken_growning', 'background']#

    for label_name in label_list:
        for i in range(1, 8):
            print(label_name + str(i))

            if label_name == 'background':
                folder_name = label_name
            else:
                folder_name = label_name + str(i)
            image_dir = sorted(os.listdir(os.path.join(path, folder_name)))
            x = np.zeros((len(image_dir), 62, 62, 3), dtype=np.uint8)
            y1 = np.zeros((len(image_dir)), dtype=np.uint8)
            y2 = np.zeros((len(image_dir)), dtype=np.uint8)
            print(label_list.index(label_name))
            for j, file in enumerate(image_dir):
                img = cv2.imread(os.path.join(path, folder_name, file))
                x[j, :, :] = cv2.resize(img, (62, 62))
                ansX.append(x[j])

                if label:
                    if label_name == 'background':
                        y1[j] = 5
                    else:
                        y1[j] = label_list.index(label_name)
                    y2[j] = i
                    ansY1.append(y1[j])
                    ansY2.append(y2[j])
            if label_name == 'background':
                break

    if label:
        return ansX, ansY1, ansY2
    else:
        return ansX


class ImgDataset(Dataset):
    def __init__(self, x, y1=None, y2=None, transform=None):
        self.x = x
        self.y1 = y1
        self.y2 = y2

        if y1 is not None:
            self.y1 = torch.LongTensor(y1)

        if y2 is not None:
            self.y2 = torch.LongTensor(y2)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]

        if self.transform is not None:
            X = self.transform(X)

        if self.y1 is not None and self.y2 is not None:
            Y1 = self.y1[index]
            Y2 = self.y2[index]
            return X, Y1, Y2
        else:
            return X


if __name__ == '__main__':
    workspace_dir = r'C:\dice_test/'

    print("Reading data")
    train_x, train_y1, train_y2 = readfile(workspace_dir, True)
    print("Size of training data =", len(train_x))
    print("Size of training labels =", len(train_y1), len(train_y2))

    train_x, test_x, train_y1, test_y1, train_y2, test_y2 = train_test_split(
        train_x, train_y1, train_y2, test_size=0.3, random_state=43)
    print("Size of training data =", len(train_x))
    print("Size of testing data =", len(test_x))

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.2), scale=(0.9, 1.1), shear=0),
        # 加入噪聲
        # transforms.RandomApply([transforms.Lambda(dill.dumps(lambda x: x + torch.randn_like(x) * 0.1))], p=0.5),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    batch_size = 256

    train_set = ImgDataset(train_x, train_y1, train_y2, train_transform)
    val_set = ImgDataset(test_x, test_y1, test_y2, test_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=3)

    from model import Classifier

    model = Classifier().cuda()
    # model.load_state_dict(torch.load('test14.pth'))
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizerSGD = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epoch = 20

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        best_acc = 0.0

        model.train()

        for i, data in enumerate(train_loader):
            if i > 30:
                optimizer.zero_grad()
            else:
                optimizerSGD.zero_grad()

            train_pred1, train_pred2 = model(data[0].cuda())
            batch_loss1 = loss(train_pred1, data[1].cuda())
            batch_loss2 = loss(train_pred2, data[2].cuda())
            batch_loss = batch_loss1 + batch_loss2
            batch_loss.backward()

            if i > 30:
                optimizer.step()
            else:
                optimizerSGD.step()

            train_acc += np.sum(np.argmax(train_pred1.cpu().data.numpy(),
                                axis=1) == data[1].numpy())
            train_loss += batch_loss.item()

        model.eval()

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred1, val_pred2 = model(data[0].cuda())
                batch_loss1 = loss(val_pred1, data[1].cuda())
                batch_loss2 = loss(val_pred2, data[2].cuda())
                batch_loss = batch_loss1 + batch_loss2

                val_acc += np.sum(np.argmax(val_pred1.cpu().data.numpy(),
                                  axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' %
                  (epoch + 1, num_epoch, time.time() - epoch_start_time,
                   train_acc / train_set.__len__(), train_loss / train_set.__len__(),
                   val_acc / val_set.__len__(), val_loss / val_set.__len__()))

            torch.save(model.state_dict(), "Sasup{}.pth".format(epoch))
            best_acc = val_acc / val_set.__len__()