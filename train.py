
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from dataload import readfile, ImgDataset
from sklearn.model_selection import train_test_split

# 定義簡單的卷積神經網路模型
class MultiAttributeClassifier(nn.Module):
    def __init__(self, num_classes_attr1, num_classes_attr2):
        super(MultiAttributeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_attr1 = nn.Linear(64 * 31 * 31, num_classes_attr1)
        self.fc_attr2 = nn.Linear(64 * 31 * 31, num_classes_attr2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 64 * 31 * 31)
        output_attr1 = self.fc_attr1(x)
        output_attr2 = self.fc_attr2(x)
        return output_attr1, output_attr2

# 資料預處理
transform = transforms.Compose([
    transforms.Resize((62, 62)),
    transforms.ToTensor(),
])
if __name__ == '__main__':
    # 資料集路徑
    data_path =  r'C:\python_dev\yinyun_auto_play_randomdice\record\train_data/'
    class_list = ['mimic', 'jocker', 'assassin', 'summon', 'bubble']
    class_total_num = [1, 2, 3, 4, 5, 6, 7, 8]
    batch_size = 32
    x, y1, y2 = readfile(data_path, class_list,
                            class_total_num, background=True)
    # ImageFolder 資料載入
    train_x, test_x, train_y1, test_y1, train_y2, test_y2 = train_test_split(
            x, y1, y2, test_size=0.3, random_state=43)

    # 資料集分割成訓練集和測試集
    train_set = ImgDataset(train_x, train_y1, train_y2, transform)
    val_set = ImgDataset(test_x, test_y1, test_y2, transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)
    test_loader = DataLoader(val_set, batch_size=batch_size,
                                shuffle=False, num_workers=3)


    # 初始化模型、損失函數和優化器
    num_classes = len(class_list)+1
    total_num = len(class_total_num)
    model = MultiAttributeClassifier(num_classes,total_num)
    criterion_attr1 = nn.CrossEntropyLoss()
    criterion_attr2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練模型
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss_attr1 = 0.0
        running_loss_attr2 = 0.0
        for inputs, (labels_attr1, labels_attr2) in train_loader:
            optimizer.zero_grad()
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
        torch.save(model.state_dict(), 'test{}.pth'.format(epoch+1))

    # 測試模型
    model.eval()
    correct_attr1, correct_attr2 = 0, 0
    total = 0
    with torch.no_grad():
        for inputs, (labels_attr1, labels_attr2) in test_loader:
            output_attr1, output_attr2 = model(inputs)
            _, predicted_attr1 = torch.max(output_attr1, 1)
            _, predicted_attr2 = torch.max(output_attr2, 1)
            total += labels_attr1.size(0)
            correct_attr1 += (predicted_attr1 == labels_attr1).sum().item()
            correct_attr2 += (predicted_attr2 == labels_attr2).sum().item()

    accuracy_attr1 = correct_attr1 / total
    accuracy_attr2 = correct_attr2 / total

    print(f'Accuracy on test set - Attribute 1: {accuracy_attr1 * 100:.2f}%')
    print(f'Accuracy on test set - Attribute 2: {accuracy_attr2 * 100:.2f}%')