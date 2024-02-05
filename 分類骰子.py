#分類.py
import os
import cv2
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
from model import Classifier,MyModel

def readfiles_recursive(path):
    image_files = []
    
    # 遍历当前目录下的所有文件和文件夹
    for root, dirs, files in os.walk(path):
        for file in files:
            # 判断文件是否以 .jpg 结尾
            if file.endswith('.jpg'):
                # 获取文件的绝对路径，并添加到列表
                file_path = os.path.join(root, file)
                image_files.append(file_path)
    
    return image_files
import torch.nn.functional as F
transform = transforms.Compose([
    transforms.Resize((62, 62)),
    # GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
])
def predict(img, model):
    # 判斷是否有gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 將模型移到 GPU 上
    # 使用gpu
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    output_attr1, output_attr2 = model(img)
    prediction_attr1 = torch.max(output_attr1, 1)[1].cpu().numpy()
    prediction_attr2 = torch.max(output_attr2, 1)[1].cpu().numpy()

    # 當機率小於0.99時，判斷 回傳None
    if output_attr1[0][prediction_attr1[0]] < 0.99 or output_attr2[0][prediction_attr2[0]] < 0.99:
        return None

    return [prediction_attr1[0], prediction_attr2[0]]



def move(imgpath,newpath,predict_label):
    
    # 檢查資料夾是否存在 = os.path.exists(newpath)
    if not os.path.exists(newpath+'/'+predict_label):
        os.makedirs(newpath+'/'+predict_label)
    # 移動檔案
    # print(imgpath, os.path.join(newpath+'/'+predict_label, os.path.basename(imgpath)))
    try:
        os.rename(imgpath, os.path.join(newpath+'/'+predict_label, os.path.basename(imgpath)))
    except:
        print('error',imgpath)
from tqdm import trange

def predict_and_move_with_progress(path, model, newpath, batch_size=3000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image_dir = readfiles_recursive(path)
    num_images = len(image_dir)

    with trange(0, num_images, batch_size, desc='Predicting') as t:
        for i in t:
            images = []
            paths = []
            count = 0

            for j in range(i, min(i + batch_size, num_images)):
                img = Image.open(image_dir[j])
                img = img.resize((62, 62))
                img_tensor =  transform(img).unsqueeze(0).to(device)
                images.append(img_tensor)
                paths.append(image_dir[j])
                count += 1

            with torch.no_grad():
                batch_images = torch.cat(images, dim=0)
                output_attr1, output_attr2 = model(batch_images)

                for j in range(count):
                    prediction_attr1 = torch.max(output_attr1[j], 0)[1].cpu().item()
                    prediction_attr2 = torch.max(output_attr2[j], 0)[1].cpu().item()

                    if (
                        output_attr1[j][prediction_attr1] >= 0.99
                        and output_attr2[j][prediction_attr2] >= 0.99
                    ):
                        result_label = dicenames2[prediction_attr1] + str(prediction_attr2)
                        move(paths[j], newpath, result_label)
                    else:
                        move(paths[j], newpath, "Unknown")

import torch.nn as nn
# label_name = ['assassin1', 'assassin2', 'assassin3', 'assassin4', 'assassin5', 'assassin6', 'assassin7', 'background', 'broken_growning1', 'broken_growning2',
            #   'broken_growning3', 'broken_growning4', 'broken_growning5', 'broken_growning6', 'broken_growning7', 'bubble1', 'bubble2', 'bubble3', 'bubble4', 'bubble5', 'bubble6', 'bubble7', 'growning1', 'growning2', 'growning3', 'growning4', 'growning5', 'growning6', 'growning7', 'jocker1', 'jocker2', 'jocker3', 'jocker4', 'jocker5', 'jocker6', 'jocker7', 'mimic1', 'mimic2', 'mimic3', 'mimic4', 'mimic5', 'mimic6', 'mimic7', 'summon1', 'summon2', 'summon3', 'summon4', 'summon5', 'summon6', 'summon7', 'sup1', 'sup2', 'sup3', 'sup4', 'sup5', 'sup6', 'sup7', 'yinyun1', 'yinyun2', 'yinyun3', 'yinyun4', 'yinyun5', 'yinyun6', 'yinyun7']
label_name=['mimic1', 'mimic2', 'mimic3', 'mimic4', 'mimic5', 'mimic6', 'mimic7', 'jocker1', 'jocker2', 'jocker3', 'jocker4', 'jocker5', 'jocker6', 'jocker7', 'assassin1', 'assassin2', 'assassin3', 'assassin4', 'assassin5', 'assassin6', 'assassin7', 'summon1', 'summon2', 'summon3', 'summon4', 'summon5', 'summon6', 'summon7', 'bubble1', 'bubble2', 'bubble3', 'bubble4', 'bubble5', 'bubble6', 'bubble7', 'background1','error']
dicenames1 = ['growning', 'yinyun', 'jocker', 'sup', 'broken_growning', ]
mode='sup'
dicenames2 = ['mimic', 'jocker', 'assassin', 'summon', 'bubble','background']
if __name__ == '__main__':
    mod = MyModel(6,8)
    mod.load_state_dict(torch.load(r'MyModeltest15.pth'))
    mod.eval().cuda()
    # for i in range(1,8):
    newpath = r'C:\python_dev\yinyun_auto_play_randomdice\2024\assassin1/'
    predict_and_move_with_progress(newpath,mod,newpath)