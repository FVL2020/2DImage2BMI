import torch.utils.data as data
import sys

sys.path.append('/home/benkesheng/BMI_DETECT/')

from sklearn.metrics import mean_absolute_error

from sklearn.svm import SVR

from Detected import Image_Processor
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import models, transforms
import numpy as np
import os
import pandas as pd
import cv2
import re
import csv
from PIL import Image
from Data import Img_info
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)

END_EPOCH = 0

mask_model = "/home/benkesheng/BMI_DETECT/pose2seg_release.pkl"
keypoints_model = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
# P = Image_Processor(mask_model,keypoints_model)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda:3")
IMG_SIZE = 224
BATCH_SIZE = 64


def _get_image_size(img):
    if transforms.functional._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class Resize(transforms.Resize):

    def __call__(self, img):
        h, w = _get_image_size(img)
        scale = max(w, h) / float(self.size)
        new_w, new_h = int(w / scale), int(h / scale)
        return transforms.functional.resize(img, (new_w, new_h), self.interpolation)


class Dataset(data.Dataset):
    def __init__(self, file, transfrom):
        self.Pic_Names = os.listdir(file)
        self.file = file
        self.transfrom = transfrom

    def __len__(self):
        return len(self.Pic_Names)

    def __getitem__(self, idx):
        img_name = self.Pic_Names[idx]
        Pic = Image.open(os.path.join(self.file, self.Pic_Names[idx]))
        Pic = self.transfrom(Pic)
        try:
            ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", img_name)
            BMI = (int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2
            Pic_name = os.path.join(self.file, self.Pic_Names[idx])
            return (Pic, Pic_name), BMI
        except:
            return (Pic, ''), 10000


transform = transforms.Compose([
    Resize(IMG_SIZE),
    transforms.Pad(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

dataset = Dataset('/home/benkesheng/BMI_DETECT/datasets/Image_train', transform)
# val_dataset = Dataset('/home/benkesheng/BMI_DETECT/datasets/Image_val', transform)
test_dataset = Dataset('/home/benkesheng/BMI_DETECT/datasets/Image_test', transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)


# Vgg16
# Pred_Net = torchvision.models.vgg16(pretrained=True)
# for param in Pred_Net.parameters():
#     param.requires_grad = True
#
# Pred_Net.classifier = nn.Sequential(
#     nn.Linear(25088, 1024),
#     nn.ReLU(True),
#     nn.Linear(1024, 512),
#     nn.ReLU(True),
#     nn.Linear(512, 256),
#     nn.ReLU(True),
#     nn.Linear(256, 20),
#     nn.ReLU(True),
#     nn.Linear(20, 1)
# )

# Resnet101

Pred_Net = models.resnet101(pretrained=True,num_classes= 1)
print(Pred_Net)
for param in Pred_Net.parameters():
    param.requires_grad = True
# Pred_Net.fc = nn.Sequential(
#     nn.Linear(2048, 1024),
#     nn.ReLU(True),
#     nn.Linear(1024, 512),
#     nn.ReLU(True),
#     nn.Linear(512, 256),
#     nn.ReLU(True),
#     nn.Linear(256, 20),
#     nn.ReLU(True),
#     nn.Linear(20, 1)
# )

Pred_Net = Pred_Net.to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam([
    {'params': Pred_Net.parameters()}
], lr=0.0001)


def train(model, device, train_loader, epoch):
    model.train()
    runing_loss = 0.0
    for idx, ((x, n), y) in enumerate(train_loader, 0):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        #         print(y_pred.shape)
        y = torch.unsqueeze(y, 1)
        loss = criterion(y_pred.double(), y.double())
        loss.backward()
        optimizer.step()

        runing_loss += loss.item()
    print('loss:', loss.item())
    print('Train Epoch:{}\t RealLoss:{:.6f}'.format(epoch, runing_loss / len(train_loader)))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def test(model, device, test_loader):
    model.eval()
    pred = []
    targ = []
    with torch.no_grad():
        for i, ((x, n), y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            # optimizer.zero_grad()
            y_pred = model(x)
            pred.append(y_pred.item())
            targ.append(y.item())
            y = torch.unsqueeze(y, 1)
    MAE = mean_absolute_error(targ, pred)
    MAPE = mean_absolute_percentage_error(targ, pred)
    print('\nTest MAE:{}\t Test MAPE:{} '.format(MAE, MAPE))
    return MAE, MAPE



MIN_MAE, MAPE = test(Pred_Net, DEVICE, val_loader)
for epoch in range(100):
    print('*' * 50)
    train(Pred_Net, DEVICE, train_loader, epoch)
    val_MAE, val_MAPE = test(Pred_Net, DEVICE, val_loader)
    if val_MAE < MIN_MAE:
        MIN_MAE = val_MAE
        torch.save(Pred_Net.state_dict(), '/home/benkesheng/BMI_DETECT/MODEL/param/MIN_RESNET101_BMI_Cache_test.pkl')
        END_EPOCH = epoch
Net = Pred_Net
Net.load_state_dict(torch.load('/home/benkesheng/BMI_DETECT/MODEL/param/MIN_RESNET101_BMI_Cache.pkl'))
Net = Net.to(DEVICE)

print('=' * 50)
Net.eval()
test(Net, DEVICE, test_loader)
# Net.train()
# train(Net, DEVICE, train_loader, 1)

print('END_EPOCH:', END_EPOCH)
print('=' * 50)





# class Dataset(data.Dataset):
#     def __init__(self, file, transfrom):
#         self.Pic_Names = os.listdir(file)
#         self.file = file
#         self.transfrom = transfrom
#
#     def __len__(self):
#         return len(self.Pic_Names)
#
#     def __getitem__(self, idx):
#         img_name = self.Pic_Names[idx]
#         Pic = Image.open(os.path.join(self.file, self.Pic_Names[idx]))
#         Pic = self.transfrom(Pic)
#
#         ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", img_name)
#         sex = 0 if (ret.group(1) == 'F' or ret.group(1) == 'f') else 1
#         age = int(ret.group(2))
#         height = int(ret.group(3)) / 100000
#         weight = int(ret.group(4)) / 100000
#         BMI = weight / (height ** 2)
#         #         BMI = (int(ret.group(4))/100000) / (int(ret.group(3))/100000)**2
#         Pic_name = os.path.join(self.file, self.Pic_Names[idx])
#         return (Pic, Pic_name, img_name, sex, age, height, weight), BMI
#
#
# dataset_train = Dataset('/home/benkesheng/BMI_DETECT/datasets/Image_train', transform)
# # dataset_val = Dataset('/home/benkesheng/BMI_DETECT/datasets/Image_val',transform)
# dataset_test = Dataset('/home/benkesheng/BMI_DETECT/datasets/Image_test', transform)
# loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)
# loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True)
#
#
# # loader_val = torch.utils.data.DataLoader(dataset_val,batch_size=1,shuffle=True)
#
# class LayerActivations:
#     features = None
#
#     def __init__(self, model, layer_num):
#         self.hook = model[layer_num].register_forward_hook(self.hook_fn)
#
#     def hook_fn(self, module, input, output):
#         self.features = output.cpu()
#
#     def remove(self):
#         self.hook.remove()
#
#
# import time
#
#
# # with open('/home/benkesheng/BMI_DETECT/ReDone_CSV/HaveArms/Image_train1.csv', 'w', newline='') as fp:
# #     writer = csv.writer(fp)
# #     cnt = 1
# #     loaders = [loader_train]
# #     for loader in loaders:
# #         for (data, name, img_name, sex, age, height, weight), target in loader:
# #             try:
# #             # if(1):
# #                 if target.numpy()[0] <= 10 or target.numpy()[0] > 100:
# #                     continue
# #                 values = []
# #                 data = data.to(DEVICE)
# #
# #                 values.append(img_name[0])
# #                 values.append(target.numpy()[0])
# #                 values.append(sex.numpy()[0])
# #
# #                 cnt += 1
# #                 t0 = time.time()
# #                 img_e = cv2.imread(name[0])
# #                 print('Handling the pic %s' % img_name[0])
# #                 F = P.Process(img_e)
# #                 print('The body features detected cost %.3f s' % (time.time() - t0))
# #                 values.append(F.WSR)
# #                 values.append(F.WTR)
# #                 values.append(F.WHpR)
# #                 values.append(F.WHdR)
# #                 values.append(F.HpHdR)
# #                 values.append(F.Area)
# #                 values.append(F.H2W)
# #                 conv_out = LayerActivations(Net.fc, 6)
# #                 t1 = time.time()
# #                 out = Net(data.to(DEVICE))
# #                 conv_out.remove()
# #                 xs = torch.squeeze(conv_out.features.detach()).numpy()
# #                 print('The deep features detected cost %.3f s' % (time.time() - t1))
# #                 for x in xs:
# #                     values.append(x)
# #
# #                 values.append(age.numpy()[0])
# #                 values.append(height.numpy()[0])
# #                 values.append(weight.numpy()[0])
# #                 writer.writerow(values)
# #                 print('The %d pic %s cost %.3f s' % (cnt, img_name[0], time.time() - t0))
# #                 print('The shape of the pic is %d * %d' % (img_e.shape[0], img_e.shape[1]))
# #                 print('*' * 40)
# #             except:
# #                 print('error')
# #                 continue
# #
# # with open('/home/benkesheng/BMI_DETECT/ReDone_CSV/HaveArms/Image_test1.csv', 'w', newline='') as fp:
# #     writer = csv.writer(fp)
# #     cnt = 0
# #     for (data, name, img_name, sex, age, height, weight), target in loader_test:
# #         try:
# #             if target.numpy()[0] <= 10 or target.numpy()[0] > 100:
# #                 continue
# #             values = []
# #             data = data.to(DEVICE)
# #
# #             values.append(img_name[0])
# #             values.append(target.numpy()[0])
# #             values.append(sex.numpy()[0])
# #
# #             print(img_name[0], '\t', str(cnt))
# #             cnt += 1
# #             img_e = cv2.imread(name[0])
# #             F = P.Process(img_e)
# #             values.append(F.WSR)
# #             values.append(F.WTR)
# #             values.append(F.WHpR)
# #             values.append(F.WHdR)
# #             values.append(F.HpHdR)
# #             values.append(F.Area)
# #             values.append(F.H2W)
# #             conv_out = LayerActivations(Net.fc, 6)
# #             out = Net(data.to(DEVICE))
# #             conv_out.remove()
# #             xs = torch.squeeze(conv_out.features.detach()).numpy()
# #             #             print(xs)
# #             for x in xs:
# #                 values.append(x)
# #
# #             values.append(age.numpy()[0])
# #             values.append(height.numpy()[0])
# #             values.append(weight.numpy()[0])
# #             writer.writerow(values)
# #         except:
# #             print('error')
# #             continue
#
#
# def Pre(raw_data):
#     raw_data = raw_data.iloc[:, 1:]
#     raw_data = raw_data.replace([np.inf, -np.inf], np.nan)
#     raw_data = raw_data.replace(np.nan, 0)
#     raw_data = raw_data.values
#     return raw_data
#
#
# def Data(raw_data):
#     x_5f = raw_data[:, 3:8]
#     x_7f = raw_data[:, 2:9]
#     x_20f = raw_data[:, 9:29]
#     y = raw_data[:, 0]
#     return x_5f, x_7f, x_20f, y
#
#
# raw_data = pd.read_csv('/home/benkesheng/BMI_DETECT/ReDone_CSV/HaveArms/Image_train.csv')
# raw_data = Pre(raw_data)
#
# x_5f_tr, x_7f_tr, x_20f_tr, y_train = Data(raw_data)
#
# x_7f_sm = raw_data[:, 2:9]
# x_5f_sm = raw_data[:, 3:8]
# Mean_7f = np.mean(x_7f_sm, axis=0)
# Std_7f = np.std(x_7f_sm, axis=0)
# Mean_5f = np.mean(x_5f_sm, axis=0)
# Std_5f = np.std(x_5f_sm, axis=0)
#
# # x_7f_tr = (x_7f_tr - Mean_7f)/Std_7f
# # x_5f_tr = (x_5f_tr - Mean_5f)/Std_5f
# x_train = np.append(x_7f_tr, x_20f_tr, axis=1)
# y_train = y_train
#
# raw_data_test = pd.read_csv('/home/benkesheng/BMI_DETECT/ReDone_CSV/HaveArms/Image_test.csv')
# raw_data_test = Pre(raw_data_test)
#
# x_5f, x_7f, x_20f, y_test = Data(raw_data_test)
#
# # x_7f = (x_7f - Mean_7f) / Std_7f
# # x_5f = (x_5f - Mean_5f) / Std_5f
#
# x_test = np.append(x_7f, x_20f, axis=1)
# y_test = y_test
#
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
# import sklearn.gaussian_process
#
# svr1 = SVR()
# svr2 = SVR()
# svr3 = SVR()
# kr1 = KernelRidge()
# kr2 = KernelRidge()
# kr3 = KernelRidge()
# kernel = DotProduct() + WhiteKernel()
# gpr1 = sklearn.gaussian_process.GaussianProcessRegressor()
# gpr2 = sklearn.gaussian_process.GaussianProcessRegressor()
# gpr3 = sklearn.gaussian_process.GaussianProcessRegressor()
#
# svr1.fit(x_train, y_train)
# svr2.fit(x_20f_tr, y_train)
# svr3.fit(np.append(x_5f_tr, x_20f_tr, axis=1), y_train)
# kr1.fit(x_train, y_train)
# kr2.fit(x_20f_tr, y_train)
# kr3.fit(np.append(x_5f_tr, x_20f_tr, axis=1), y_train)
# gpr1.fit(x_train, y_train)
# gpr2.fit(x_20f_tr, y_train)
# gpr3.fit(np.append(x_5f_tr, x_20f_tr, axis=1), y_train)
#
# y_svr1 = svr1.predict(x_test)
# y_svr2 = svr2.predict(x_20f)
# y_svr3 = svr3.predict(np.append(x_5f, x_20f, axis=1))
# y_kr1 = kr1.predict(x_test)
# y_kr2 = kr2.predict(x_20f)
# y_kr3 = kr3.predict(np.append(x_5f, x_20f, axis=1))
# y_gpr1 = gpr1.predict(x_test)
# y_gpr2 = gpr2.predict(x_20f)
# y_gpr3 = gpr3.predict(np.append(x_5f, x_20f, axis=1))
#
#
# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#
#
# print('SVR1 7+20: MAE: ', mean_absolute_error(y_test, y_svr1), ' MAPE: ',
#       mean_absolute_percentage_error(y_test, y_svr1))
# print('SVR2 20: MAE: ', mean_absolute_error(y_test, y_svr2), ' MAPE: ', mean_absolute_percentage_error(y_test, y_svr2))
# print('SVR3 5+20: MAE: ', mean_absolute_error(y_test, y_svr3), ' MAPE: ',
#       mean_absolute_percentage_error(y_test, y_svr3))
#
# print('KRR1 7+20: MAE: ', mean_absolute_error(y_test, y_kr1), ' MAPE: ', mean_absolute_percentage_error(y_test, y_kr1))
# print('KRR2 20: MAE: ', mean_absolute_error(y_test, y_kr2), ' MAPE: ', mean_absolute_percentage_error(y_test, y_kr2))
# print('KRR3 5+20: MAE: ', mean_absolute_error(y_test, y_kr3), ' MAPE: ', mean_absolute_percentage_error(y_test, y_kr3))
#
# print('GPR1 7+20: MAE: ', mean_absolute_error(y_test, y_gpr1), ' MAPE: ',
#       mean_absolute_percentage_error(y_test, y_gpr1))
# print('GPR2 20: MAE: ', mean_absolute_error(y_test, y_gpr2), ' MAPE: ', mean_absolute_percentage_error(y_test, y_gpr2))
# print('GPR3 5+20: MAE: ', mean_absolute_error(y_test, y_gpr3), ' MAPE: ',
#       mean_absolute_percentage_error(y_test, y_gpr3))
