import numpy as np
import sklearn.gaussian_process  # GPR
from sklearn.gaussian_process.kernels import Exponentiation, RationalQuadratic
from sklearn.svm import SVR
import json
import os
from utils import get_sex_BMI, Resize, mean_absolute_error
from torchvision import models, transforms
import torch
import torch.nn as nn
import cv2
import csv

IMG_SIZE = 224
BATCH_SIZE = 1
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda:0")

transform = transforms.Compose([
    transforms.ToPILImage(),
    Resize(IMG_SIZE),
    transforms.Pad(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])


def Stdm(x):
    Mean = np.mean(x, axis=0)
    Std = np.std(x, axis=0)
    return Mean, Std


def Test(BodyFeature):
    Train_path = "/home/ungraduate/hjj/BMI_DETECT/datasets/Image_train"
    Test_path = "/home/ungraduate/hjj/BMI_DETECT/datasets/Image_test"
    TrainList = os.listdir(Train_path)
    TestList = os.listdir(Test_path)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_train_vgg = []
    x_test_vgg = []
    VGG_NET = models.vgg16(pretrained=True).to(DEVICE)
    VGG_NET.classifier = nn.Sequential(*list(VGG_NET.classifier.children())[:-6])
    VGG_NET.eval()

    for img in TrainList:
        img_path = os.path.join(Train_path, img)
        data = cv2.imread(img_path, flags=1)[:, :, ::-1]
        data = torch.unsqueeze(transform(data), 0).to(DEVICE)
        data = torch.squeeze(VGG_NET(data).detach()).cpu().numpy()
        x_train_vgg.append(data)
        sex, BMI = get_sex_BMI(img)
        bf = BodyFeature[img]
        x_train.append(np.asarray([bf['WTR'], bf['WHdR'], bf['WHpR'], bf['HpHdR'], bf['Area']]))
        y_train.append(BMI)

    for img in TestList:
        img_path = os.path.join(Test_path, img)
        data = cv2.imread(img_path, flags=1)[:, :, ::-1]
        data = torch.unsqueeze(transform(data), 0).to(DEVICE)
        data = torch.squeeze(VGG_NET(data).detach()).cpu().numpy()
        x_test_vgg.append(data)
        sex, BMI = get_sex_BMI(img)
        bf = BodyFeature[img]
        x_test.append(np.asarray([bf['WTR'], bf['WHdR'], bf['WHpR'], bf['HpHdR'], bf['Area']]))
        y_test.append(BMI)

    Mean, Std = Stdm(x_train)
    x_train = (x_train - Mean) / Std
    x_test = (x_test - Mean) / Std

    svr = SVR(kernel='rbf')
    svr_vgg = SVR(kernel='rbf')
    KN = Exponentiation(RationalQuadratic(), exponent=2)
    gpr = sklearn.gaussian_process.GaussianProcessRegressor(kernel=KN, alpha=1e-3)

    regressors = [svr, gpr, svr_vgg]
    y_pred = [[], [], []]
    with open('/home/benkesheng/BMI_With_BFDF/SOTA/ANOVA/SVRGPRVGG.csv', 'a+', newline='') as fp:
        writer = csv.writer(fp)
        for i, reg in enumerate(regressors):
            if i != 2:
                reg.fit(x_train, y_train)
                y_pred[i] = reg.predict(x_test)
                print("MAE:", mean_absolute_error(y_test, y_pred[i]))
            else:
                reg.fit(x_train_vgg, y_train)
                y_pred[i] = reg.predict(x_test_vgg)
                print("MAE:", mean_absolute_error(y_test, y_pred[i]))
            writer.writerow(y_pred[i])


if __name__ == '__main__':
    with open('/home/benkesheng/BMI_With_BFDF/SOTA/BdyFeature.json') as f:
        BodyFeature = json.load(f)
        Test(BodyFeature)
