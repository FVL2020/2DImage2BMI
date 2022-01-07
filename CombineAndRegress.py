import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from model import Dfembeding
from sklearn.kernel_ridge import KernelRidge
import torch
from PIL import Image
from utils import *
import csv
import torch.utils.data as data
import pandas as pd


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


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
#         try:
#             ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", img_name)
#             BMI = (int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2
#             Pic_name = os.path.join(self.file, self.Pic_Names[idx])
#             return (Pic, Pic_name), BMI
#         except:
#             return (Pic, ''), 10000


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

        ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", img_name)
        sex = 0 if (ret.group(1) == 'F' or ret.group(1) == 'f') else 1
        age = int(ret.group(2))
        height = int(ret.group(3)) / 100000
        weight = int(ret.group(4)) / 100000
        BMI = weight / (height ** 2)
        #         BMI = (int(ret.group(4))/100000) / (int(ret.group(3))/100000)**2
        Pic_name = os.path.join(self.file, self.Pic_Names[idx])
        return (Pic, Pic_name, img_name, sex, age, height, weight), BMI


def CombineDFBF(model, BodyFeatures, df, loader_test, loader_train):
    # test(model, DEVICE, loader_test)
    loaders = [ loader_test, loader_train,]
    files = [ 'test', 'train',]
    for loader, file in zip(loaders, files):
        with open('/home/benkesheng/BMI_DETECT/Deep_Learning_Method/DF_BF_csv/20-1_{}.csv'.format(file), 'w',
                  newline='') as fp:
            writer = csv.writer(fp)
            model.eval()
            pred = []
            targ = []
            for (img, name, img_name, sex, age, height, weight), target in loader:
                values = []
                img, target = img.to(DEVICE), target.to(DEVICE)
                img_name = img_name[0]
                # print('Processing IMage :', img_name)
                values.append(img_name)
                values.append(target.cpu().numpy()[0])
                values.append(sex.numpy()[0])

                values.append(BodyFeatures[img_name]['WSR'])
                values.append(BodyFeatures[img_name]['WTR'])
                values.append(BodyFeatures[img_name]['WHpR'])
                values.append(BodyFeatures[img_name]['WHdR'])
                values.append(BodyFeatures[img_name]['HpHdR'])
                values.append(BodyFeatures[img_name]['Area'])
                values.append(BodyFeatures[img_name]['H2W'])
                conv_out = LayerActivations(model.fc, 1)
                out = model(img)
                pred.append(out.item())
                targ.append(target.item())
                conv_out.remove()
                xs = torch.squeeze(conv_out.features.detach()).numpy()
                # print(xs.shape)

                for x in xs:
                    values.append(float(x))

                values.append(age.numpy()[0])
                values.append(height.numpy()[0])
                values.append(weight.numpy()[0])
                writer.writerow(values)
            MAE = mean_absolute_error(targ, pred)
            print(file,' ',MAE)

def Pre(raw_data, name):
    if (name != 'vgg16'):
        raw_data = raw_data.iloc[:, 1:]
    raw_data = raw_data.replace([np.inf, -np.inf], np.nan)
    #     raw_data = raw_data.fillna(raw_data.mean())
    raw_data = raw_data.replace(np.nan, 0)
    raw_data = raw_data.values.astype(np.float64)
    return raw_data


def Feature(data, df, name):
    if (name == 'author'):
        x_5f = data[:, 0:5]
        y = data[:, 9]
        return x_5f, y
    elif (name == 'vgg16'):
        x_df = data[:, 2:]
        y = data[:, 0]
        return x_df, y
    elif (name == 'ours'):
        x_5f = data[:, 3:8]
        x_7f = data[:, 2:9]
        x_df = data[:, 9:9 + df]
        y = data[:, 0]
        return x_5f, x_7f, x_df, y


def Stdm(x):
    Mean = np.mean(x, axis=0)
    Std = np.std(x, axis=0)
    return Mean, Std


def Regression(df=20, file='test'):
    # raw_data_train = pd.read_csv('/home/benkesheng/BMI_DETECT/ReDone_CSV/Ours/Image_train.csv')
    # raw_data_test = pd.read_csv('/home/benkesheng/BMI_DETECT/ReDone_CSV/Ours/Image_test.csv')

    raw_data_train = pd.read_csv('/home/benkesheng/BMI_DETECT/Deep_Learning_Method/DF_BF_csv/20-1_train.csv')
    raw_data_test = pd.read_csv('/home/benkesheng/BMI_DETECT/Deep_Learning_Method/DF_BF_csv/20-1_test.csv')
    raw_data_name = raw_data_test.values
    raw_data_train = Pre(raw_data_train, 'ours')
    raw_data_test = Pre(raw_data_test, 'ours')
    x_5f_train, x_7f_train, x_df_train, y_train = Feature(raw_data_train, df, 'ours')
    x_5f_test, x_7f_test, x_df_test, y_test = Feature(raw_data_test, df, 'ours')

    x_body_train = x_7f_train
    Mean, Std = Stdm(x_body_train)
    x_body_train = (x_body_train - Mean) / Std

    x_train = np.append(x_body_train, x_df_train, axis=1)
    y_train = y_train

    x_body_test = x_7f_test
    x_body_test = (x_body_test - Mean) / Std
    x_test = np.append(x_body_test, x_df_test, axis=1)
    y_test = y_test

    print(x_test.shape)
    print(x_train.shape)

    krr = KernelRidge()
    krr.fit(x_train, y_train)
    y_krr = krr.predict(x_test)
    print('KRR: MAE: ', mean_absolute_error(y_test, y_krr), ' MAPE: ', mean_absolute_percentage_error(y_test, y_krr))

    if file == 'demo':
        for i, data in enumerate(x_test):
            y_pred = krr.predict(data[None,:])
            print('Name: ', raw_data_name[i][0], ' y_pred:', y_pred[0], '  y_ture:', y_test[i])

if __name__ == '__main__':
    IMG_SIZE = 224
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        Resize(IMG_SIZE),
        transforms.Pad(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD)
    ])
    DEVICE = torch.device("cuda:0")
    dataset_train = Dataset('/home/benkesheng/BMI_DETECT/datasets/Image_train', transform)
    dataset_test = Dataset('/home/benkesheng/BMI_DETECT/datasets/Image_test', transform)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True)

    df = 20
    model = Dfembeding()
    # model.load_state_dict(torch.load('/home/benkesheng/BMI_DETECT/ReDone_CSV/model/Ours.pkl'.format(df)))

    model.load_state_dict(torch.load('/home/benkesheng/BMI_DETECT/MODEL/9-1reexperiment/MIN_RESNET101_BMI_20-1fc.pkl'))
    model.to(DEVICE)

    Path = '/home/benkesheng/BMI_DETECT/Deep_Learning_Method/datasets_bodyfeature/BodyFeature.json'

    with open(Path, 'r') as f:
        BodyFeatures = json.load(f)
    # CombineDFBF(model, BodyFeatures, df, loader_test, loader_train)
    Regression(df)