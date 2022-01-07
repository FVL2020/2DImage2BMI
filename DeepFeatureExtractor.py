import pandas as pd
import csv
import torch
from model import DFNet
from Datasets_BFDF import get_dataloader
import os
import json
from sklearn.metrics import mean_absolute_error
from OtherTypeNet import *


class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()


loader_train, loader_val, loader_test = get_dataloader(None)
loaders = [loader_val, loader_test, loader_train]
files = ['train', 'test', 'train']

DEVICE = torch.device("cuda:1")
df = 15
model = DFNet(df=df, bf=0)
# model = Densenet121(df)
print(model)
model.load_state_dict(torch.load('MODEL/model_epoch_50.ckpt',
                                 map_location=DEVICE)['state_dict'])

model.to(DEVICE)
model.eval()
# print(model)

BFPath = os.path.join('bodyfeature', 'BodyFeature_imagenet.json')
with open(BFPath, 'r') as f:
    BodyFeatures = json.load(f)

for loader, file in zip(loaders, files):
    cnt = 0
    with open(
            'ALL_feature/Image_{}.csv'.format(file),
            'a+', newline='') as fp:
        writer = csv.writer(fp)
        pred = []
        targ = []
        for (data, name, img_name, sex, age, height, weight), target in loader:
            cnt += 1
            print(cnt)
            values = []
            data, target = data.to(DEVICE), target.to(DEVICE)
            img_name = img_name[0]

            values.append(img_name)
            values.append(target.cpu().numpy()[0])
            values.append(sex.numpy()[0])

            if img_name not in BodyFeatures:
                continue
            values.append(BodyFeatures[img_name]['WSR'])
            values.append(BodyFeatures[img_name]['WTR'])
            values.append(BodyFeatures[img_name]['WHpR'])
            values.append(BodyFeatures[img_name]['WHdR'])
            values.append(BodyFeatures[img_name]['HpHdR'])
            values.append(BodyFeatures[img_name]['Area'])
            values.append(BodyFeatures[img_name]['H2W'])

            conv_out = LayerActivations(model.fc1, None)
            out = model(data)
            pred.append(out.item())
            targ.append(target.item())
            conv_out.remove()
            xs = torch.squeeze(conv_out.features.cpu().detach()).numpy()

            for x in xs:
                values.append(float(x))

            values.append(age.numpy()[0])
            values.append(height.numpy()[0])
            values.append(weight.numpy()[0])

            writer.writerow(values)
        MAE = mean_absolute_error(targ, pred)
        print(file, ' ', MAE)
