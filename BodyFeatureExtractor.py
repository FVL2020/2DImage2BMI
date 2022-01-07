import json
import os
import cv2
import torch
import sys
from Detected import Image_Processor
from Datasets_BFDF import get_dataloader
import numpy as np

mask_model = "MODEL/pose2seg_release.pkl"
keypoints_model = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
P = Image_Processor(mask_model, keypoints_model)

DEVICE = torch.device("cuda:2")
BATCH_SIZE = 64
Path = os.path.join('bodyfeature', 'BodyFeature_imagenet.json')


BodyFeature = {}
cnt = 1
loader_train, loader_val, loader_test = get_dataloader(None, dataset='Ours')

loaders = [loader_val, loader_test, loader_train]
for loader in loaders:
    for (data, name, img_name, sex, age, height, weight), target in loader:
        values = {}
        data = data.to(DEVICE)

        cnt += 1
        img_e = cv2.imread(name[0])
        print('Handling the %d pic %s' % (cnt, img_name[0]))
        try:
            F = P.Process(img_e)
        except:
            print("Can't Handle this pic!")
            continue
        # print(type(F.WSR))
        values['WSR'] = float(F.WSR)
        values['WTR'] = float(F.WTR)
        values['WHpR'] = float(F.WHpR)
        values['WHdR'] = float(F.WHdR)
        values['HpHdR'] = float(F.HpHdR)
        values['Area'] = float(F.Area)
        values['H2W'] = float(F.H2W)
        values['Age'] = float(age.numpy()[0])
        values['Height'] = float(height.numpy()[0])
        values['Weight'] = float(weight.numpy()[0])
        values['BMI'] = float(target.numpy()[0])
        values['Sex'] = int(sex.numpy()[0])

        BodyFeature[img_name[0]] = values

json_str = json.dumps(BodyFeature)
with open(Path, 'w') as json_file:
    json_file.write(json_str)


