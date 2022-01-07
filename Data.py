import numpy as np
import pandas
import cv2
from PIL import Image
from Detected import Image_Processor
import os
import re

class Data_Processor(object):
    def __init__(self,dirname,mask_model="pose2seg_release.pkl",
        keypoints_model = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"):
        self._dirname = dirname
        self._img_pro = Image_Processor(mask_model,keypoints_model)
        # self._dataframe = self.Process()

    def Data_check(self,figure,img_info):
        pass
        # if figure.WTR > 7 or figure.WHpR > 2 or figure.WHdR > 5 or figure.HpHdR > 7 :
        #     raise Exception("InvalidData!")
        # if img_info.age < 13 or img_info.age > 60 or img_info.height < 1 or img_info.height > 2 or img_info.weight < 30 or img_info.weight > 500:
        #     raise Exception("InvalidData!")

    def Process(self):
        path =os.path.join(self._dirname)
        img_names =os.listdir(path)
        Data_list = []
        columns=['WTR','WHpR','WHdR','HpHdR','Area','H2W','WSR','sex','age','height','weight','BMI']
        for img_name in img_names:
            try:
                print("processing the picture: %s"%img_name)
                ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+",img_name)
                sex = 0 if (ret.group(1) == 'F' or ret.group(1) == 'f') else 1
                img_info = Img_info(sex,int(ret.group(2)),int(ret.group(3))/100000,int(ret.group(4))/100000)
                img_path = os.path.join(path,img_name)
                    # img = Image.open(img_path)
                    # img = np.array(img)
                img = cv2.imread(img_path)
                figure = self._img_pro.Process(img)
                self.Data_check(figure,img_info)
                Data_list.append([figure.WTR,figure.WHpR,figure.WHdR,figure.HpHdR,figure.Area,figure.H2W,figure.WSR,
                                      img_info.sex,img_info.age,img_info.height,img_info.weight,img_info.BMI])

            except AssertionError as ae:
                print('!' * 50)
                print("%s in picture: %s"%(ae,img_name))
                print('!' * 50)
            except Exception as ep:
                print('!'*50)
                print("%s : %s"%(ep,img_name))
                print('!'*50)

            DataFrame = pandas.DataFrame(data=Data_list,columns=columns)
        return DataFrame

    # @property
    # def DataFrame(self):
    #     return self._dataframe

    # @property
    # def Img_info(self):
    #     pass

class Img_info(object):
    def __init__(self,sex,age,height,weight):
        self._sex = sex
        self._age = age
        self._height = height
        self._weight = weight

    @property
    def BMI(self):
        return self._weight / self._height**2

    @property
    def sex(self):
        return self._sex

    @property
    def age(self):
        return self._age

    @property
    def weight(self):
        return self._weight

    @property
    def height(self):
        return self._height




