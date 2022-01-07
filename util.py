import torch
import random
import numpy as np
from torchvision import transforms
import re
import os
import torch.utils.data as data
from PIL import Image

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

def get_loader(mode = 'normal'):
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
    if mode == 'normal':
        dataset_train = Dataset('/home/benkesheng/BMI_DETECT/datasets/Image_train', transform)
        # dataset_val = Dataset('/home/benkesheng/BMI_DETECT/datasets/Image_val',transform)
        dataset_test = Dataset('/home/benkesheng/BMI_DETECT/datasets/Image_test', transform)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True)
        return loader_train, loader_test
    elif mode == 'demo':
        dataset_demo = Dataset('/home/benkesheng/BMI_DETECT/datasets/Demo/RealPicture', transform)
        loader_demo = torch.utils.data.DataLoader(dataset_demo, batch_size=1)
        return loader_demo

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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


class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()

