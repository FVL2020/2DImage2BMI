import torch.utils.data as data
from torchvision import transforms
import torch
import os
import re
import cv2
import numpy as np

# COCO
# IMG_MEAN = [0.471, 0.448, 0.408]
# IMG_STD = [0.234, 0.239, 0.242]

# ImageNet
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


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


def get_dataloader(args, dataset='Ours'):
    if args is None:
        if dataset == 'Ours':
            print(11111111)
            root = '/home/benkesheng/BMI_With_BFDF/datasets/Rebuttle'
            train_dataset = OurDatasets(root, 'Image_train_Down2')
            test_dataset = OurDatasets(root, 'Image_test_Down2')
            val_dataset = OurDatasets(root, 'Image_val_Down2')
        elif dataset == 'Author':
            root = '/home/benkesheng/BMI_DETECT/author_datasets'
            train_dataset = Authordataset(root, 'Image_train')
            test_dataset = Authordataset(root, 'Image_test')
            val_dataset = Authordataset(root, 'Image_val')

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                                   num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    else:
        train_dataset = OurDatasets(args.root, 'Image_train_consist2')
        test_dataset = OurDatasets(args.root, 'Image_test_consist2')
        val_dataset = OurDatasets(args.root, 'Image_val_consist2')

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=args.workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=args.workers)

    return train_loader, val_loader, test_loader


class Authordataset(data.Dataset):
    def __init__(self, root, file, sim=False):
        self.file = os.path.join(root, file)
        self.img_names = os.listdir(self.file)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            Resize(IMG_SIZE),
            transforms.Pad(IMG_SIZE, fill=0),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
        ])
        self.sim = sim

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_name_path = os.path.join(self.file, img_name)

        img = cv2.imread(os.path.join(self.file, img_name), flags=3)[:, :, ::-1]

        h, w, _ = img.shape

        img = self.transform(img)
        img = transforms.Normalize(IMG_MEAN, IMG_STD)(img)

        ret = re.match(r"[a-zA-Z0-9]+_[a-zA-Z0-9]+__?(\d+)__?(\d+)__?([a-z]+)_*", img_name)
        height = float(ret.group(2)) * 0.0254
        weight = float(ret.group(1)) * 0.4536
        sex = (lambda x: x == 'false')(ret.group(3))
        BMI = weight / (height ** 2)

        if self.sim:
            return img, BMI
        return (img, img_name_path, img_name, sex, 20, height, weight), BMI


class OurDatasets(data.Dataset):
    def __init__(self, root, file):
        self.file = os.path.join(root, file)
        self.img_names = os.listdir(self.file)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            Resize(IMG_SIZE),
            transforms.Pad(IMG_SIZE, fill=0),
            transforms.CenterCrop(IMG_SIZE),
            # transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img = cv2.imread(os.path.join(self.file, img_name), flags=1)
        # print(img_name)
        img = img[:, :, ::-1]
        h, w, _ = img.shape
        img = self.transform(img)

        # Gray
        # img = torch.cat((img, img, img), dim=0)
        img = transforms.Normalize(IMG_MEAN, IMG_STD)(img)

        ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", img_name)
        sex = 0 if (ret.group(1) == 'F' or ret.group(1) == 'f') else 1
        age = int(ret.group(2))
        height = int(ret.group(3)) / 100000
        weight = int(ret.group(4)) / 100000
        BMI = torch.from_numpy(np.asarray((int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2))
        Pic_name = os.path.join(self.file, img_name)
        return (img, Pic_name, img_name, sex, age, height, weight), BMI
