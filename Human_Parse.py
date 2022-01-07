import torch
import numpy as np
import torch.nn as nn
import cv2
import torchvision.transforms as transforms
from human_parse_model import network

_input_size = [473, 473]
_num_classes = 20
_label = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
          'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
          'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']


class HumanParser(object):
    def __init__(self):
        self.model = self.__init_Model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

    def __init_Model(self):
        model = network(num_classes=_num_classes, pretrained=None)
        model = nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load('/home/benkesheng/BMI_DETECT/exp-schp-201908261155-lip.pth'))
        model.eval()
        return model

    def Arms_detect(self, img):  # img read by cv2
        h, w, _ = img.shape
        aspect_ratio = _input_size[1] * 1.0 / _input_size[0]
        person_center, s = self.box2cs([0, 0, w - 1, h - 1], aspect_ratio)
        r = 0
        trans = self.get_affine_transform(person_center, s, r, _input_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(_input_size[1]), int(_input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        input = self.transform(input).unsqueeze(dim=0)
        output = self.model(input.cuda())
        upsample_output = torch.nn.functional.interpolate(output, size=_input_size, mode='bilinear', align_corners=True)
        # upsample_output = upsample(output)
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
        upsample_output = upsample_output.data.cpu().numpy()
        trans = self.get_affine_transform(person_center, s, 0, _input_size, inv=1)
        channel = upsample_output.shape[2]
        target_logits = []
        for i in range(channel):
            target_logit = cv2.warpAffine(
                upsample_output[:, :, i],
                trans,
                (int(w), int(h)),  # (int(width), int(height)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0)
            )
            target_logits.append(target_logit)
        target_logits = np.stack(target_logits, axis=2)
        parsing_result = np.argmax(target_logits, axis=2)
        result = list(map(lambda x: list(map(lambda y: y == 14 or y == 15, x)), parsing_result))
        result = np.asarray(result)
        return result

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def get_affine_transform(self, center,
                             scale,
                             rot,
                             output_size,
                             shift=np.array([0, 0], dtype=np.float32),
                             inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            print(scale)
            scale = np.array([scale, scale])

        src_w = scale[0]
        dst_w = output_size[1]
        dst_h = output_size[0]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
        dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def box2cs(self, box, ar):
        x, y, w, h = box[:4]
        return self.xywh2cs(x, y, w, h, ar)

    def xywh2cs(self, x, y, w, h, ar):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > ar * h:
            h = w * 1.0 / ar
        elif w < ar * h:
            w = h * ar
        scale = np.array([w, h], dtype=np.float32)
        return center, scale
