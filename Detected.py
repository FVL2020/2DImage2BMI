
# coding: utf-8
import sys
sys.path.append('/home/benkesheng/BMI_DETECT')
sys.path.append('/home/benkesheng/BMI_DETECT/Single-Human-Parsing-LIP-master')
sys.path.append('/home/benkesheng/BMI_DETECT/CRF-RNN_CPM_SVR_Method/pytorch-openpose')
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from modeling.build_model import Pose2Seg
from Human_Parse import HumanParser
from PSP import HumanParser_PSP
from CPM import CPM_Keypoint
from CRFRNN import CRFRNN_Contour
import time



# import Visual


# "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"

class Body_Figure(object):
    def __init__(self, waist_width, thigh_width, hip_width, head_width, Area, height, shoulder_width):
        self._waist_width = waist_width
        self._thigh_width = thigh_width
        self._hip_width = hip_width
        self._head_width = head_width
        self._Area = Area
        self._height = height
        self._shoulder_width = shoulder_width
        if self._head_width == 0:
            self._head_width = self._hip_width/3

    @property
    def WSR(self):
        return self._waist_width / self._shoulder_width

    @property
    def WTR(self):
        return (self._waist_width / self._thigh_width)  # **2

    @property
    def WHpR(self):
        return (self._waist_width / self._hip_width)  # **2

    @property
    def WHdR(self):
        return (self._waist_width / self._head_width)  # **2

    @property
    def HpHdR(self):
        return (self._hip_width / self._head_width)  # **2

    @property
    def Area(self):
        return self._Area

    @property
    def H2W(self):
        return self._height / self._waist_width


class Image_Processor(object):

    def __init__(self, masks_file, key_file, key_thresh=0.7):
        self._KeypointCfg = self.__init_key(key_file, key_thresh)
        self._ContourPredictor = self.__init_mask(masks_file)
        # self._ContourPredictor_MaskRCNN = DefaultPredictor(self.__init_mask_RCNN())
        # self._ContourPredictor_CRF = CRFRNN_Contour()
        # self._KeypointsPredictor = CPM_Keypoint()
        self._KeypointsPredictor = DefaultPredictor(self._KeypointCfg)
        self._HumanParser = HumanParser()

    def __init_mask_RCNN(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        return cfg

    def __init_mask(self, masks_file):
        Model = Pose2Seg().cuda()  # .to("cuda:1")
        Model.init(masks_file)
        Model.eval()
        return Model

    def __init_key(self, key_file, key_thresh):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(key_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = key_thresh  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(key_file)
        # cfg.MODEL.DEVICE = 3
        return cfg

    def _detected(self, img):
        """ Detect Keypoints by CPM """
        # Keypoints = self._KeypointsPredictor.Keypoint_Detect(img)
        # gt_kpts = Keypoints[None, :, :]

        """ Detect Keypoints by Mask RCNN """
        KeypointsOutput = self._KeypointsPredictor(img)
        sorted_idxs = np.argsort(-KeypointsOutput["instances"].scores.cpu().numpy())
        Keypoints = KeypointsOutput["instances"].pred_keypoints[sorted_idxs[0]].cpu().numpy()
        gt_kpts = Keypoints[None, :, :]

        ''' Detect Contour by CRFRNN '''
        # ContourOutput = self._ContourPredictor_CRF.Contour_Detect(img)

        ''' Detect Contour by Pos2Seg '''
        ContourOutput = self._ContourPredictor([img], [gt_kpts])
        ContourOutput = np.squeeze(np.asarray(ContourOutput))

        ''' Detect Contour by Mask RCNN '''
        # ContourOutput = self._ContourPredictor_MaskRCNN(img)
        # sorted_idxs = np.argsort(-ContourOutput["instances"].scores.cpu().numpy())
        # for sorted_idx in sorted_idxs:
        #     if ContourOutput["instances"].pred_classes[sorted_idx] == 0:
        #         ContourMasks = ContourOutput["instances"].pred_masks[sorted_idx].cpu().numpy()
        # ContourOutput = ContourMasks

        """ Detect Arms Mask by Human parser """
        Arms_mask = self._HumanParser.Arms_detect(img)
        ContourOutput = ContourOutput ^ Arms_mask

        return Keypoints, ContourOutput

    # def _Contour_detected(self,img,gt_kpts):
    #
    #     return ContourOutput

    # sorted_idxs = np.argsort(-ContourOutput["instances"].scores.cpu().numpy())
    # for sorted_idx in sorted_idxs:
    #     if ContourOutput["instances"].pred_classes[sorted_idx] == 0:
    #         ContourMasks = ContourOutput["instances"].pred_masks[sorted_idx].cpu().numpy()
    #         return ContourMasks
    # return None

    # def _Keypoints_detected(self,img):
    #     KeypointsOutput = self._KeypointsPredictor(img)
    #     sorted_idxs = np.argsort(-KeypointsOutput["instances"].scores.cpu().numpy())
    #     Keypoints = KeypointsOutput["instances"].pred_keypoints[sorted_idxs[0]].cpu().numpy()
    #     return Keypoints

    # def Process(self, img_RGB):
    def Process(self, img_RGB):
        img_keypoints, img_mask = self._detected(img_RGB)
        nose, left_ear, right_ear, left_shoulder, right_shoulder = img_keypoints[0], img_keypoints[4], img_keypoints[3], \
                                                                   img_keypoints[6], img_keypoints[5]
        left_hip, right_hip, left_knee, right_knee = img_keypoints[12], img_keypoints[11], img_keypoints[14], \
                                                     img_keypoints[13]
        # return img_keypoints
        y_hip = (left_hip[1] + right_hip[1]) / 2
        y_knee = (left_knee[1] + right_knee[1]) / 2

        center_shoulder = (left_shoulder + right_shoulder) / 2
        y_waist = y_hip * 2 / 3 + (nose[1] + center_shoulder[1]) / 6
        left_thigh = (left_knee + left_hip) / 2
        right_thigh = (right_knee + right_hip) / 2

        # estimate the waist width
        waist_width = self.waist_width_estimate(center_shoulder, y_waist, img_mask)
        # estimate the thigh width
        thigh_width = self.thigh_width_estimate(left_thigh, right_thigh, img_mask)
        # estimate the hip width
        hip_width = self.hip_width_estimate(center_shoulder, y_hip, img_mask)
        # estimate the head_width
        head_width = self.head_width_estimate(left_ear, right_ear)
        # estimate the Area
        Area = self.Area_estimate(y_waist, y_hip, waist_width, hip_width, img_mask)
        # estimate the height2waist
        height = self.Height_estimate(y_knee, nose[1])
        # estimate tht shoulder_width
        shoulder_width = self.shoulder_width_estimate(left_shoulder, right_shoulder)

        figure = Body_Figure(waist_width, thigh_width, hip_width, head_width, Area, height, shoulder_width)

        return figure

    def Height_estimate(self, y_k, y_n):
        Height = np.abs(y_n - y_k)
        return Height

    def Area_estimate(self, y_w, y_h, W_w, H_w, mask):
        '''
            Area is expressed as thenumber of
            pixels per unit area between waist and hip
        '''
        pixels = np.sum(mask[int(y_w):int(y_h)][:])
        area = (y_h - y_w) * 0.5 * (W_w + H_w)
        Area = pixels / area
        return Area

    def shoulder_width_estimate(self, left_shoulder, right_shoulder):
        shoulder_width = np.sqrt(
            (right_shoulder[0] - left_shoulder[0]) ** 2 + (right_shoulder[1] - left_shoulder[1]) ** 2)
        return shoulder_width

    def head_width_estimate(self, left_ear, right_eat):
        head_width = np.sqrt((right_eat[0] - left_ear[0]) ** 2 + (right_eat[1] - left_ear[1]) ** 2)
        return head_width

    def hip_width_estimate(self, center_shoulder, y_hip, img_mask):
        x_hip_center = int(center_shoulder[0])
        x_lhb = np.where(img_mask[int(y_hip)][:x_hip_center] == 0)[0]
        x_lhb = x_lhb[-1] if len(x_lhb) else 0
        x_rhb = np.where(img_mask[int(y_hip)][x_hip_center:] == 0)[0]
        x_rhb = x_rhb[0] + x_hip_center if len(x_rhb) else len(img_mask[0])
        hip_width = x_rhb - x_lhb
        return hip_width

    def thigh_width_estimate(self, left_thigh, right_thigh, mask):
        lx, ly = int(left_thigh[0]), int(left_thigh[1])
        rx, ry = int(right_thigh[0]), int(right_thigh[1])

        x_ltb = np.where(mask[ly][:lx] == 0)[0]
        x_ltb = x_ltb[-1] if len(x_ltb) else 0
        x_rtb = np.where(mask[ry][rx:] == 0)[0]
        x_rtb = x_rtb[0] + rx if len(x_rtb) else len(mask[0])

        l_width = (lx - x_ltb) * 2
        r_width = (x_rtb - rx) * 2

        thigh_width = (l_width + r_width) / 2
        return thigh_width

    def waist_width_estimate(self, center_shoulder, y_waist, img_mask):
        x_waist_center = int(center_shoulder[0])
        # plt.imshow(img_mask)
        # plt.show()
        x_lwb = np.where(img_mask[int(y_waist)][:x_waist_center] == 0)[0]
        x_lwb = x_lwb[-1] if len(x_lwb) else 0
        x_rwb = np.where(img_mask[int(y_waist)][x_waist_center:] == 0)[0]
        x_rwb = x_rwb[0] + x_waist_center if len(x_rwb) else len(img_mask[0])
        # print(x_rwb)
        waist_width = x_rwb - x_lwb
        return waist_width

    def Vis(self, img):
        pass
