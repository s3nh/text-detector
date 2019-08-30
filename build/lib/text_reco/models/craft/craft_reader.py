import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn
from  torch.autograd import Variable
import text_reco.models.craft.craft_utils as craft_utils
import text_reco.models.craft.imgproc as img_proc
from text_reco.models.craft.craft import CRAFT
from text_reco.boxdetect.box_detection import BoxDetect
from text_reco.models.crnn.crnn_run import CRNNReader

from PIL import Image

import cv2 
from skimage import io 
import numpy as np 
from text_reco.models.craft.imgproc import ImageConvert
import json 
import zipfile
from collections import OrderedDict
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean

class CraftReader(ImageConvert):
    def __init__(self,  image):
        super(CraftReader, self).__init__(image)
        self.model_path = 'text_reco/models/craft/pretrain/craft_mlt_25k.pth'
        self.net = CRAFT()
        self.net.load_state_dict(self.copyStateDict(torch.load(self.model_path)))
        self.net.eval()
        self.mag_ratio = 1
        self.square_size = 1280

    @staticmethod
    def copyStateDict(state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    @staticmethod
    def str2bool(v):
        return v.lower() in ("yes", "y", "t", "1")
        
    def image_preprocess(self, image):
        image = self.normalizeMeanVariance(image)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = Variable(image.unsqueeze(0))
        return image

    def boxes_detect(self):
        img_resized, target_ratio, size_heatmap = self.resize_aspect_ratio(self.image)
        ratio_h = ratio_w = 1/ target_ratio
        x =  self.image_preprocess(img_resized)
        y, _ = self.net(x)
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()
        boxes = craft_utils.getDetBoxes(textmap =score_text, linkmap = score_link, text_threshold =0.7, link_threshold=0.4, low_text=0.4)
        print("Ilosc boxow {}".format(len(boxes)))
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        return boxes, img_resized 

