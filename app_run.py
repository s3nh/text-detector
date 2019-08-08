import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image

import cv2
from skimage import io 
import numpy as np 
import json 
import zipfile
from collections import OrderedDict

import text_reco.models.craft.craft_utils as craft_utils
import text_reco.models.craft.imgproc as img_proc
import text_reco.models.craft.file_utils as file_utils

from text_reco.models.craft.craft import CRAFT
from text_reco.models.craft.craft_reader import CraftReader
from text_reco.boxdetect.box_detection import BoxDetect
from text_reco.models.crnn.crnn_run import CRNNReader

def cut_box(tmp_box):
    x = int(tmp_box[0][0])
    y = int(tmp_box[0][1])
    w = int(np.abs(tmp_box[0][0] - tmp_box[1][0]))
    h = int(np.abs(tmp_box[0][1] - tmp_box[2][1]))
    return x, y, w, h

def main():
    crnn = CRNNReader('data/test.png')
    crr = CraftReader('data/test.png')
    box, img_res = crr.boxes_detect()
    for tmp_box in box:
        x, y, w, h = cut_box(tmp_box)
        tmp_img =  img_res[y:y+h, x:x+w]
        cv2.imshow('tmp_img', tmp_img)
        cv2.waitKey(0)
        tmp_img = crnn.transformer(tmp_img)
        tmp_img = img.view(1, *tmp_img.size())
        tmp_img = Variable(tmp_img)
        crnn.get_predictions(tmp_img)
		
if __name__ == "__main__":
    main()
