import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
import skimage
import argparse


import flask 
from flask import Flask, request, render_template 

from skimage import io 
import numpy as np 
import json 
import zipfile
from collections import OrderedDict

import text_reco.models.craft.craft_utils as craft_utils
import text_reco.models.craft.imgproc as img_proc

from text_reco.models.craft.craft import CRAFT
from text_reco.models.craft.craft_reader import CraftReader
from text_reco.boxdetect.box_detection import BoxDetect
from text_reco.models.crnn.crnn_run import CRNNReader

#def build_args():
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--infile', type = str, help = 'dataset to preprocess')
#    args = parser.parse_args()
#   return args
app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        file_=request.files['image']
        if not file_:
            return render_template('index.html', label = "No file")
        crr = CraftReader(file_)
        boxes, img_res = crr.boxes_detect()
        results = {}
        for _, tmp_box in enumerate(boxes):
            x = int(tmp_box[0][0])
            y = int(tmp_box[0][1])
            w = int(np.abs(tmp_box[0][0] - tmp_box[1][0]))
            h = int(np.abs(tmp_box[0][1] - tmp_box[2][1]))
            tmp_img =  img_res[y:y+h, x:x+w]
            tmp_img = Image.fromarray(tmp_img.astype('uint8')).convert('L')
            tmp_img = crnn.transformer(tmp_img)
            tmp_img = tmp_img.view(1, *tmp_img.size())
            tmp_img = Variable(tmp_img)
            results['{}'.format(_)] = crnn.get_predictions(tmp_img)
        return render_template('index.html', label = results)

if __name__ == "__main__":
    crnn=CRNNReader()

    app.run(host='0.0.0.0', port=8000, debug = True)
