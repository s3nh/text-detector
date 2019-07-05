# https://github.com/clovaai/CRAFT-pytorch/blob/master/basenet/vgg16_bn.py


# Imports 

from collections import namedtuple

import torch
import torch.nn as nn 
import torch.nn.init as init
from torchvision import models
from torchvision.models.vgg import model_urls

def init_weights(modules):
    # https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/ 

    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
       
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

            

