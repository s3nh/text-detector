import torch
import torch.nn as nn 
from torch.autograd import Variable
import collections



class strLabelConverter(object):


    def __init__(self, alphabet, ignore_case=True):

        self._ignore_case = ignore_case
        if self._ignore_case:

            alphabet = alphabet.lower()
        self.alphabet = alphat + '-'


        # To do: encode char 


