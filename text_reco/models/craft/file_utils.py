import os 
import numpy as np 
import cv2 
from PIL import Image



class DataLoader():


    def __init__(self, _file):

        self._file = _file
        self.extensions = ['.pdf', '.tif', '.png']

    

    def load_image(self):

        try:
            _img = Image.open(self._file)
            return _img
        except:
            ValueError("File does not exist!")
           

def main():
    pass


if __name__ == "__main__":
    main()
