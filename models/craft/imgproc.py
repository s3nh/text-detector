import nump as np 
from skimage import io 
import cv2 




class ImageConvert():

    def __init__(self, img_array):

        self.img_array = img_array
        print("Shape of processed file {}".format(len(self.img_array)))
        self.mean = (0.485, 0.486, 0.406)
        self.variance =  (0.229, 0.224, 0.225)

        print("Wartosc srednia \n {} \n variance {} \n".format(self.mean, self.variance))


   def normalizeMeanVariance(self, self.mean, self.variance):

        pass


def main():

    pass


if __name__ == "__main__":

    main()
