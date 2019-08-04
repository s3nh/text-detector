# Simple logic 
# box on input - preprocess - sliced image on output
import cv2
import numpy as np 
class BoxDetect():

    def __init__(self, boxes):
        self.boxes = boxes
        self.n_boxes = len(self.boxes)

    def preprocess(self, image):
        img_storage = dict()
        print("To sa boxy")
        print(self.boxes)
        for el in self.boxes:
            tmp_img = cv2.rectangle(image, (el[0], el[1]), (el[2], el[3]))
    
    @staticmethod
    def load_box(path):
        with open(path, 'r') as outfile:
            file_ = json.load(outfile)
        return file_

    @staticmethod 
    def preprocess_box(file_, img):
        for el in file_.keys():
            x,y,w,h = cv2.boundingRect(np.array(file_[el]))
            roi = img[x:x+w, y:y+h]
            cv2.imshow('image', roi)
            cv2.waitKey(0)



