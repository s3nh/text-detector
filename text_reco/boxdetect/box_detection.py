# Simple logic 
# box on input - preprocess - sliced image on output
import cv2

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



