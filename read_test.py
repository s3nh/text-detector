from text_reco.boxdetect.box_detection import BoxDetect
import cv2 
import numpy as np 
import json
import numpy as np 

# Test loadboxow


def main():

    
    img = cv2.imread('data/resized.png')
    cv2.imshow('image', img)
    cv2.waitKey(0)
    with open('data/box.json', 'r') as jsonfile:
        boxes = json.load(jsonfile)
    for el in boxes.keys():
        print(boxes[el])
    bd = BoxDetect(boxes)
    bd.preprocess_box(boxes, img)

if __name__ == "__main__":
    main()
