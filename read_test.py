from text_reco.boxdetect.box_detection import BoxDetect
import cv2 
import numpy as np 
import json
import numpy as np 

# Test loadboxow

def main():
    img = cv2.imread('data/resized.png')
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    with open('data/box.json', 'r') as jsonfile:
        boxes = json.load(jsonfile)
    
    for key_ in boxes.keys(): 
        tmp_box = boxes[key_]
        x = int(tmp_box[0][0])
        y = int(tmp_box[0][1])
        w = int(np.abs(tmp_box[0][0] -  tmp_box[1][0]))
        h = int(np.abs(tmp_box[0][1] - tmp_box[2][1]))

        print(tmp_box)
        print("Y {} X {} W {} H {}".format(y,x ,w,h))
    
        tmp_img = img[y:y+h, x:x+w]
        cv2.imshow('tmp_img', tmp_img)
        cv2.waitKey(0)
    

if __name__ == "__main__":
    main()
