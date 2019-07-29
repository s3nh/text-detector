# Simple logic 
# box on input - preprocess - sliced image on output


class BoxDetect():

    def __init__(self, boxes):
        self.boxes = boxes
        self.n_boxes = len(self.boxes)

    def preprocess(self, image):
        img_storage = dict()
        print("To sa boxy")
        print(self.boxes)
        for el in self.boxes:
            print("to jest el")
            print(el)




