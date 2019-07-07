import torch
from torch.autograd import Variable

import models.crnn.utils  as utils
import models.crnn.preprocess as preprocess
from PIL import Image

import models.crnn.crnn as crnn

class CRNNReader():

    def __init__(self,img_path,  model_path= 'models/crnn/pretrain/crnn.pth'):

        self.model_path = model_path
        self.model = crnn.CRNN(32, 1,37, 256)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.img_path = img_path
        self.alphabet = '0123456789abcdefghijklmnopqrstuv2xyz'
        self.transformer =  preprocess.resizeNormalize((100, 32))

        self.converter = utils.strLabelConverter(self.alphabet)

    def load_image(self):

        img = Image.open(self.img_path).convert('L')
        img = self.transformer(img)
        # Resizing 
        img = img.view(1, *img.size())
        img = Variable(img)
        return img

    def get_predictions(self, img):
        predictions = self.model(img)
        _, predictions = predictions.max(2)
        predictions = predictions.transpose(1, 0).contiguous().view(-1)
        pred_size = Variable(torch.IntTensor([predictions.size(0)]))
        results =  self.converter.decode(predictions.data, pred_size.data, raw=False)
        return results

def main():
    crnn = CRNNReader()
    print(crnn.model)

if __name__ == "__main__":
    main()



