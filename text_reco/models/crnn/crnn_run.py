from PIL import Image
from skimage import io
import text_reco.models.crnn.crnn as crnn

import torch
from torch.autograd import Variable
import text_reco.models.crnn.utils  as utils
import text_reco.models.crnn.preprocess as preprocess

class CRNNReader():
    def __init__(self, model_path= 'text_reco/models/crnn/pretrain/crnn.pth'):
        self.model_path = model_path
        self.model = crnn.CRNN(32, 1,37, 256)
        self.model = self.model.float()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.alphabet = '0123456789abcdefghijklmnopqrstuv2xyz'
        self.transformer =  preprocess.resizeNormalize((100, 32))
        self.converter = utils.strLabelConverter(self.alphabet)

    def get_predictions(self, img):
        self.model = self.model.float()
        img = img.float()
        predictions = self.model(img)
        _, predictions = predictions.max(2)
        predictions = predictions.transpose(1, 0).contiguous().view(-1)
        pred_size = Variable(torch.IntTensor([predictions.size(0)]))
        results =  self.converter.decode(predictions.data, pred_size.data, raw=False)
        return results
