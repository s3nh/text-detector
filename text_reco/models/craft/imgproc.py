import numpy as np 
from skimage import io 
import cv2 


class ImageConvert():
    def __init__(self, img_array, interpolation =cv2.INTER_LINEAR ,  square_size = 1280,  mag_ratio=1):
        self.img_array = img_array
        print("Shape of processed file {}".format(len(self.img_array)))
        self.mean = (0.485, 0.486, 0.406)
        self.variance =  (0.229, 0.224, 0.225)
        self.square_size = square_size
        self.interpolation = interpolation
        self.mag_ratio = mag_ratio

    def normalizeMeanVariance(self):
        self.image_array -= np.array([self.mean[0]  * 255.0, self.mean[1] * 255.0, self.mean[2] * 255.0], dtype = np.float32)
        self.image_array /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype = np.float32)
        return self.image_array

    def resize_aspect_ratio(self):
        heigt, width, channel = self.image_array.shape
        target_size  = self.mag_ratio * max(height, width)

        if target_size > self.square_size:
            target_size = self.square_size

        ratio = target_size / max(height, width)
        target_h, target_w = int(height * ratio), int(width * ratio)
        proc = cv2.resize(self.img_array, (target_w, target_h), interpolation = interpolation)

        # Canvas

        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)

        resized = np.zeros((taget_h32, target_w32, channel), dtype = np.float32)
        resized[0:target_h, 0:target_w, :] = proc
        target_h, target_w = target_h32, target_w32

        size_heatmap = (int(target_w/2), int(target_h/2))
        return resized, ratio, size_heatmap

    def cvt2HeatmapImg(img):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        return img
        

