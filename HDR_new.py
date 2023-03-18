import cv2
from skimage import color, data, restoration
import numpy as np
import sys
from skimage import io,  metrics
import os
from tqdm import tqdm

EPSILON = sys.float_info.epsilon

class ImageDeblur:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.image = cv2.resize(self.image, (312, 312))
        self.psf = np.ones((5, 5)) / 15
    
    def preprocess(self):
        self.image = self.image / 255.
        self.image[self.image >= 1] = 1 - EPSILON
    
    def deblur(self):
        B = self.image.reshape(self.image.shape[0], self.image.shape[1], 3)[:,:,0]
        G = self.image.reshape(self.image.shape[0], self.image.shape[1], 3)[:,:,1]
        R = self.image.reshape(self.image.shape[0], self.image.shape[1], 3)[:,:,2]
        deconvolved_B = restoration.wiener(B, self.psf, 1, clip=False)
        deconvolved_G = restoration.wiener(G, self.psf, 1, clip=False)
        deconvolved_R = restoration.wiener(R, self.psf, 1, clip=False)
        rgbArray = np.zeros((self.image.shape[0], self.image.shape[1], 3))
        rgbArray[..., 0] = deconvolved_B
        rgbArray[..., 1] = deconvolved_G
        rgbArray[..., 2] = deconvolved_R
        self.normal_deblur = rgbArray
    
    def f(self, q, a=2.5, c=0.6, func=None):
        if func is not None:
            return func(q)
        q_a = q**a
        return np.power((q_a/(1+q_a)), c)
    
    def f_inverse(self, f, a=2.5, c=0.6, func=None):
        if func is not None:
            return func(f)
        f_cth_root = np.power(f, 1/c)
        return np.power(f_cth_root/(1-f_cth_root), 1/a)
    
    def lightspace_deblur(self):
        self.lightspace_deblur = self.f_inverse(self.image)
    
    def show_result(self):
        result = np.hstack((self.image, self.normal_deblur, self.lightspace_deblur))
        cv2.imshow('result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

image_deblur = ImageDeblur('night_blur4.jpg')
image_deblur.preprocess()
image_deblur.deblur()
image_deblur.lightspace_deblur()
image_deblur.show_result()
