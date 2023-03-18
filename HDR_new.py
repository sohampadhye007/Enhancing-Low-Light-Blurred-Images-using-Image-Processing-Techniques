import cv2
from skimage import color, data, restoration
import numpy as np
import sys
from skimage import io,  metrics
import os
from tqdm import tqdm

EPSILON=sys.float_info.epsilon

# Define a function that applies the lightness transform to a given input value q
def f(q, a=2.5, c=0.6, func=None):
    # If a custom transform function is provided, use that instead
    if func is not None:
        return func(q)
    # Otherwise, apply the default lightness transform
    q_a = q ** a
    return np.power((q_a / (1 + q_a)), c)

# Define a function that applies the inverse lightness transform to a given input value f
def f_inverse(f, a=2.5, c=0.6, func=None):
    # If a custom inverse transform function is provided, use that instead
    if func is not None:
        return func(f)

    # Otherwise, apply the default inverse lightness transform
    f_cth_root = np.power(f, 1/c)
    return np.power(f_cth_root / (1 - f_cth_root), 1/a)

# Define a function to preprocess an image for deblurring
def preprocess(image):
    # Normalize pixels to [0, 1] range
    image = image / 255.
    # Clip values greater than or equal to 1 to avoid numerical issues
    image[image >= 1] = 1 - EPSILON
    return image

# Define a function to deblur an image using the Wiener filter
def deblur(image, n=5, m=15):
    # Define the point spread function
    psf = np.ones((n, n)) / m

    # Separate the image into its RGB channels
    B = image.reshape(image.shape[0], image.shape[1], 3)[:,:,0]
    G = image.reshape(image.shape[0], image.shape[1], 3)[:,:,1]
    R = image.reshape(image.shape[0], image.shape[1], 3)[:,:,2]

    # Apply the Wiener filter to each channel
    deconvolved_B = restoration.wiener(B, psf, 1, clip=False)
    deconvolved_G = restoration.wiener(G, psf, 1, clip=False)
    deconvolved_R = restoration.wiener(R, psf, 1, clip=False)

    # Combine the filtered channels back into an RGB image
    rgbArray = np.zeros((image.shape[0], image.shape[1], 3))
    rgbArray[..., 0] = deconvolved_B
    rgbArray[..., 1] = deconvolved_G
    rgbArray[..., 2] = deconvolved_R

    return rgbArray

# Create a class for deblurring images using the lightness transform
class LightSpaceDeblurrer:
    def __init__(self, a=2.5, c=0.6, n=5, m=15):
        self.a = a
        self.c = c
        self.n = n
        self.m = m
        self.psf = np.ones((self.n, self.n)) / self.m
    
    # Define a method to apply the lightness transform to an image
    def apply_transform(self, image):
        return f(image, self.a, self.c)
    
    # Define a method to apply the inverse lightness transform to an image
    def apply_inverse_transform(self, image):
        return f_inverse
