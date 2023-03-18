import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread
from skimage.color import rgb2yuv, rgb2hsv, rgb2gray, yuv2rgb, hsv2rgb
from scipy.signal import convolve2d
import os

#Reading Image
dog= cv2.imread('IMG_20230226_001127.jpg')
dog=cv2.resize(dog,(300,300))
# Apply box filter with kernel size of 3x3
dog= cv2.boxFilter(dog, -1, (3, 3))
dog_grey = cv2.cvtColor(dog,cv2.COLOR_BGR2GRAY)
cv2.imshow("Original",dog)


#Kernels
# Sharpen
sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
# Gaussian Blur
gaussian = (1 / 16.0) * np.array([[1., 2., 1.],
                                  [2., 4., 2.],
                                  [1., 2., 1.]])



#Convolution
def multi_convolver(image, kernel, iterations):
    for i in range(iterations):
        image = convolve2d(image, kernel, 'same', boundary = 'fill',fillvalue = 0)
    return image

convolved_image = multi_convolver(dog_grey, gaussian, 20)



def convolver_rgb(image, kernel, iterations = 1):
    img_yuv = rgb2yuv(image)   
    img_yuv[:,:,0] = multi_convolver(img_yuv[:,:,0], kernel, 
                                     iterations)
    final_image = yuv2rgb(img_yuv)
    return final_image
final_image = convolver_rgb(dog, sharpen, iterations = 2)
# cv2.imshow("FINAL",final_image)

#Convolution RGB
def convolver_comparison(image, kernel, iterations):
    img_yuv = rgb2yuv(image)
    img_yuv[:,:,0] = multi_convolver(img_yuv[:,:,0], kernel, 
                      iterations)
    final_image_yuv = yuv2rgb(img_yuv)

    img_hsv = rgb2hsv(image)
    img_hsv[:,:,2] = multi_convolver(img_hsv[:,:,2], kernel, 
                      iterations)
    final_image_hsv = hsv2rgb(img_hsv)

    convolved_image_r = multi_convolver(image[:,:,0], kernel, 
                         iterations)
    convolved_image_g = multi_convolver(image[:,:,1], kernel, 
                         iterations)
    convolved_image_b = multi_convolver(image[:,:,2], kernel,
                         iterations)

    final_image_rgb = np.dstack((np.rint(abs(convolved_image_r)), np.rint(abs(convolved_image_g)), np.rint(abs(convolved_image_b)))) /255


    

    return final_image_rgb
# convolved_rgb_gauss = convolver_comparison(dog, gaussian, 10)
# plt.figure()
# plt.imshow(convolved_rgb_gauss)

output_path=r'C:\Users\91981\Desktop\Mtech RMS\Image.jpg'


# Update the progress bar
#Sharpening
convolved_rgb_sharpen= convolver_comparison(dog, sharpen, 2)
print(convolved_rgb_sharpen.dtype)
# cv2.imshow("COnvolved",convolved_rgb_sharpen)



cv2.imwrite(output_path,convolved_image)
# cv2.imshow("Sharpened Image",filtered)
cv2.imshow("Convolved",convolved_rgb_sharpen)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

from skimage.metrics import structural_similarity as ssim

def psnr(img1, img2):
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((img1 - img2) ** 2)

    # If images are identical, PSNR is infinity
    if mse == 0:
        return float('inf')

    # Calculate PSNR using formula: PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    
    # Calculate PSNR and print the result
    # gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # gray2=cv2.cvtColor(img2,cv2.COLOR_HSV2BGR)
    # gray2=cv2.cvtColor(gray2,cv2.COLOR_BGR2GRAY)
    # ssim_index = ssim(gray1,gray2,win_size=5)
    return psnr


# psnr_val=psnr(dog,final_image)
# print(psnr_val)
# print(f"SSIM index for the enhanced image: {ssim_index}")
