#This code is used to deblure the image using Wiener deconvolution method
#Owner karan Bhakuni, Soham Padhye (IIT Jodhpur)
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Read the input image and convert to grayscale
img = cv2.imread('Image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Apply a horizontal motion blur filter to the image
kernel_length = 15
kernel_angle = 0
kernel = np.zeros((kernel_length, kernel_length))
kernel[int((kernel_length-1)/2),:] = np.ones(kernel_length)
kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((kernel_length/2,kernel_length/2),kernel_angle,1.0), (kernel_length, kernel_length))

blurred = cv2.filter2D(gray, -1, kernel)

# Step 3: Compute the 2D Fourier Transform of the blurred image
f = np.fft.fft2(blurred)
fshift = np.fft.fftshift(f)

# Step 4: Create a filter in Fourier domain that will attenuate the frequencies corresponding to the blur
rows, cols = gray.shape
crow, ccol = int(rows/2), int(cols/2)
mask = np.ones((rows,cols), np.uint8)
mask[crow-kernel_length//2:crow+kernel_length//2, ccol-kernel_length//2:ccol+kernel_length//2] = 0

# Step 5: Apply the filter to the Fourier transform of the blurred image
fshift_filtered = fshift * mask

# Step 6: Compute the inverse Fourier Transform of the filtered Fourier image
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Step 7: Normalize the resulting image to obtain a grayscale image with pixel values between 0 and 255
img_deblurred = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Step 8: Display the original and deblurred images side by side
plt.subplot(121),plt.imshow(gray,cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blurred,cmap = 'gray')
plt.title('Deblurred Image'), plt.xticks([]), plt.yticks([])
plt.show()
