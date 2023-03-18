#Owner Soham Padhye, Karan Bhakuni
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

# Path to the folder containing images
folder_path = r"C:\Users\91981\Desktop\OnePlus_Photo\Night_images"

# Get a list of all the image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_images = len(image_files)

#Define a function to enhance night images
def Night_image():
    with tqdm(total=total_images, desc="Processing images") as pbar:
        # Loop through all the image files
        for i, file_name in enumerate(image_files):
            # Load the image
            img_path = os.path.join(folder_path, file_name)
            img1 = cv2.imread(img_path)
            #Resize the image
            img1=cv2.resize(img1,(812,612))
            kernel=np.array([[0, -7, 0], [-7, 10, -7], [0, -7, 0]])
            img=cv2.bilateralFilter(img1,2,8,8)
            #Sharpening of image
            gaussian_blur=cv2.GaussianBlur(img,(3,3),2)
            img=cv2.addWeighted(img,1,gaussian_blur,-0.5,0)
            #Brightness Matrix
            matrix=np.ones(img.shape,dtype="uint8")*6
            #Contrast Matrix
            matrix1=np.ones(img.shape)*1.5
            #Changing Brightness
            bright=cv2.add(img,matrix)
            #Changing contrast
            contrast_bright=np.uint8(np.clip(cv2.multiply(np.float64(bright),matrix1),0,255))
            #define the function for Histogram equalization
            def Hist_Equ(img):
                r,g,b=cv2.split(img)
                eq_r=cv2.equalizeHist(r)
                eq_g=cv2.equalizeHist(g)
                eq_b=cv2.equalizeHist(b)
                eq_img=cv2.merge((eq_r,eq_g,eq_b))
                return eq_img
            #calling the function
            equalized_gray_dehazed_img = Hist_Equ(contrast_bright)
            #Difference of images
            diff_img=cv2.absdiff(contrast_bright,img)
            Sharp_img=cv2.add(6*diff_img,img)
            # Convert the image to grayscale
            gray = cv2.cvtColor(Sharp_img, cv2.COLOR_BGR2GRAY)
            # Apply adaptive thresholding to create a binary image
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            # Invert the binary image
            thresh = cv2.bitwise_not(thresh)
            # Use the inverted binary image as a mask to set the shadow areas to a fixed value (e.g. white)
            result = cv2.add(Sharp_img, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))
            # Invert the binary image
            result = cv2.bitwise_not(result)
            hsv_img = cv2.cvtColor(Sharp_img, cv2.COLOR_BGR2HSV)
            # Increase the saturation by a factor of 1.3
            hsv_img[..., 1] = np.uint8(np.clip(hsv_img[..., 1] * 1.7,20,150))
            # Convert back to original color space
            sat_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            output_path=r'C:\Users\91981\Desktop\OnePlus_Photo\Night_images_enhanced'
            out_path = os.path.join(output_path,file_name)

            # result=np.hstack((img1,denoised_img))....Uncomment this to stack two images

            cv2.imwrite(out_path,sat_img)
            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix_str("{:.1f}%".format((i+1)/total_images*100))

#Calling the function for low light enhancement
Night_image()
