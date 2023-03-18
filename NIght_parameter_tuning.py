#Owner karan Bhakuni, Soham Padhye(IIT Jodhpur)
import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# Path to the folder containing images
folder_path = r"C:\Users\91981\Desktop\OnePlus_Photo\Night_images"

# Get a list of all the image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_images = len(image_files)

def Sharp_Img():
    # Use tqdm to create a progress bar with message and percentage
    with tqdm(total=total_images, desc="Processing images") as pbar:
        # Loop through all the image files
        for i, file_name in enumerate(image_files):
            # Load the image
            img_path = os.path.join(folder_path, file_name)
            img1 = cv2.imread(img_path)
            img1=cv2.resize(img1,(812,612))
            kernel=np.array([[0, -5, 0], [-5, 8, -5], [0, -5, 0]])
            img=cv2.bilateralFilter(img1,2,10,10)

            #Sharpening of image
            gaussian_blur=cv2.GaussianBlur(img,(5,5),2)
            img=cv2.addWeighted(img,1,gaussian_blur,-0.5,0)

            # Calculate the average pixel value
            avg_pixel_value = np.mean(img1)
            #Apply different conditions for enhancing images as each image has different lighting conditions
            if avg_pixel_value<10:#Very dark image
                matrix=np.ones(img.shape,dtype="uint8")*5
                matrix1=np.ones(img.shape)*3
                
            elif avg_pixel_value>10 and avg_pixel_value<20:#Medium dark image
                matrix=np.ones(img.shape,dtype="uint8")*3
                matrix1=np.ones(img.shape)*2
                
            else:
                matrix=np.ones(img.shape,dtype="uint8")*2# rest images
                matrix1=np.ones(img.shape)*2
                

            #Changing Brightness
            bright=cv2.add(img,matrix)
            dark=cv2.subtract(img,matrix)

            #Changing contrast
            contrast_bright=np.uint8(np.clip(cv2.multiply(np.float64(bright),matrix1),0,255))
            
            diff_img=cv2.absdiff(contrast_bright,img1)
            Sharp_img=cv2.add(3*diff_img,img1)
            hsv_img = cv2.cvtColor(Sharp_img, cv2.COLOR_BGR2HSV)

            # Increase the saturation by a factor of 1.3
            hsv_img[..., 1] = np.uint8(np.clip(hsv_img[..., 1] * 1.3,20,150))

            # Convert back to original color space
            sat_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

            # Apply Non-Local Means Denoising
            denoised_img = cv2.fastNlMeansDenoisingColored(sat_img, None, 3, 3, 20, 15)
            
            output_path=r'C:\Users\91981\Desktop\OnePlus_Photo\Night_images_enhanced_by_KS'
            out_path = os.path.join(output_path,file_name)

            # result=np.hstack((img1,denoised_img))
            cv2.imwrite(out_path,denoised_img)

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix_str("{:.1f}%".format((i+1)/total_images*100))
            return None
        
#Calling the function
Sharp_Img()
        
