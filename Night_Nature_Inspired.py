#Owner Karan Bhakuni, Soham Padhye(IIT Jodhpur)
import cv2
import numpy as np
from tqdm import tqdm
import os
from skimage.metrics import structural_similarity as ssim

# Path to the folder containing images
folder_path = r"C:\Users\91981\Desktop\OnePlus_Photo\Night_images"

# Get a list of all the image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_images = len(image_files)


def psnr(img1, img2):
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((img1 - img2) ** 2)

    # If images are identical, PSNR is infinity
    if mse == 0:
        return float('inf')

    # Calculate PSNR using formula: PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    return psnr
psnr_list=[]
ssim_list=[]
def Night_Nature():
    # Use tqdm to create a progress bar with message and percentage
    with tqdm(total=total_images, desc="Processing images") as pbar:
        # Loop through all the image files
        for i, file_name in enumerate(image_files):
            # Load the image
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)
            img=cv2.resize(img,(612,812))
            # Split the image into its color channels
            b, g, r = cv2.split(img)

            # Perform nature-inspired low light enhancement on each channel
            clahe = cv2.createCLAHE(clipLimit=9.0, tileGridSize=(20,20))
            enhanced_b = clahe.apply(b)
            enhanced_g = clahe.apply(g)
            enhanced_r = clahe.apply(r)

            # Merge the enhanced channels back into an RGB image
            enhanced_img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))

            # Apply gamma correction for additional enhancement
            gamma = 1.2
            enhanced_img = np.clip((enhanced_img / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)

            output_path=r'C:\Users\91981\Desktop\OnePlus_Photo\Night_Enhanced_Nature_Inspired'
            out_path = os.path.join(output_path,file_name)
            # result=np.hstack((img,enhanced_img))
            cv2.imwrite(out_path,enhanced_img)

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix_str("{:.1f}%".format((i+1)/total_images*100))

            # # Calculate PSNR and print the result
            psnr_value = psnr(img,enhanced_img)
            psnr_list.append(np.around(psnr_value))
            
            # # Calculate SSIM
            img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            enhanced_img_gray=cv2.cvtColor(enhanced_img,cv2.COLOR_BGR2GRAY)
            ssim_index = ssim(img_gray,enhanced_img_gray,win_size=5)
            ssim_list.append(np.around(ssim_index))
            return None


#Calling the function
Night_Nature()




