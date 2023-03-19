#Owner Soham Padhye, Karan Bhakuni (IIT Jodhpur)
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm

# Apply the proposed decomposition and enhancement algorithm
def enhance_image(image):
    # Convert the image to HSV space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Normalize the V channel
    norm_V = hsv[:,:,2] / 255.0

    # Apply the proposed decomposition
    L = np.exp(-1 * np.log(norm_V + 0.001))
    R = norm_V / L

    # Adjust the illumination
    L_adjusted = cv2.normalize(L, None, alpha=0.5, beta=400, norm_type=cv2.NORM_MINMAX)

    # Generate the enhanced V channel image
    enhanced_V = L_adjusted * R
    enhanced_V = cv2.normalize(enhanced_V, None, alpha=0, beta=1500, norm_type=cv2.NORM_MINMAX)
    enhanced_V = enhanced_V.astype(np.uint8)

    # Merge the enhanced V channel with the H and S channels to get the enhanced HSV image
    enhanced_hsv = cv2.merge((hsv[:,:,0], hsv[:,:,1], enhanced_V))

    # Convert the enhanced HSV image to RGB space
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
  
    return enhanced_image


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


# Path to the folder containing images
folder_path = r"C:\Users\91981\Desktop\OnePlus_Photo\Night_images"

# Get a list of all the image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_images = len(image_files)

# Use tqdm to create a progress bar with message and percentage
with tqdm(total=total_images, desc="Processing images") as pbar:
    # Loop through all the image files
    lst1=[]
    lst2=[]
    for i, file_name in enumerate(image_files):
        # Load the image
        img_path = os.path.join(folder_path, file_name)
        image = cv2.imread(img_path)
        image=cv2.resize(image,(612,812))
        # Apply the enhancement algorithm
        enhanced_image = enhance_image(image)

        #Saving the image on local drive
        output_path=r'C:\Users\91981\Desktop\OnePlus_Photo\Night_Enhanced_by_RP'
        out_path = os.path.join(output_path,file_name)
        #Uncomment this to stack two images(Original image and enhanced image)
        # result=np.hstack((image,enhanced_image))
        cv2.imwrite(out_path,enhanced_image)

        #  Calculate PSNR and print the result
        psnr_value = psnr(image,enhanced_image)
        # print('PSNR:', psnr_value)
        lst1.append(psnr_value)

        # # Calculate SSIM
        image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        enhanced_image_gray=cv2.cvtColor(enhanced_image,cv2.COLOR_BGR2GRAY)
        ssim_index = ssim(image_gray,enhanced_image_gray,win_size=5)
        # print(f"SSIM index for the enhanced image: {ssim_index}")
        lst2.append(ssim_index)

        # Update the progress bar
        pbar.update(1)
        pbar.set_postfix_str("{:.1f}%".format((i+1)/total_images*100))

        


