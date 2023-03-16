import cv2
import os
from tqdm import tqdm

src_dir = r'C:\Users\SOHAM PADHYE\Documents\Sharp_blur_dataset\CV_project\Input\Night_KB'

# Get a list of all the image files in the folder
image_files = [f for f in os.listdir(src_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
total_images = len(image_files)

images = os.listdir(src_dir)
dst_dir = r'C:\Users\SOHAM PADHYE\Documents\Sharp_blur_dataset\CV_project\output\Blurred_images'

for i, img in tqdm(enumerate(images), total=len(images)):
    img = cv2.imread(f"{src_dir}/{images[i]}")
    # Define the kernel size for horizontal blurring
    kernel_size = (121, 11)  # should be odd

    # Apply the horizontal blur using a Gaussian filter
    blur = cv2.GaussianBlur(img, kernel_size, 0)
    cv2.imwrite(f"{dst_dir}/{images[i]}", blur)
    

print('DONE')



