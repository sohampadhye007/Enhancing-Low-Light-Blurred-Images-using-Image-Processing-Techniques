# Enhancing-Low-Light-Blurred-Images-using-Image-Processing-Techniques
# Objectives- 
1.Enhance the low light images and remove the noise in them. 
2.Deblur the images taken by camera mounted on Autonomous vehicle. 

# Low Light Image Enhancement

Methods for low light image enhancement- 

1.Histogram Equalization------->> NIght_Enhancement_by_Histogram_equalization.py\\
2.Image enhancement by KS algorithm. We have developed this algorithm for enhancing very low light image enhancement ------->>Night_KS_algorithm.py
3.Parameter tuning for different kind of low light images (medium low light, very low light image enhancement) ------->>Night_parameter_tuning.py 
4.Nature inspired low light image enhancement ------->>Night_Nature_Inspired.py 


 
I have collected low light images dataset by capturing the images using two mobile phones namely OnePlus8 and RealmeXT. I have blurred these images to create the blurred image dataset. 

Code for horizontal blurring of the images ------->> add_gaussian_blur.py 

# Deblurring the images 

Low light image dataset is uploaded on google drive(Link has access to only IITJ mail ids, please send request for dataset if  anyone outside from IIT Jodhpur wants access). Link for the dataset ------->> https://drive.google.com/drive/folders/19wZ61NiU9pYiMUaeF_RjB9hLcpecleJq?usp=sharing 

 
# Methods for deblurring the images- 

1.Deblurring using Sharpening kernel ------->> Bluring_and_Debluring_Images.py 
2.Deblurring using Wiener deconvolution ------->> Deblur_Wiener_Deconvolution.py 
3.Light Space debluring ------->>Light_space_debluring.py 
4.Training the deep learning model on custom low light image dataset. Main program file is deblur_ae.py  model file is model.py.  First run model.py file then run deblur_ae file. 

 
# Steps to run the project files- 

1.Enhance the low light images using a suitable method for your application. 
2.Use the "add_gaussian_blur.py" file to blur the images. 
3.Choose one of the methods for deblurring. 
4.Make sure to properly organize the blurred and sharp images for training the model. 

 
