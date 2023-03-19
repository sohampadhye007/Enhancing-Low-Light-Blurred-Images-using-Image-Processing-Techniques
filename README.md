
# Project Title

Enhancing low light blured images using image processing techniques.
 
# Objectives

- Enhance the low light images and remove the noise in them. 

- Deblur the images taken by camera mounted on Autonomous vehicle.

# Low Light Image Enhancement 

## Methods for low light image enhancement- 
- HistogramEqualization------->>NIght_Enhancement_by_Histogram_equalization.py

- Image enhancement by KS algorithm. We have developed this algorithm for enhancing very low light images ------->>Night_KS_algorithm.py 

- Parameter tuning for different kind of low light images (medium low light, very low light image enhancement) ------->>Night_parameter_tuning.py 

- Nature inspired low light image enhancement ------->>Night_Nature_Inspired.py
#
I have collected low light images dataset by capturing the images using two mobile phones namely OnePlus8 and RealmeXT. I have blurred these images to create the blurred image dataset. 
#

# Bluring images
Code for horizontal blurring of the images ------->> add_gaussian_blur.py 

# Debluring the images 

Low light image dataset is uploaded on google drive(Link has access to only IITJ mail ids, please send request for dataset if  anyone outside from IIT Jodhpur wants access). Link for the dataset ------->> https://drive.google.com/drive/folders/19wZ61NiU9pYiMUaeF_RjB9hLcpecleJq?usp=sharing 

# Methods for deblurring the images- 

- Deblurring using Sharpening kernel ------->> Bluring_and_Debluring_Images.py 

- Deblurring using Wiener deconvolution ------->> Deblur_Wiener_Deconvolution.py 

- Light Space debluring ------->>Light_space_debluring.py 

- Training the deep learning model on custom low light image dataset. Main program file is deblur_ae.py  model file is model.py.  First run model.py file then run deblur_ae file.


# Steps to run the project files- 

- Enhance the low light images using a suitable method for your application. 

- Use the "add_gaussian_blur.py" file to blur the images. 

- Choose one of the methods for deblurring. 

- Make sure to properly organize the blurred and sharp images for training the model. 


## Demo

<img src="https://raw.githubusercontent.com/sohampadhye007/Enhancing-Low-Light-Blurred-Images-using-Image-Processing-Techniques/main/sharp0.jpg?token=GHSAT0AAAAAAB7FQKRBRIGVBBTF4WWJRBHUZAWZ6OQ" alt="Sharp Image" title="TSharp Image">




## Documentation

[Documentation](https://linktodocumentation)


## Installation


```bash
  Numpy
  MatplotLib
  OpenCv
  os
  tensorflow
  pyTorch
  skimage
```
    
## Related

Here are some related projects

(https://github.com/sovit-123/image-deblurring-using-deep-learning)


## Authors

- [@sohampadhye007](https://github.com/sohampadhye007)
- [@karansspk](https://github.com/karansspk)



## Feedback

If you have any feedback, please reach out to us at sohampadhye1998@gmail.com, karansspk@gmail.com

