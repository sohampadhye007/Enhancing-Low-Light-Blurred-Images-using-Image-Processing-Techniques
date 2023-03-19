
# Project Title

Enhancing low light blured images using image processing techniques.
 
# Objectives

- Enhance the low light images and remove the noise in them. 

- Deblur the images taken by camera mounted on Autonomous vehicle.

# Low Light Image Enhancement 

## Methods for low light image enhancement- 
- HistogramEqualization------->> <em>NIght_Enhancement_by_Histogram_equalization.py</em>

- Image enhancement by KS algorithm. We have developed this algorithm for enhancing very low light images ------->> <em> Night_KS_algorithm.py </em>

- Parameter tuning for different kind of low light images (medium low light, very low light image enhancement) ------->> <em> Night_parameter_tuning.py </em>

- Nature inspired low light image enhancement ------->> <em> Night_Nature_Inspired.py </em>
#
I have collected low light images dataset by capturing the images using two mobile phones namely OnePlus8 and RealmeXT. I have blurred these images to create the blurred image dataset. 
#

# Bluring images
Code for horizontal blurring of the images ------->> <em> add_gaussian_blur.py </em>

# Debluring the images 

Low light image dataset is uploaded on google drive(Link has access to only IITJ mail ids, please send request for dataset if  anyone outside from IIT Jodhpur wants access). Link for the dataset ------->> https://drive.google.com/drive/folders/19wZ61NiU9pYiMUaeF_RjB9hLcpecleJq?usp=sharing 

# Methods for deblurring the images- 

- Deblurring using Sharpening kernel ------->>  <em> Bluring_and_Debluring_Images.py </em>

- Deblurring using Fourier transform ------->> <em> Deblur_FT.py </em>

- Wiener filter debluring ------->> <em> Wiener_debluring.py </em>

- Training the deep learning model on custom low light image dataset. Main program file is deblur_ae.py  model file is model.py.  First run  <em> model.py </em> file then run <em> deblur_ae </em> file.


# Steps to run the project files- 

 1. Enhance the low light images using a suitable method for your application. 

 2. Use the " <em> add_gaussian_blur.py </em> " file to blur the images. 

 3. Choose one of the methods for deblurring. 

 4. Make sure to properly organize the blurred and sharp images for training the model. 


## Demo

<figure>
<figcaption><em>Deblurred using CNN</em></figcaption>
  <img src="https://github.com/sohampadhye007/Enhancing-Low-Light-Blurred-Images-using-Image-Processing-Techniques/blob/main/Deblur_CNN.png?raw=true" alt="Deblured Image" title="deblured Image">

</figure>

<!-- Figure with caption -->
<figure>
 <figcaption><em>Low ligth enhanced images</em></figcaption>
  <img src="https://github.com/sohampadhye007/Enhancing-Low-Light-Blurred-Images-using-Image-Processing-Techniques/blob/main/Figure_1.png?raw=true" alt="Fig 1" width="400" title="Figure 1">
</figure>



<!-- Figure with caption -->
<figure>

  <img src="https://github.com/sohampadhye007/Enhancing-Low-Light-Blurred-Images-using-Image-Processing-Techniques/blob/main/Figure_2.png?raw=true" alt="Fig 2" width="400" title="Figure 2">
</figure>



<!-- Figure with caption -->
<figure>

  <img src="https://github.com/sohampadhye007/Enhancing-Low-Light-Blurred-Images-using-Image-Processing-Techniques/blob/main/Figure_3.png?raw=true" alt="Fig 2" width="400" title="Figure 2">
</figure>



<!-- Figure with caption -->
<figure>

  <img src="https://github.com/sohampadhye007/Enhancing-Low-Light-Blurred-Images-using-Image-Processing-Techniques/blob/main/Figure_4.png?raw=true" alt="Fig 2" width="400" title="Figure 2">
</figure>



<!-- Figure with caption -->
<figure>

  <img src="https://github.com/sohampadhye007/Enhancing-Low-Light-Blurred-Images-using-Image-Processing-Techniques/blob/main/Figure_5.png?raw=true" alt="Fig 2" width="400" title="Figure 2">
</figure>




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

