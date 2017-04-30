# SVHN-Convolutional-TensorFlow

This code is broken into three primary sections. The first part is to initialize and load in the SVHN data .mat files into Python. A large portion of this code is for formatting the data into something more accessible and convenient.

The second part of this code is to classify the 32x32 SVHN digits. These images contain one digit each. A convolutional network was created to identify what digit is in the picture. The network achieves 85.9% accuracy, and is written entirely in TensorFlow with GPU support. 

Here is an example 32x32 image. Its label is 3, or [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].

![image](https://cloud.githubusercontent.com/assets/24555661/25565566/3c91f810-2d86-11e7-96b8-c2f89b1550a4.png)

The third part of the code is to determine what is an image and what is not an image. First, a new set of data must be constructed to determine what is not a digit. To do this, random 32x32 squares were pulled from the SVHN dataset that were known not to include digits (based on the location info of the dataset). 

![image](https://cloud.githubusercontent.com/assets/24555661/25565579/7c36d990-2d86-11e7-8876-3ec801ca7f76.png)

The top image is a digit [1, 0], and the bottom image is not a digit [0, 1]. The Convolutional Neural Network to determine if a digit exists or not was also created in TensorFlow and achieves 95.5% accuracy!
