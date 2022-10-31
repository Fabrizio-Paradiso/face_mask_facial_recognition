
# Face-Mask and Face Identification

This project consists in a web application running on a local server with the aim of identifying a person who belongs to a close group of people and then detecting if that person is wearing a Face-Mask or not.

## Overview
![overview](static/assets/styles/overview.jpg?raw=true)

## Face-Mask Detection

In order to implement Face-Mask detection, it's been trained a Machine Learning model using SVM (Support-Vector Machines) as a classifier, considering the classes 'Mask' and 'No-Mask'.

### Dataset

Dataset for Face-Mask detection is been built inserting artificially the mask in a set of images of faces belonging to the close people group.

To make that possible, it is been used this [repository](https://github.com/Prodesire/face-mask), where it can be set different positions to insert masks in human faces considering several face landmarks.

In this study case, it is been considered three face landmarks to insert masks at different heights. These ones are shown in the image below and correspond to 'nose bridge', 'top lip' and 'bottom lip'.

![landmarks](static/assets/styles/landmarks.jpg?raw=true)

At the same time, it is been taken into account three possible mask to wearing. These templates are shown below.

![masks](static/assets/styles/masks.jpg?raw=true)

Finally, once all this possibilites are been applied to the set of images, for each person you have these images.

![dataset](static/assets/styles/dataset.jpg?raw=true)
