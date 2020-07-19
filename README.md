# Object-Detection-for-Traffic-Surveillance-System

## Problem Definition

In India there are many traffic rule violations such as not wearing of helmet, overspeeding, breaking of signals and over loading of two wheelers due to triple seat i.e more than two people are sitting on two wheeler vehicles which as also called as tripsy in India. Deep learning has provided us with the tools to tackle this problem. Specifically we are tackling the problem of detecting triple seats on two wheeler.

## Approach

### Data Collection

Since the formulation of the problem definitions we had a lack of data. Such data  to train deep learning models didn't exist so we collected our own data with 500 images captured at various locations. 

![image_collect](https://github.com/Atharva-Phatak/Object-Detection-for-Traffic-Surveillance-System/blob/master/Images/Data_collect.PNG)

### Data Labelling

We had to label all our images manually but before labelling we resized every image to (416,416) and then did the labelling. We only used two labels "Triple Seat" and "Human". We used an opensource tool called "LabelImg" and saved the annotations in YOLO format as required. For propriety reasons we can't give access to the dataset.

![label_1](https://github.com/Atharva-Phatak/Object-Detection-for-Traffic-Surveillance-System/blob/master/Images/Data_label.PNG)

![label_2](https://github.com/Atharva-Phatak/Object-Detection-for-Traffic-Surveillance-System/blob/master/Images/Data_label_2.PNG)

### Training YOLOv3
YOLO is one of the fastest object detection algorithms out there. It frames the object
detection problem as a regression problem. Yolo architecture is more like CNN
(convolutional neural network) and passes the image through the FCNN and output is
prediction. In Yolo a single convolutional network simultaneously predicts multiple
bounding boxes and class probabilities for those boxes. YOLO trains on full images and
directly optimizes detection performance. Some of the benefits of choosing YOLO over
other algorithms are as follows:

* Extremely Fast
* YOLO reasons globally about the image
* YOLO learns general representations of objects.

We trained the network for 1500 iterations and we were able to get satisfactory results since it was a protoype.

![train](https://github.com/Atharva-Phatak/Object-Detection-for-Traffic-Surveillance-System/blob/master/Images/Training_yolo.PNG)

### Results and Deployment

For creating a web app we used newly introduced streamlit library to create simple web app. You can host it on heroku or any hosting platform.
Here is the demo of the app.

![app](https://github.com/Atharva-Phatak/Object-Detection-for-Traffic-Surveillance-System/blob/master/Images/streamlit-app.gif)

Here are some results below.

![img_results](https://github.com/Atharva-Phatak/Object-Detection-for-Traffic-Surveillance-System/blob/master/Images/Result_1.PNG)


Finally I would like to thank my team members without whom this project wouldn't have been possible. This is a group project for my final year thesis.

## Requirements

* Python
* OpenCV
* Streamlit
* Numpy

# References

* [Yolo Framework](https://github.com/AlexeyAB/darknet)
* [Yolo Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
