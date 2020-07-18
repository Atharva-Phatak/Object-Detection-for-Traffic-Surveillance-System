#Importing Libraries

import time 
import os
import argparse
import cv2
import numpy as np

#Constructing the argument parser to parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('-i' , '--image' , required = True , help = 'path to the input image')
parser.add_argument('-y' , '--YOLO' , required = True , help = 'path the YOLO folder')
parser.add_argument('-c' , '--confidence' , type = float , default = 0.5 , help = 'minimum confidence level to filter out weak detections')
parser.add_argument('-t' , '--threshold' , type = float  , default = 0.3 , help = 'Threshold when applying non maxima suppresion')

args = vars(parser.parse_args())


#loading COCO class labels on which YOLO was trained

label_path = os.path.sep.join([args["YOLO"] , 'custom.names'])
labels = open(label_path).read().strip().split('\n')

#initializing colours for each class label randomly
np.random.seed(42)
colors = np.random.randint(0,255 , size =(len(labels) , 3) , dtype = 'uint8')

#Loading YOLOV3 weights and Configurations

weights_path = os.path.sep.join([args['YOLO'] , 'yolov3-custom_1500.weights'])
config_path = os.path.sep.join([args['YOLO'] , 'yolov3-custom.cfg'])

#Loading the model
model = cv2.dnn.readNetFromDarknet(config_path , weights_path)

img = cv2.imread(args['image'])
h,w = img.shape[:2]

#GEtting the layer names

names = model.getLayerNames()
names = [names[i[0] - 1] for  i in model.getUnconnectedOutLayers()]

#Constructing Blobs from images and performing a forward pass through YOLO
blobs = cv2.dnn.blobFromImage(img , 1/255.0 , (416,416) , swapRB = True , crop = False)
model.setInput(blobs)
layerOP = model.forward(names)


boxes , confidences , ClassID = [],[],[]

for output in layerOP:
    
    for detection in output:
        
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        
        
        if confidence > args['confidence']:
            
            box = detection[:4]*np.array([w, h , w, h])
            (CentreX , CentreY , Width , Height) = box.astype('int')
            
            x = int(CentreX - (Width/2))
            y = int(CentreY - (Height/2))
            
            boxes.append([x,y,int(Width) , int(Height) ,])
            confidences.append(float(confidence))
            ClassID.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes , confidences , args['confidence'] , args['threshold'])

print(idxs)
if len(idxs) > 0:
	#loop over the indexes we are keeping
    for i in idxs.flatten():
        #extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
                

        #draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[ClassID[i]]]
        print(ClassID[i])
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = "{} : {:.4f}".format(labels[ClassID[i]] , confidences[i])
        cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)


# show the output image
cv2.imshow("Image", img)
cv2.waitKey(0)
