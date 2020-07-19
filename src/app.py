import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os


@st.cache
def generate_predictions(img , conf_threshold , nms_threshold):

    label_path = os.path.sep.join([ 'Yolo', 'custom.names'])
    labels = open(label_path).read().strip().split('\n')

    #initializing colours for each class label randomly
    np.random.seed(42)
    colors = np.random.randint(0,255 , size =(len(labels) , 3) , dtype = 'uint8')

    #Loading YOLOV3 weights and Configurations

    weights_path = os.path.sep.join(['Yolo' , 'yolov3-custom_1000.weights'])
    config_path = os.path.sep.join(['Yolo' , 'yolov3-custom.cfg'])

    #Loading the model
    model = cv2.dnn.readNetFromDarknet(config_path , weights_path)

    
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
            
            
            if confidence > conf_threshold:
                
                box = detection[:4]*np.array([w, h , w, h])
                (CentreX , CentreY , Width , Height) = box.astype('int')
                
                x = int(CentreX - (Width/2))
                y = int(CentreY - (Height/2))
                
                boxes.append([x,y,int(Width) , int(Height) ,])
                confidences.append(float(confidence))
                ClassID.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes , confidences , conf_threshold , nms_threshold)

    #print(idxs)
    if len(idxs) > 0:
        #loop over the indexes we are keeping
        for i in idxs.flatten():
            #extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
                    

            #draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[ClassID[i]]]
            #print(ClassID[i])
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{} : {:.4f}".format(labels[ClassID[i]] , confidences[i])
            cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
    
    return img


st.title("Triple Seat Violation Detection using YoloV3")

st.markdown("## Problem Definition")
st.markdown("In India, there are many traffic rule violations some of the most common ones are not wearing a helmet, breaking signals, overspeeding and going triple seat on two wheelers i.e more than two people sitting on two wheelers,etc")
st.markdown("To tackle the problem of triple seat detection we have used Yolov3 to detect the triple seaters on two wheeler vehicles.")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
#st.write(img_file_buffer)
st.sidebar.text("Select the confidence Threshold")
conf_threshold = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, 0.5, 0.05
)

st.sidebar.text("Select the NMS Suppression Threshold")
nms_threshold = st.sidebar.slider(
    "NMS threshold", 0.0, 1.0, 0.5, 0.05
)

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

    

else:
    img_file_buffer = 'Test/image_235.jpg'
    image = np.array(Image.open(img_file_buffer))


st.image(image , caption = f'Input Image')
op_img = generate_predictions(img = image , conf_threshold = conf_threshold , nms_threshold= nms_threshold)

st.image(op_img , caption=f"Processed image"
)

