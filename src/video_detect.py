import cv2
import os
import numpy as np 
import argparse
import time
import imutils

parser = argparse.ArgumentParser()
parser.add_argument('-i' , '--input' , required = True , help = 'input path to the video')
parser.add_argument('-o' , '--output' , required = True , help = 'output path to the video')
parser.add_argument('-y' , '--yolo' , required = True , help = 'path to the yolo files')
parser.add_argument('-c' , '--confidence' , default = 0.5 ,type = float, help = 'minimum probability to filter out weak detections')
parser.add_argument('-t' , '--threshold' , default = 0.3 , type = float , help = 'threshold for NMS suppression')

args = vars(parser.parse_args())

labelsPath = os.path.sep.join([args["yolo"], "custom.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


weightsPath = os.path.sep.join([args["yolo"], "yolov3-custom_1500.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3-custom.cfg"])


print("loading YOLO from disk...")
print(weightsPath)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#print(ln)
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print(" {} total frames in video".format(total))

# an error occurred while trying to determine the total

except:
	print("could not determine # of frames in video")
	print("no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward pass
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			
			if confidence > args["confidence"]:
				#Scaling the bounding boxes relative to size of image
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		
		if total > 0:
			elap = (end - start)
			print("single frame took {:.4f} seconds".format(elap))
			print("estimated total time to finish: {:.4f}".format(
				elap * total))

	
	writer.write(frame)



writer.release()
vs.release()
