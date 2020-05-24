# USAGE
# python liveness_demo.py --model liveness.model --le le.pickle --detector face_detector

# import the necessary packages
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.svm import LinearSVC
from skimage import feature
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,  required=True,help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
	help="path to label encoder")
ap.add_argument("-s", "--svm", type=str, required=True,
	help="path to svm trained model")
ap.add_argument("-d", "--detector", type=str,  required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		return hist

def getDoGFeature(greyImg, sigma1=0.5, sigma2=1.0):
    """
    get the DoG feature (Difference of Gaussian) of image.

    :param img: the image waited to obtain DoG feature
    :param sigma1:
    :param sigma2:
    :return: the DoG feature of image saved in list
    """
    size1 = 2 * (int)(3 * sigma1) + 3
    size2 = 2 * (int)(3 * sigma2) + 3
    blur1 = cv2.GaussianBlur(greyImg, (size1, size1), sigmaX=sigma1,
                             sigmaY=sigma1, borderType=cv2.BORDER_REPLICATE)
    blur2 = cv2.GaussianBlur(greyImg, (size2, size2), sigmaX=sigma2,
                             sigmaY=sigma2, borderType=cv2.BORDER_REPLICATE)
    DoG = blur2 - blur1
    featureMat = np.abs(np.fft.fftshift(np.fft.fft2(DoG)))
    return np.reshape(DoG, (1, DoG.shape[0] * DoG.shape[1]))[0]


lbp=LocalBinaryPatterns(8,1)

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())
svm = pickle.loads(open(args["svm"], "rb").read())



# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the detected bounding box does fall outside the
			# dimensions of the frame
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			# extract the face ROI and then preproces it in the exact
			# same manner as our training data
			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (32, 32))
			gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)

			# pass the face ROI through the trained liveness detector
			# model to determine if the face is "real" or "fake"

			hist=lbp.describe(gray)
			pred=svm.predict(getDoGFeature(gray).reshape(1, -1))
			preds = model.predict(face)[0]
			j = np.argmax(preds)
			print("SVM:{}".format(pred))
			print("CNN:{}".format(preds))
			label = le.classes_[pred]

			# draw the label and bounding box on the frame
			label = "{}".format(label)
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)

	# show the output frame and wait for a key press
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
