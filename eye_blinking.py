import os
import cv2
#import face_recognition
import numpy as np
import argparse
import imutils
import pickle
import time
from collections import defaultdict
from sklearn.svm import LinearSVC
from skimage import feature
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from eye_status import *

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default=r'C:\Users\arunav\Desktop\liveness-detection-opencv\liveness.model',
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, default=r'C:\Users\arunav\Desktop\liveness-detection-opencv\le.pickle',
	help="path to label encoder")
ap.add_argument("-s", "--svm", type=str, default=r'C:\Users\arunav\Desktop\liveness-detection-opencv\SVM.pickle',
	help="path to svm trained model")
ap.add_argument("-d", "--detector", type=str, default=r'C:\Users\arunav\Desktop\liveness-detection-opencv\face_detector',
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




def init():
    face_cascPath = 'haarcascade_frontalface_alt.xml'
    # face_cascPath = 'lbpcascade_frontalface.xml'

    open_eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'
    left_eye_cascPath = 'haarcascade_lefteye_2splits.xml'
    right_eye_cascPath ='haarcascade_righteye_2splits.xml'
    dataset = 'faces'

    face_detector = cv2.CascadeClassifier(face_cascPath)
    open_eyes_detector = cv2.CascadeClassifier(open_eye_cascPath)
    left_eye_detector = cv2.CascadeClassifier(left_eye_cascPath)
    right_eye_detector = cv2.CascadeClassifier(right_eye_cascPath)
    svm = pickle.loads(open(args["svm"], "rb").read())
    le = pickle.loads(open(args["le"], "rb").read())
    lbp=LocalBinaryPatterns(8,1)
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
    	"res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    print("[LOG] Opening webcam ...")
    video_capture = VideoStream(src=0).start()

    model = load_model()


    return (model,face_detector, open_eyes_detector, left_eye_detector,right_eye_detector, video_capture,svm,le,lbp,net)



def isBlinking(history, maxFrames):

    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    return False

def detect_and_display(model, video_capture, face_detector, open_eyes_detector, left_eye_detector, right_eye_detector, eyes_detected,svm,le,lbp,net):
        frame = video_capture.read()
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        name="unknown"
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
                        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        # pass the face ROI through the trained liveness detector
                        # model to determine if the face is "real" or "fake"
                        y=startY
                        h=startY-endY
                        x=startX
                        w=startX-endX
                        eyes = []
                        # Eyes detection
                        # check first if eyes are open (with glasses taking into account)
                        open_eyes_glasses = open_eyes_detector.detectMultiScale(
                            gray_face,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(30, 30),
                            flags = cv2.CASCADE_SCALE_IMAGE
                        )
                        # if open_eyes_glasses detect eyes then they are open
                        if len(open_eyes_glasses) == 2:
                                eyes_detected[name]+='1'
                                for (ex,ey,ew,eh) in open_eyes_glasses:
                                        cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                        # otherwise try detecting eyes using left and right_eye_detector
                        # which can detect open and closed eyes
                        else:
                                
                                # separate the face into left and right sides
                                left_face = frame[startY:endY, int(startX/2):endX]
                                left_face_gray = gray[startY:endY, int(startX/2):endX]
                                right_face = frame[startY:endY, startX:int(endX/2)]
                                right_face_gray = gray[startY:endY, startX:int(endX/2)]
                                # Detect the left eye
                                left_eye = left_eye_detector.detectMultiScale(
                                        left_face_gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30, 30),
                                        flags = cv2.CASCADE_SCALE_IMAGE
                                )
                                # Detect the right eye
                                right_eye = right_eye_detector.detectMultiScale(
                                        right_face_gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30, 30),
                                        flags = cv2.CASCADE_SCALE_IMAGE
                                )
                                eye_status = '1'
                                print(right_eye)
                                # we suppose the eyes are open
                                # For each eye check wether the eye is closed.
                                # If one is closed we conclude the eyes are closed
                                if(len(left_eye)!=0 and len(right_eye)!=0):
                                        
                                        for (ex,ey,ew,eh) in right_eye:
                                                
                                                color = (0,255,0)
                                                pred = predict(right_face[ey:ey+eh,ex:ex+ew],model)
                                                print(pred)
                                                if pred == 'closed':
                                                        eye_status='0'
                                                        color = (0,0,255)
                                                        cv2.rectangle(right_face,(ex,ey),(ex+ew,ey+eh),color,2)
                                        for (ex,ey,ew,eh) in left_eye:
                                                
                                                color = (0,255,0)
                                                pred = predict(left_face[ey:ey+eh,ex:ex+ew],model)
                                                
                                                if pred == 'closed':
                                                        eye_status='0'
                                                        color = (0,0,255)
                                                        cv2.rectangle(left_face,(ex,ey),(ex+ew,ey+eh),color,2)

                                eyes_detected[name] += eye_status
                        hist=lbp.describe(gray_face)
                        pred=svm.predict(hist.reshape(1, -1))
                        label = le.classes_[pred]
                        #print(eyes_detected)
                        # Each time, we check if the person has blinked
                        # If yes, we display its name
                        if isBlinking(eyes_detected[name],3) and label=="real" :
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                # Display name
                                y = y - 15 if y - 15 > 15 else y + 15
                                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

        return frame


if __name__ == "__main__":
    (model, face_detector, open_eyes_detector,left_eye_detector,right_eye_detector, video_capture,svm,le,lbp,net) = init()
    #data = process_and_encode(images)

    eyes_detected = defaultdict(str)
    while True:
        frame = detect_and_display(model, video_capture, face_detector, open_eyes_detector,left_eye_detector,right_eye_detector, eyes_detected,svm,le,lbp,net)
        cv2.imshow("Face Liveness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    video_capture.stop()
