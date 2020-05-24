import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import argparse
import os




ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
        help="path to input video")
ap.add_argument("-d", "--detector", type=str, required=True,
        help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16,
        help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0
energy_spectrum =[]
FD = []
# loop over frames from the video file stream
while True:
        # grab the frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
                break

        # increment the total number of frames read thus far
        read += 1

        # check to see if we should process this frame
        if read % args["skip"] != 0:
                continue

        # grab the frame dimensions and construct a blob from the frame
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # ensure at least one face was found
        if len(detections) > 0:
                # we're making the assumption that each image has only ONE
                # face, so find the bounding box with the largest probability
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                # ensure that the detection with the largest probability also
                # means our minimum probability test (thus helping filter out
                # weak detections)
                if confidence > args["confidence"]:
                        # compute the (x, y)-coordinates of the bounding box for
                        # the face and extract the face ROI
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        face = frame[startY:endY, startX:endX]

                        i=0
                        if(i<4):

                                gray_face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                                f=np.fft.fft2(gray_face)
                                fshift = np.fft.fftshift(f)
                                a=((20*np.log(np.abs(fshift)))**2)/4
                                b=sum(np.trapz(a,axis=1))/(a.shape[1]*a.shape[0])

                                energy_spectrum.append(b)
                                i+=1

for i in range(len(energy_spectrum)):
	FD.append((energy_spectrum[i]-(sum(energy_spectrum)/len(energy_spectrum)))**2/len(energy_spectrum))


fd=sum(FD)/len(FD)
print(fd)

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
