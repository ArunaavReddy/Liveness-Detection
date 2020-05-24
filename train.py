from livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from skimage import feature
import numpy as np
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments


ap = argparse.ArgumentParser()

ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")

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


lbp=LocalBinaryPatterns(8,1)
# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 8
EPOCHS = 30

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

imagepaths =[]
data = []
labels = []
train_hist=[]
test_hist=[]
currentDirectory = os.getcwd()
for filename in os.listdir(currentDirectory+r"/data2/"):
	for imgname in os.listdir(currentDirectory+r"/data2/"+filename):
		imagepaths.append(currentDirectory+r"/data2/"+filename+r"/"+imgname)
		labels.append(filename)


for imagePath in imagepaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32))
	data.append(image)



# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

for image in trainX:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        train_hist.append(lbp.describe(gray))

model2 = SVC(probability=True)

model2.fit(train_hist,trainY.argmax(axis=1))
predictions2=[]

for image in testX:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist=lbp.describe(gray)
        predictions2.append(model2.predict(hist.reshape(1, -1)))



# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]

trainX= np.array(trainX, dtype="float") / 255.0
testX= np.array(testX, dtype="float") / 255.0


# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,
	classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(accuracy_score(testY.argmax(axis=1),predictions.argmax(axis=1)))
print(accuracy_score(testY.argmax(axis=1),predictions2))


print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

# save the network to disk
print("[INFO] serializing network to '{}'...".format('liveness.model'))
model.save('liveness.model')

# save the label encoder to disk
f = open('le.pickle', "wb")
f.write(pickle.dumps(le))
f.close()

with open('SVM.pickle','wb') as f:
	pickle.dump(model2,f)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
