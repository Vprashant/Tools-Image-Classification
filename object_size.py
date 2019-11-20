# USAGE
# python object_size.py --image images/testimg.jpg --width 0.955

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import mahotas
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib
import pydip as dip

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


def modelprediction(image):
	filename = 'trained_model.sav'
	#cv2.imshow('img--', image)
	image = cv2.resize(image, (120, 120))
	##----feature calculation -----------##
	fv_hu_moments = fd_hu_moments(image)
	fv_haralick   = fd_haralick(image)
	#fv_histogram  = fd_histogram(image)
	
	global_feature = np.hstack([fv_haralick, fv_hu_moments])
	reshape_feature = global_feature.reshape(1,-1)
  
	loaded_model = joblib.load(filename)
	prediction = loaded_model.predict(reshape_feature)[0]
	print('prediction-value...', prediction)
	return prediction

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

i =1
for c in cnts:
	# if the contour is not sufficiently large, ignore it

	if cv2.contourArea(c) < 800:
		continue
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	box = perspective.order_points(box)
	(tl, tr, br, bl) = box
	print(tl, tr, br, bl)
	xmin = min(tl[0], tr[0], br[0], bl[0])
	xmax = max(tl[0], tr[0], br[0], bl[0]) 
	ymin = min(tl[1], tr[1], br[1], bl[1]) 
	ymax = max(tl[1], tr[1], br[1], bl[1])

	print(xmin, xmax, ymin,ymax)
	
	crop_img = image[int(ymin):int(ymax), int(xmin):int(xmax) ]

	cv2.imshow('crop_img.jpg', crop_img)
	cv2.imwrite('{}.jpg'.format(i), crop_img)
	i = i + 1

	prediction = modelprediction(crop_img)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	# loop over the original points and draw them
	for (x, y) in box:
		#print('x--y', x, y)
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]

	img2 = orig.copy()
	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric
	train_labels = ['Bolt', 'Nut']
	# draw the object sizes on the image
	x, y = 0,0
	h, w = orig.shape[:2]
	W = int( w + (w * 40)/100)
	B_IMG = np.zeros((h,W,3), np.uint8)
	B_IMG[0:y+h, x:x+W] = (164, 151, 61)
	Bh, Bw = B_IMG.shape[:2]
	B_IMG[0:y+h, x:x+w] = orig
	Bh, Bw = B_IMG.shape[:2]
	#cv2.imshow('BIMG', B_IMG)
	#cv2.waitKey(0)
	Y = 183
	X1, X2 = int(w + (w*5)/100), int(Bw - (Bw*5)/100) 
	Y1, Y2 = int(h - (h*70)/100), int(h - (h*30)/100) 
	cv2.rectangle(B_IMG, (X1, Y1), (X2, Y2), (255, 255, 255), 1)
	#cv2.line(image, (int(tltrX - 20), int(tltrY -20)), (517, 167), (0,255,0), 2)
	
	cv2.rectangle(img2, (xmin, ymin),(xmax , ymax), (0, 0 , 255) , 1)
	cv2.putText(B_IMG, "Tool Classifier", (int(X1), (40)), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
	cv2.putText(B_IMG, "Class: {}".format(train_labels[prediction]), (int(X1), int(Y1 + (Y1*15)/100)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1) 
	cv2.putText(B_IMG, "Height: {:.1f}in".format(dimA),(int(X1), int(Y1+ (Y1*25)/100)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
	cv2.putText(B_IMG, "Width: {:.1f}in".format(dimB),(int(X1), int(Y1 + (Y1*35)/100)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1)
	cv2.imwrite('Img_{}.jpg'.format(i), B_IMG)
	
	# show the output image
	cv2.imshow("ssdd",  cv2.resize(img2, (0, 0), fx =1, fy=1))
	cv2.imshow("Image", cv2.resize(B_IMG, (0, 0), fx =1, fy=1))
	cv2.waitKey(0)