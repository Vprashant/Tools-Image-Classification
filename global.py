#-----------------------------------
# GLOBAL FEATURE EXTRACTION
#-----------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py

#--------------------
# tunable-parameters
#--------------------
images_per_class = 40
fixed_size       = tuple((120, 120))
train_path       = "dataset/train"
h5_data          = 'output/data.h5'
h5_labels        = 'output/labels.h5'
bins             = 8

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
# feature-descriptor-4: Cany Edges'
def fd_CannyEdges(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(image, 100, 200).flatten()
    print(len(canny))
    return canny
# feature-descriptor-5: Countor Area'
def ContourArea(image):
    kernel = np.ones((3,3), np.uint8) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_1 = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)
    _,thresh = cv2.threshold(img_1,127,255,0)
    dil_erode = cv2.erode(thresh,kernel,2)
    image, contours, hierarchy  = cv2.findContours(dil_erode,1,2)
    ar = [i for i in range(len(contours))]
    for i in range(len(contours)):
        ar[i] = cv2.contourArea(contours[i])
        #print(ar[i])    
    ar.sort()
    print(ar[-2])
    return ar[-2]


 
train_labels = os.listdir(train_path)
print('train_labels--', train_labels)
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels          = []

# loop over the training data sub-folders
for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    print('dir---', dir)
    current_label = training_name
    print('current_label----', current_label)

    for x in range(0,images_per_class):
        file = dir + "/" + str(x) + ".jpg"
        print('file pathe--', file)
        image = cv2.imread(file)
        cv2.imshow('image--',image )
        cv2.waitKey(0)
        image = cv2.resize(image, fixed_size)

        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_canny = fd_CannyEdges(image)
        fv_cnt = ContourArea(image)
        #fv_histogram  = fd_histogram(image)

        global_feature = np.hstack([ fv_haralick, fv_hu_moments, fv_cnt])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

print("[STATUS] training Labels {}".format(np.array(labels).shape))

targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

#scaler            = MinMaxScaler(feature_range=(0, 1))
#rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(global_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")
