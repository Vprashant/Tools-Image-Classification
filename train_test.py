#-----------------------------------
#MODEL TRAING FOR  
#TRAINING OUR MODEL
#-----------------------------------
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import cv2
import warnings
import mahotas
from sklearn import tree
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score



warnings.filterwarnings('ignore')

#--------------------
# tunable-parameters
#--------------------
num_trees = 50
test_size = 0.20
seed      = 4
train_path = "dataset/train"
test_path  = "dataset/test"
h5_data    = 'output/data.h5'
h5_labels  = 'output/labels.h5'
scoring    = "accuracy"
bins= 8
#----------------------------------
# FEATURE DESCRIPTOR-1: Hu Moments
#----------------------------------
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
#--------------------------------------
# FEATURE DESCRIPTOR-2 :HARLICK TEXTURE
#--------------------------------------
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)

# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed)))

# variables to hold the results and names
results = []
names   = []

# import the feature vector and trained labels
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)
print('global_labels--', global_labels)
h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features), np.array(global_labels), test_size=test_size, random_state=seed)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print('trainDataGlobal',trainDataGlobal)
print("Test data   : {}".format(testDataGlobal.shape))
print('testDataGlobal',testDataGlobal)
print("Train labels: {}".format(trainLabelsGlobal.shape))
print('trainLabelsGlobal',trainLabelsGlobal)
print("Test labels : {}".format(testLabelsGlobal.shape))

'''
# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
'''

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics

clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
#clf = svm.SVC(kernel='linear')
clf.fit(trainDataGlobal, trainLabelsGlobal)


filename = 'trained_model.sav'
joblib.dump(clf, filename)

#check the test data prediction...................

for file in glob.glob(test_path + "/*.jpg"):
    
    image = cv2.imread(file)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    #image = cv2.resize(image, (120, 120))

    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)

      
    global_feature = np.hstack([fv_haralick, fv_hu_moments])
    reshape_feature = global_feature.reshape(1,-1)
    loaded_model = joblib.load(filename)
    prediction = loaded_model.predict(reshape_feature)[0]
 
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    # # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
