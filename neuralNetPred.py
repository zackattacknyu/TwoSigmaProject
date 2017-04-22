import numpy as np
from sklearn import cross_validation
import xgboost as xgb
import time
import datetime
import csv
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave
from sklearn import datasets, linear_model
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D

folder = 'dataSets/'
trainX = np.load(folder + 'trainXarray.npy')
trainY = np.load(folder + 'trainYarray.npy')
testX = np.load(folder + 'testXarray.npy')
testID = np.load(folder + 'testIDarray.npy')

print(trainX.shape)

Yenc = np_utils.to_categorical(trainY-1, 3)
#Xnew = np.reshape(trainX,(trainX.shape[0],1,5))

trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(
    trainX, Yenc, random_state=42, stratify=Yenc,test_size=0.10)

inputImg = Input(shape=(5,))
layer1 = Dense(100, init='normal', activation='relu')(inputImg)
layer2 = Dense(20, init='normal', activation='sigmoid')(layer1)
outputLayer = Dense(3, init='normal', activation='softmax')(layer2)
noduleModel = Model(input=inputImg, output=outputLayer)
noduleModel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print("Now fitting Neural Network")
noduleModel.fit(trn_x, trn_y, batch_size=5000, nb_epoch=100,
                  verbose=1, validation_data=(val_x, val_y))
testPrediction = noduleModel.predict(testX)
def obtainPred(origPred):
    if(origPred<0):
        return 0
    elif(origPred>1):
        return 1
    else:
        return origPred



ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
fileName = 'submissions/twoSigmaSubmission_' + st + '.csv'

with open(fileName, 'w') as csvfile:
    fieldnames = ['listing_id','high','medium','low']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for ind in range(len(testID)):
        curPred1 = obtainPred(testPrediction[ind,0])
        curPred2 = obtainPred(testPrediction[ind,1])
        curPred3 = obtainPred(testPrediction[ind,2])
        totalPred = curPred1+curPred2+curPred3
        # outPred1 = curPred1/totalPred
        # outPred2 = curPred2/totalPred
        # outPred3 = curPred3/totalPred
        outPred1 = curPred1
        outPred2 = curPred2
        outPred3 = curPred3
        writer.writerow({'listing_id': str(testID[ind]), 'low': str(outPred1),
                         'medium' : str(outPred2), 'high' : str(outPred3)})
