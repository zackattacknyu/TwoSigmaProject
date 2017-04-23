import numpy as np
from sklearn import cross_validation
import xgboost as xgb
import time
import datetime
import csv
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

folder = 'dataSets/'
trainX = np.load(folder + 'trainXarray4_5encoded.npy')
trainY = np.load(folder + 'trainYarray.npy')
testX = np.load(folder + 'testXarray4_5encoded.npy')
testID = np.load(folder + 'testIDarray.npy')

numFeatures = trainX.shape[1]

def getBinaryArray(array,num):
    return (array == num).astype('int')

print(trainX.shape)

Yenc = np_utils.to_categorical(trainY-1, 3)
trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(
    trainX, trainY, random_state=42, stratify=trainY,test_size=0.10)
trn_xA, val_xA, trn_yA, val_yA = cross_validation.train_test_split(
    trainX, Yenc, random_state=42, stratify=Yenc,test_size=0.10)

trn_y1 = getBinaryArray(trn_y,1)
trn_y2 = getBinaryArray(trn_y,2)
trn_y3 = getBinaryArray(trn_y,3)
val_y1 = getBinaryArray(val_y,1)
val_y2 = getBinaryArray(val_y,2)
val_y3 = getBinaryArray(val_y,3)

def getPrediction(trainY,validationY):
    clf = xgb.XGBRegressor(max_depth=15,
                               n_estimators=1500,
                               min_child_weight=9,
                               learning_rate=0.05,
                               nthread=8,
                               subsample=0.80,
                               colsample_bytree=0.80,
                               seed=4242)
    clf.fit(trn_x, trainY, eval_set=[(val_x, validationY)], verbose=True,
            eval_metric='logloss', early_stopping_rounds=100)
    return clf.predict(testX,output_margin=True)


pred1 = getPrediction(trn_y1,val_y1)
pred2 = getPrediction(trn_y2,val_y2)
pred3 = getPrediction(trn_y3,val_y3)

def getNNPrediction():
    inputImg = Input(shape=(numFeatures,))
    layer1 = Dense(100, init='normal', activation='relu')(inputImg)
    layer2 = Dense(20, init='normal', activation='sigmoid')(layer1)
    outputLayer = Dense(3, init='normal', activation='softmax')(layer2)
    noduleModel = Model(input=inputImg, output=outputLayer)
    noduleModel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print("Now fitting Neural Network")
    noduleModel.fit(trn_xA, trn_yA, batch_size=5000, nb_epoch=100,
                      verbose=1, validation_data=(val_xA, val_yA))
    return noduleModel.predict(testX)

def obtainPred(origPred):
    if(origPred<0):
        return 0
    elif(origPred>1):
        return 1
    else:
        return origPred

def ensembleNumbers(predA,normA,predB,normB):
    #NOTE: BEST RESULTS OBTAINED WITH THIS 50/50 SPLIT
    pred = 0.5*predA/normA + 0.5*predB/normB
    return obtainPred(pred)

testPrediction = getNNPrediction()
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
fileName = 'submissions/twoSigmaSubmission_' + st + '.csv'

with open(fileName, 'w') as csvfile:
    fieldnames = ['listing_id','high','medium','low']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for ind in range(len(testID)):
        curPred1A = pred1[ind]
        curPred2A = pred2[ind]
        curPred3A = pred3[ind]

        curPred1B = testPrediction[ind, 0]
        curPred2B = testPrediction[ind, 1]
        curPred3B = testPrediction[ind, 2]

        totalPredA = curPred1A+curPred2A+curPred3A
        totalPredB = curPred1B + curPred2B + curPred3B
        # outPred1 = curPred1/totalPred
        # outPred2 = curPred2/totalPred
        # outPred3 = curPred3/totalPred
        outPred1 = ensembleNumbers(curPred1A,totalPredA,curPred1B,totalPredB)
        outPred2 = ensembleNumbers(curPred2A, totalPredA, curPred2B, totalPredB)
        outPred3 = ensembleNumbers(curPred3A, totalPredA, curPred3B, totalPredB)
        writer.writerow({'listing_id': str(testID[ind]), 'low': str(outPred1),
                         'medium' : str(outPred2), 'high' : str(outPred3)})
