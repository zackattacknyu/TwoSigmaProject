import json
import numpy as np
import time
import math
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

fileNtest = 'test.json'
fileNtrain = 'train.json'
json_data_test=open(fileNtest).read()
json_data_train=open(fileNtrain).read()

dataTrain = json.loads(json_data_train)
dataTest = json.loads(json_data_test)

def getList(dataArray,fieldKey):
    return [dataArray[fieldKey][key1] for key1 in dataArray['listing_id'].keys()]

def getDictionaryForField(fieldKey):
    listTrain = getList(dataTrain, fieldKey)
    listTest = getList(dataTest, fieldKey)
    return getDictionary(listTrain,listTest)

def getFeatureList(curSet,dataArray):
    #dInd = 0
    #totalNum = len(dataArray['listing_id'].keys())
    for key1 in dataArray['listing_id'].keys():
        #dInd = dInd + 1
        #print('Feat List for Apt ' + str(dInd) + ' of ' + str(totalNum))
        for featureNm in dataArray['features'][key1]:
            curSet.add(featureNm)
    return curSet

def getDictionary(listTest,listTrain):
    allItems = listTest + listTrain
    allIDs = set(allItems)
    return getDictionary2(allIDs)

def getDictionary2(allIDs):
    return dict(zip(allIDs, range(len(allIDs))))  # maps the manager ids to numbers

def getFeaturesDictionary():
    outputSet = set()
    trainSet = getFeatureList(outputSet,dataTrain)
    allItems = getFeatureList(trainSet,dataTest)
    return getDictionary2(allItems)

print('Obtaining Apt Features')
allAptFeatures = getFeaturesDictionary()
numAptFeatures = len(allAptFeatures)
print('Finished obtain Feature List dictionary')

def getBinaryVectorOfApartmentFeatures(dataArray,key1):
    outputVector = np.zeros((numAptFeatures))
    for featureName in dataArray['features'][key1]:
        numericIndex = allAptFeatures[featureName]
        if not math.isnan(numericIndex) and not math.isinf(numericIndex):
            outputVector[numericIndex]=1
    return outputVector

print("now obtaining manager dictionary")
#there are 3481 unique managers of ~50,000 training example
managerDictionary = getDictionaryForField('manager_id')

#there are 7585 unique buildings of
print("now obtaining building dictionary")
buildingDictionary = getDictionaryForField('building_id')

def convertToTimeSinceEpoch(timeStr):
    format1 = '%Y-%m-%d %H:%M:%S'
    return time.mktime(time.strptime(timeStr,format1))

#Initial try will only use the following fields:
#   Latitude, Longitude, Price, Bedrooms, Bathrooms
def obtainArrays(data,hasY):
    ind = 0
    numPts = len(data['listing_id'].keys())
    arrY = np.zeros((numPts))
    arrListID = []
    for idKey in data['listing_id'].keys():
        arrListID.append(data['listing_id'][idKey])
        ind = ind+1
    if hasY:
        ind=0
        for idKey in data['listing_id'].keys():
            curYtext = data['interest_level'][idKey]
            if(curYtext=='low'):
                arrY[ind] = 1
            elif(curYtext=='medium'):
                arrY[ind]=2
            else:
                arrY[ind]=3
            ind = ind + 1
    return arrY,arrListID

numFeatures=9
def obtainXarray(data,binaryArray):
    ind = 0
    numPts = len(data['listing_id'].keys())
    numEncFeatures = binaryArray.shape[1]
    numTotFeatures = numFeatures + numEncFeatures
    arrX = np.zeros((numPts, numTotFeatures))
    for idKey in data['listing_id'].keys():
        print("Obtaining X features for apt " + str(ind) + " of " + str(numPts))
        arrX[ind, 0] = data['latitude'][idKey]
        arrX[ind, 1] = data['longitude'][idKey]
        arrX[ind, 2] = data['price'][idKey]
        arrX[ind, 3] = data['bedrooms'][idKey]
        arrX[ind, 4] = data['bathrooms'][idKey]
        arrX[ind, 5] = len(data['photos'][idKey]) #make number of photos a feature
        arrX[ind, 6] = managerDictionary[data['manager_id'][idKey]] #manager id (1-3481) is feature
        arrX[ind, 7] = buildingDictionary[data['building_id'][idKey]] #building id (1-7585) is feature
        arrX[ind, 8] = convertToTimeSinceEpoch(data['created'][idKey])
        arrX[ind, 9:numTotFeatures] = binaryArray[ind,:]
        ind = ind+1
    return arrX

def obtainBinaryAptFeatureArray(data):
    ind = 0
    numPts = len(data['listing_id'].keys())
    arrX = np.zeros((numPts, numAptFeatures))
    for idKey in data['listing_id'].keys():
        print("Obtaining apt features for apt " + str(ind) + " of " + str(numPts))
        vect = getBinaryVectorOfApartmentFeatures(data,idKey)
        #print(vect[0:10])
        arrX[ind, 0:numAptFeatures] = vect
        ind = ind+1
    return arrX

trainY,trainIDs = obtainArrays(dataTrain,True)
testY,testIDs = obtainArrays(dataTest,False)

numTrain = len(dataTrain['listing_id'].keys())
numTest = len(dataTest['listing_id'].keys())
numTotal = numTrain + numTest

binaryArrayTrain = obtainBinaryAptFeatureArray(dataTrain)
binaryArrayTest = obtainBinaryAptFeatureArray(dataTest)
binaryArrayTotal = np.concatenate((binaryArrayTrain,binaryArrayTest),axis=0)

print("Now doing AutoEncoding")
encoding_dim=5
inputImg = Input(shape=(numAptFeatures,))
layer1 = Dense(encoding_dim, init='normal', activation='relu')(inputImg)
outputLayer = Dense(numAptFeatures, init='normal', activation='sigmoid')(layer1)
autoEncoder = Model(input=inputImg, output=outputLayer)
autoEncoder.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
autoEncoder.fit(binaryArrayTotal,binaryArrayTotal,batch_size=1000,nb_epoch=20,verbose=1)
compressedRep = Model(input=inputImg,output=layer1)
binaryArrayEncoded = compressedRep.predict(binaryArrayTotal)

newBinaryTrain = binaryArrayEncoded[0:numTrain,:]
newBinaryTest = binaryArrayEncoded[numTrain:numTotal,:]

trainX = obtainXarray(dataTrain,newBinaryTrain)
testX = obtainXarray(dataTest,newBinaryTest)

print("Number of Apt Features in Vector: " + str(numAptFeatures))

allX = np.zeros((numTotal,trainX.shape[1]))
allX[0:numTrain,:] = trainX
allX[numTrain:numTotal,:]=testX
allXFeatureRange = np.max(allX,axis=0)-np.min(allX,axis=0)
allXZeroMin = allX-np.min(allX,axis=0)
allXNormed = allXZeroMin/allXFeatureRange
trainXNormed = allXNormed[0:numTrain]
testXNormed = allXNormed[numTrain:numTotal]

np.save('dataSets/trainXarray4_5encoded.npy',trainXNormed)
np.save('dataSets/trainYarray.npy',trainY)
np.save('dataSets/testXarray4_5encoded.npy',testXNormed)
np.save('dataSets/testIDarray.npy',testIDs)
