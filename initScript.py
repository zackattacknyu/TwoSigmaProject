import json
import numpy as np

fileNtest = 'test.json'
fileNtrain = 'train.json'
json_data_test=open(fileNtest).read()
json_data_train=open(fileNtrain).read()

dataTrain = json.loads(json_data_train)
dataTest = json.loads(json_data_test)


#Initial try will only use the following fields:
#   Latitude, Longitude, Price, Bedrooms, Bathrooms
def obtainNParray(data,hasY):
    ind = 0
    numPts = len(data['listing_id'].keys())
    arrX = np.zeros((numPts, 5))
    arrY = np.zeros((numPts))
    arrListID = []
    for idKey in data['listing_id'].keys():
        arrX[ind, 0] = data['latitude'][idKey]
        arrX[ind, 1] = data['longitude'][idKey]
        arrX[ind, 2] = data['price'][idKey]
        arrX[ind, 3] = data['bedrooms'][idKey]
        arrX[ind, 4] = data['bathrooms'][idKey]
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
    return arrX,arrY,arrListID


trainX,trainY,trainIDs = obtainNParray(dataTrain,True)
testX,testY,testIDs = obtainNParray(dataTest,False)

np.save('trainXarray.npy',trainX)
np.save('trainYarray.npy',trainY)
np.save('testXarray.npy',testX)
np.save('testIDarray.npy',testIDs)
