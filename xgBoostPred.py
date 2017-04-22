import numpy as np
from sklearn import cross_validation
import xgboost as xgb
import time
import datetime
import csv

folder = 'dataSets/'
trainX = np.load(folder + 'trainXarray.npy')
trainY = np.load(folder + 'trainYarray.npy')
testX = np.load(folder + 'testXarray.npy')
testID = np.load(folder + 'testIDarray.npy')

def getBinaryArray(array,num):
    return (array == num).astype('int')

print(trainX.shape)


trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(
    trainX, trainY, random_state=42, stratify=trainY,test_size=0.20)

trn_y1 = getBinaryArray(trn_y,1)
trn_y2 = getBinaryArray(trn_y,2)
trn_y3 = getBinaryArray(trn_y,3)
val_y1 = getBinaryArray(val_y,1)
val_y2 = getBinaryArray(val_y,2)
val_y3 = getBinaryArray(val_y,3)

def getPrediction(trainY,validationY):
    clf = xgb.XGBRegressor(max_depth=10,
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

def obtainPred(array,index):
    pred = array[index]
    if(pred<0):
        return 0
    elif(pred>1):
        return 1
    else:
        return pred

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d__%H_%M_%S')
fileName = 'submissions/twoSigmaSubmission_' + st + '.csv'

with open(fileName, 'w') as csvfile:
    fieldnames = ['listing_id','high','medium','low']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for ind in range(len(testID)):
        curPred1 = obtainPred(pred1,ind)
        curPred2 = obtainPred(pred2,ind)
        curPred3 = obtainPred(pred3,ind)
        totalPred = curPred1+curPred2+curPred3
        # outPred1 = curPred1/totalPred
        # outPred2 = curPred2/totalPred
        # outPred3 = curPred3/totalPred
        outPred1 = curPred1
        outPred2 = curPred2
        outPred3 = curPred3
        writer.writerow({'listing_id': str(testID[ind]), 'low': str(outPred1),
                         'medium' : str(outPred2), 'high' : str(outPred3)})
