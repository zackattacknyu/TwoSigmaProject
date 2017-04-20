import numpy as np
from sklearn import cross_validation
import xgboost as xgb


folder = 'dataSets/'
trainX = np.load(folder + 'trainXarray.npy')
trainY = np.load(folder + 'trainYarray.npy')
testX = np.load(folder + 'testXarray.npy')
testID = np.load(folder + 'testIDarray.npy')

print(trainX.shape)


trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(
    trainX, trainY, random_state=42, stratify=trainY,test_size=0.20)

clf = xgb.XGBRegressor(max_depth=10,
                           n_estimators=1500,
                           min_child_weight=9,
                           learning_rate=0.05,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True,
        eval_metric='mlogloss', early_stopping_rounds=100)

pred2 = clf.predict(testX,output_margin=True)
np.save('tempPred.npy',pred2)