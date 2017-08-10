# Parameters
XGB_WEIGHT = 0.6500
BASELINE_WEIGHT = 0.0056

BASELINE_PRED = 0.0115

# version 61
#   Drop fireplacecnt and fireplaceflag, following Jayaraman:
#     https://www.kaggle.com/valadi/xgb-w-o-outliers-lgb-with-outliers-combo-tune5

# version 60
#   Try BASELINE_PRED=0.0115, since that's the actual baseline from
#     https://www.kaggle.com/aharless/oleg-s-original-better-baseline

# version 59
#   Looks like 0.0056 is the optimum BASELINE_WEIGHT

# versions 57, 58
#   Playing with BASELINE_WEIGHT parameter:
#     3 values will determine quadratic approximation of optimum

# version 55
#   OK, it doesn't get the same result, but I also get a different result
#     if I fork the earlier version and run it again.
#   So something weird is going on (maybe software upgrade??)
#   I'm just going to submit this version and make it my new benchmark.

# version 53
#   Re-parameterize ensemble (should get same result).

# version 51
#   Quadratic approximation based on last 3 submissions gives 0.3533
#     as optimal lgb_weight.  To be slightly conservative,
#     I'm rounding down to 0.35

# version 50
#   Quadratic approximation based on last 3 submissions gives 0.3073
#     as optimal lgb_weight

# version 49
#   My latest quadratic approximation is concave, so I'm just taking
#     a shot in the dark with lgb_weight=.3

# version 45
#   Increase lgb_weight to 0.25 based on new quadratic approximation.
#   Based on scores for versions 41, 43, and 44, the optimum is 0.261
#     if I've done the calculations right.
#   I'm being conservative and only going 2/3 of the way there.
#   (FWIW my best guess is that even this will get a worse score,
#    but you gotta pay some attention to the math.)

# version 44
#   Increase lgb_weight to 0.23, per Nikunj's suggestion, even though
#     my quadratic approximation said I was already at the optimum

# verison 43
#   Higher lgb_weight, so I can do a quadratic approximation of the optimum

# version 42
#   The answer to the ultimate question of life, the universe, and everything
#     comes down to a slightly higher lgb_weight

# version 41
#   Trying Nikunj's suggestion of imputing missing values.

# version 39
#   Trying higher lgb_weight again but with old learning rate.
#   The new one did better with LGB only but makes the combination worse.

# version 38
#   OK back to baseline 0.2 weight

# version 37
#   Looks like increasing lgb_weight was better

# version 34
#   OK, try reducing lgb_weight instead

# version 32
#   Increase lgb_weight because LGB performance has imporved more than XGB
#   Increase learning rate for LGB: 0029 is compromise;  CV prefers 0033
#     (and reallly would prefer more boosting rounds with old value instead
#      but constaints on running time are getting hard)

# Version 27:
#   Control LightGBM's loquacity

# Version 26:
# Getting rid of the LightGBM validation, since this script doesn't use the result.
# Now use all training data to fit model.
# I have a separate script for validation:
#    https://www.kaggle.com/aharless/lightgbm-outliers-remaining-cv


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc


##### READ IN RAW DATA

print( "\nReading data from disk ...")
prop = pd.read_csv('../properties_2016.csv')
train = pd.read_csv("../train_2016_v2.csv")



##### PROCESS DATA FOR LIGHTGBM

print( "\nProcessing data for LightGBM ..." )
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')
df_train.fillna(df_train.median(),inplace = True)

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)


train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

x_train = x_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(x_train, label=y_train)



##### RUN LIGHTGBM

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.5      # feature_fraction -- OK, back to .5, but maybe later increase this
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0

print("\nFitting LightGBM model ...")
clf = lgb.train(params, d_train, 430)

del d_train; gc.collect()
del x_train; gc.collect()

print("\nPrepare for LightGBM prediction ...")
print("   Read sample file ...")
sample = pd.read_csv('../sample_submission.csv')
print("   ...")
sample['parcelid'] = sample['ParcelId']
print("   Merge with property data ...")
df_test = sample.merge(prop, on='parcelid', how='left')
print("   ...")
del sample, prop; gc.collect()
print("   ...")
x_test = df_test[train_columns]
print("   ...")
del df_test; gc.collect()
print("   Preparing x_test...")
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
print("   ...")
x_test = x_test.values.astype(np.float32, copy=False)

print("\nStart LightGBM prediction ...")
# num_threads > 1 will predict very slow in kernal
clf.reset_parameter({"num_threads":1})
p_test = clf.predict(x_test)

del x_test; gc.collect()

print( "\nUnadjusted LightGBM predictions:" )
print( pd.DataFrame(p_test).head() )



##### RE-READ PROPERTIES FILE
##### (I tried keeping a copy, but the program crashed.)

print( "\nRe-reading properties file ...")
properties = pd.read_csv('../properties_2016.csv')



##### PROCESS DATA FOR XGBOOST

print( "\nProcessing data for XGBoost ...")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)
# shape
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.418 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))



##### RUN XGBOOST

print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,
    'alpha': 0.4,
    'base_score': y_mean,
    'silent': 1
}
# Enough with the ridiculously overfit parameters.
# I'm going back to my version 20 instead of copying Jayaraman.
# I want a num_boost_rounds that's chosen by my CV,
# not one that's chosen by overfitting the public leaderboard.
# (There may be underlying differences between the train and test data
#  that will affect some parameters, but they shouldn't affect that.)

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

# cross-validation
#print( "Running XGBoost CV ..." )
#cv_result = xgb.cv(xgb_params,
#                   dtrain,
#                   nfold=5,
#                   num_boost_round=350,
#                   early_stopping_rounds=50,
#                   verbose_eval=10,
#                   show_stdv=False
#                  )
#num_boost_rounds = len(cv_result)

# num_boost_rounds = 150
num_boost_rounds = 242
print("\nXGBoost tuned with CV in:")
print("   https://www.kaggle.com/aharless/xgboost-without-outliers-tweak ")
print("num_boost_rounds="+str(num_boost_rounds))

# train model
print( "\nTraining XGBoost ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost ...")
xgb_pred = model.predict(dtest)

print( "\nXGBoost predictions:" )
print( pd.DataFrame(xgb_pred).head() )



##### COMBINE PREDICTIONS

print( "\nCombining XGBoost, LightGBM, and baseline predicitons ..." )
lgb_weight = 1 - XGB_WEIGHT - BASELINE_WEIGHT
pred = XGB_WEIGHT*xgb_pred + BASELINE_WEIGHT*BASELINE_PRED + lgb_weight*p_test

print( "\nCombined predictions:" )
print( pd.DataFrame(pred).head() )



##### WRITE THE RESULTS

print( "\nPreparing results for write ..." )
y_pred=[]

for i,predict in enumerate(pred):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
from datetime import datetime

print( "\nWriting results to disk ..." )
output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ..." )