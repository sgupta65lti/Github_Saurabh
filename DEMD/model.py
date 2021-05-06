import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
from imblearn.under_sampling import RandomUnderSampler
import pickle

# reading the file of training data
train = pd.read_csv('train.csv')
y = train.Response.values

# data pre-processing

train1=pysqldf('''select ID,Holding_Policy_Duration, case when Holding_Policy_Duration in ('1.0','2.0') then '1-2' when Holding_Policy_Duration in ('3.0','4.0') then '3-4' 
when Holding_Policy_Duration in ('5.0','6.0','7.0') then '5-7' 
when Holding_Policy_Duration in ('8.0','9.0','10.0','11.0','12.0','13.0','14.0') then '8-14'
else Holding_Policy_Duration end as new  from train''')
train=train.merge(train1, on = 'ID') # merging the modified field of duration back to original data
train.rename(columns={'new':'Holding_Policy_Duration'},inplace=True)
train.drop(['Holding_Policy_Duration_x','Holding_Policy_Duration_y'],axis=1,inplace=True)

train = train.fillna("Missing") # encoding all the nans with 'Missing'

for f in train.columns:
    if train[f].dtype=='object':
        # print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values))
        train[f] = lbl.transform(list(train[f].values))

train.drop(['Reco_Policy_Premium','Response'],axis=1,inplace=True)
train.set_index('ID',inplace=True)

# feature selection ## selecting 4 best features

chi2_features = SelectKBest(chi2, k = 4) 
X_kbest_features = chi2_features.fit(train, y)
X=train.loc[:][train.columns[X_kbest_features.get_support()]]
# print(X.columns) #['Region_Code', 'Reco_Insurance_Type', 'Upper_Age', 'Reco_Policy_Cat']

## Model building with XGBoost Classifier since the data is imbalanced with some values as missing for few variables
clf = xgb.XGBClassifier(n_estimators=130,
                        max_depth=20,
                        learning_rate=0.09,
                        subsample=0.8,
                        verbosity=0,
                       tree_method='gpu_hist')
xgb_model = clf.fit(X.values,y, eval_metric="auc")
preds=clf.predict_proba(X.values)[:,1]
print(roc_auc_score(y,preds)) #0.826968245125282
# decent enough roc_auc score for imbalanced data

# Creating an under sampled dataset to check the model performance with higher weightage to less occuring class

rus = RandomUnderSampler(random_state=42,sampling_strategy=0.95) ## assigning higher weight to class with less values
X_res, y_res = rus.fit_resample(X, y)

preds_rus=clf.predict_proba(X_res.values)[:,1]
print(roc_auc_score(y_res,preds_rus)) # 0.8267141236133757

# well a good and consistent score on under sampled data for handling imbalance class
print(np.array(X.values)[2])
print(clf.predict_proba(np.array([[3732, 0 ,32,19]]))[:,1])

pickle_out = open("classifier.pkl","wb")
pickle.dump(clf, pickle_out)
pickle_out.close()