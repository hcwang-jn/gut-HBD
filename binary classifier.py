import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from sklearn.model_selection import cross_val_score, RepeatedKFold, RepeatedStratifiedKFold


d=pd.read_csv("metadata.csv")
labs=[]
for i in range(len(d)):
    labs.append(d.iloc[i,1])
labs=pd.DataFrame(labs)
labs
X = np.array(d.iloc[:,2:])
y = np.array(labs)
# Pre
def format_check_np_ndarray(data):
    if type(data) == pd.core.frame.DataFrame:
        return data.values
    elif type(data) != np.ndarray:
        print('Input format must be pandas dataframe or numpy array')
    else:
        return data

def format_check_np_ndarray_ravel(data):
    if type(data) == pd.core.frame.DataFrame:
        if data.shape[1] > 1:
            print('Dim Error')
        else:
            return data.values.ravel()

    elif type(data) == pd.core.series.Series:
        return np.array(data)

    elif type(data) != np.ndarray:
        print('Input format must be pandas dataframe or numpy array')
        
    else:
        return data.ravel()

lab = labs
enc = LabelEncoder()
labe = enc.fit_transform(lab.values)
labe

model = LGBMClassifier()
cv = RepeatedStratifiedKFold(n_repeats=10, n_splits=4)
cross_val_score(model, d.iloc[:,2:].values, labe, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
cross_val_score(model, d.iloc[:,2:].values, labe, cv=cv, scoring='roc_auc', n_jobs=-1)
cross_val_score(model, d.iloc[:,2:].values, labe, cv=cv, scoring='roc_auc', n_jobs=-1).mean()

model = LGBMRegressor()
cv = RepeatedKFold(n_repeats=10, n_splits=5)
cross_val_score(model, d.iloc[:,2:].values, labe, cv=cv, scoring='r2', n_jobs=-1).mean()


models = [KNeighborsClassifier(), SVC(probability=True,random_state=0), DecisionTreeClassifier(random_state=0),
          RandomForestClassifier(random_state =0), GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0), XGBClassifier(use_label_encoder=False, n_jobs=1,random_state =0), 
          XGBRFClassifier(use_label_encoder=False, n_jobs=1,random_state =0),LGBMClassifier(random_state =0),RandomForestClassifier(n_estimators=50,criterion='entropy',random_state =0)]
names = ['KNN', 'SVM', 'DT', 'RF', 'GB', 'XGB', 'XGBRF', 'LGB']
cv = RepeatedStratifiedKFold(n_repeats=10, n_splits=5)

aucs = []
for i,n in zip(names, models):
    print(i)
    auc = cross_val_score(n, d.iloc[:,2:].values, labe, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    aucs.append(auc)

pd.DataFrame(aucs, index=names) 


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier, XGBRFClassifier

otu_table = pd.read_csv('otu.csv', index_col=0)
labels = pd.read_csv('metadata.csv', index_col=0) 
otu_table.columns[1:]
labels1= LabelEncoder().fit_transform(labels['Group'])
labels1

from sklearn.model_selection import cross_val_score, KFold
import numpy as np

X = otu_table.iloc[:,1:]
y = labels1 # 标签
rf = XGBClassifier(n_estimators=100, random_state=42)
n_repeats = 10
n_splits = 5

all_scores = []
all_importances = []
for i in range(n_repeats):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=i)
    scores = []
    importances = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        rf.fit(X_train, y_train)
        scores.append(rf.score(X_test, y_test))
        importances.append(rf.feature_importances_)
    all_scores.append(scores)
    all_importances.append(importances)

all_scores = np.array(all_scores)
all_importances = np.array(all_importances)

print("Cross-validation scores:")
print(np.mean(all_scores, axis=0))
print(np.std(all_scores, axis=0))
print("\nFeature importances:")
print(np.mean(all_importances, axis=0))
print(np.std(all_importances, axis=0))
aa=np.mean(all_importances, axis=0)
bb=list(np.mean(aa, axis=0))
    
feature_importance=pd.DataFrame({'index':otu_table.columns[1:],'feature_importance':bb})
A=feature_importance.sort_values(by=['feature_importance'],ascending=False)

A.iloc[:100].to_csv("feature.csv",index=False)





