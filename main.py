
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import time
from ydata_profiling import ProfileReport #pandas_profiling v2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('healthcare-dataset-stroke-data.csv')
data.info()

#there are unique values in the data. We should drop them because they are not useful for our analysis.
data.nunique()
data.drop(["id"], axis=1, inplace=True)

data.isnull().sum()

#Dealing with missing data using the mean.
data["bmi"].fillna(data["bmi"].mean(), inplace=True)
data.isnull().sum()

profile = ProfileReport(data, title='Stroke Prediction Report', html={'style' : {'full_width':True}})
profile.to_file(output_file="Stroke Prediction.html") 

#setting categorical features with cat_features variable
cat_features = [0,4,5,6,9]

#set dependent and indepedent variable.
X = data.drop(["stroke"], axis=1)
y = data["stroke"]

#Splitting data to train and test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Catboost
from catboost import CatBoostClassifier
clf = CatBoostClassifier()

start = time.time()

clf.fit(X_train, y_train, 
        cat_features=cat_features, 
        eval_set=(X_test, y_test), 
        verbose = 100
)

end = time.time()
diff = end - start

print('CatBoost model is fitted: ' + str(clf.is_fitted()))

y_pred = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy Score: ",acc_score)
print("Execution Time: ",diff)

for c in X.columns:
  col_type = X[c].dtype
  if col_type == 'object' or col_type.name == 'category' :
    X[c] = X[c].astype('category')
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
print('length of X_train and x_test: ', len(X_train), len(X_test))
print('length of y_train and y_test: ', len(y_train), len(y_test))


#Lighgtgbm
import lightgbm as lgb
clf = lgb.LGBMClassifier()
start = time.time()
clf.fit(X_train, y_train)
end = time.time()
diff = end - start

y_pred=clf.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy Score: ",acc_score)
print("Execution Time: ",diff)