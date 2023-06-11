import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('wine.csv')

#查詢有無缺失資料
dataset.isna().sum()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values

# 補齊缺漏數值
from sklearn.impute import SimpleImputer

imputer =SimpleImputer(missing_values=np.nan,strategy="most_frequent",fill_value=None)
y=np.reshape(y,(-1, 1))
imputer = imputer.fit(y[:,0:1])
y[:,0:1] = imputer.transform(y[:,0:1])
imputer =SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None)
imputer = imputer.fit(X[:,0:11])
X[:,0:11] = imputer.transform(X[:,0:11])

# 建立訓練測試集合
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

#特徵縮放
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)
#建立邏輯回歸模型
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
#比對y_pred和y_test的資料的差異
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


