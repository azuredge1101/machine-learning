import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('wine.csv')
#查詢有無缺失資料
dataset.isna().sum()

x = data.loc[:, ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']].values
y = data.loc[:, ['quality']].values

# 補齊缺漏數值
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean", fill_value=None)
imputer = imputer.fit(x[:,:])
x[:,:] = imputer.transform(x[:,:])

imputer2 = SimpleImputer(missing_values=np.nan, strategy="most_frequent", fill_value=None)
imputer2 = imputer.fit(y[:,:])
y[:,:] = imputer2.transform(y[:,:])


# 建立訓練測試集合
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 建立邏輯迴歸模型
from sklearn.linear_model import LogisticRegression 
classifier= LogisticRegression(multi_class='auto', solver='newton-cg',random_state=0)
classifier.fit(x_train, y_train.astype('int')) # 訓練模型

# 測試集預測值
y_pred = classifier.predict(x_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
