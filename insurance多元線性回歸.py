import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('insurance.csv')
#查詢有無缺失資料
dataset.isna().sum()

X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, 6].values

from sklearn.impute import SimpleImputer

#sex 和 smoker 為分類型欄位，用出現頻率代替空白
imputer =SimpleImputer(missing_values=np.nan,strategy="most_frequent",fill_value=None)
imputer = imputer.fit(X[:,1:2])
X[:,1:2]= imputer.transform(X[:,1:2])
imputer = imputer.fit(X[:,4:5])
X[:,4:5]= imputer.transform(X[:,4:5])

#age﹑bmi﹑childern 為連續型欄位，用平均代替空白
imputer =SimpleImputer(missing_values=np.nan,strategy="mean",fill_value=None)
imputer = imputer.fit(X[:,0:1])
X[:,0:1] = imputer.transform(X[:,0:1])
imputer = imputer.fit(X[:,2:4])
X[:,2:4] = imputer.transform(X[:,2:4])

#charges 為連續型欄位，用平均代替空白
y=np.reshape(y,(-1, 1))
imputer = imputer.fit(y[:,0:1])
y[:,0:1] = imputer.transform(y[:,0:1])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X= LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
ct=ColumnTransformer([("sex", OneHotEncoder(), [1])],
remainder='passthrough')
X = ct.fit_transform(X)

# 避免共線性，忽略第一行
X = X[:,1:]
labelencoder_x = LabelEncoder()
X[:,4] = labelencoder_X.fit_transform(X[:,4])
ct=ColumnTransformer([("smoker", OneHotEncoder(), [4])] ,
remainder='passthrough')
X = ct.fit_transform(X)
X=X[:,1:]

#分割資料
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,
random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

#淘汰資料欄位
import statsmodels.api as sm
X_train=np.append(arr=np.ones((1070,1)).astype(int),values=X_train,axis=1)
X_opt=X_train[:,[0,1,2,3,4,5]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=y_train,exog=X_opt).fit()
regressor_OLS.summary()

#去掉地區第 2 行
X_opt=X_train[:,[0,1,3,4,5]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=y_train,exog=X_opt).fit()
regressor_OLS.summary()

# 去掉地區第 5 行
X_opt=X_train[:,[0,1,3,4]]
X_opt=np.array(X_opt,dtype=float)
regressor_OLS=sm.OLS(endog=y_train,exog=X_opt).fit()
regressor_OLS.summary()

