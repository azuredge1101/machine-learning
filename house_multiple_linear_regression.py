import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Downloads\house.csv')

x = dataset.iloc[:, [2,3,4,5,6,7,9]].values
y = dataset.iloc[:, [8]].values

# 補齊缺漏數值
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean", fill_value=None)
imputer = imputer.fit(x[:,0:6])
x[:,0:6] = imputer.transform(x[:,0:6])

imputer = imputer.fit(y[:,:])
y[:,:] = imputer.transform(y[:,:])

imputer2 = SimpleImputer(missing_values=np.nan, strategy="most_frequent", fill_value=None)
imputer2 = imputer2.fit(x[:, [6]])
x[:, [6]] = imputer2.transform(x[:, [6]])

# 編碼文字型態欄位
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()

x[:,6] = labelencoder_x.fit_transform(x[:,6])
ct = ColumnTransformer([('ocean_proximity', OneHotEncoder(), [6])], remainder='passthrough')
x = ct.fit_transform(x)

x = x[:, 1:]

# 建立訓練測試集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 建立線性回歸模型
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train) # 訓練模型

y_pred = regressor.predict(x_test) # 使用測試集x預測y值

# 反向淘汰
import statsmodels.api as sm
x_train = np.append(arr = np.ones((16512, 1)).astype(int), values = x_train, axis = 1)
x_opt = x_train[:, [0,1,2,3,4,5,6,7,8,9,10]]
x_opt = np.array(x_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()