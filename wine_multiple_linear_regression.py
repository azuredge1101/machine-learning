import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('wine.csv')
#查詢有無缺失資料
dataset.isna().sum()

x = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, [11]].values

# 補齊缺漏數值
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean", fill_value=None)
imputer = imputer.fit(x[:,:])
x[:,:] = imputer.transform(x[:,:])

imputer = imputer.fit(y[:,:])
y[:,:] = imputer.transform(y[:,:])

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
x_train = np.append(arr = np.ones((1279, 1)).astype(int), values = x_train, axis = 1)
x_opt = x_train[:, [0,1,2,3,4,5,6,7,8,9,10,11]]
x_opt = np.array(x_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x_train[:, [0,1,2,3,4,5,6,7,9,10,11]]
x_opt = np.array(x_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x_train[:, [0,2,3,4,5,6,7,9,10,11]]
x_opt = np.array(x_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x_train[:, [0,2,3,5,6,7,9,10,11]]
x_opt = np.array(x_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x_train[:, [0,2,5,6,7,9,10,11]]
x_opt = np.array(x_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()
