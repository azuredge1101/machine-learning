import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('house.csv')
#查詢有無缺失資料
dataset.isna().sum()

x = dataset.iloc[:, [7]].values
y = dataset.iloc[:, [8]].values

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

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('median_income VS median_house_value(training set)')
plt.xlabel('Median_income')
plt.ylabel('Median_house_value')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, regressor.predict(x_test), color = 'blue')
plt.title('median_income VS median_house_value(testing set)')
plt.xlabel('Median_income')
plt.ylabel('Median_house_value')
plt.show()
