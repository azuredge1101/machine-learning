import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('wine.csv')
#查詢有無缺失資料
dataset.isna().sum()

x = dataset.iloc[:, [4]].values
y = dataset.iloc[:, [10]].values

X = dataset.iloc[:, [0]].values
Y = dataset.iloc[:, [9]].values


# 補齊缺漏數值
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean", fill_value=None)
imputer = imputer.fit(x[:,:])
x[:,:] = imputer.transform(x[:,:])

imputer = imputer.fit(y[:,:])
y[:,:] = imputer.transform(y[:,:])

imputer = imputer.fit(X[:,:])
X[:,:] = imputer.transform(X[:,:])

imputer = imputer.fit(Y[:,:])
Y[:,:] = imputer.transform(Y[:,:])

# 建立訓練測試集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 建立線性回歸模型
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train) # 訓練模型

Regressor= LinearRegression()
Regressor.fit(X_train, Y_train) # 訓練模型

y_pred = regressor.predict(x_test) # 使用測試集x預測y值
Y_pred = Regressor.predict(X_test)

# 繪圖
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('chlorides VS alcohol(training set)')
plt.xlabel('chlorides')
plt.ylabel('alcohol')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, regressor.predict(x_test), color = 'blue')
plt.title('chlorides VS alcohol(testing set)')
plt.xlabel('chlorides')
plt.ylabel('alcohol')
plt.show()

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, Regressor.predict(X_train), color = 'blue')
plt.title('fixed acidity VS sulphates(training set)')
plt.xlabel('Fixed acidity')
plt.ylabel('Sulphates')
plt.show()

plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, Regressor.predict(X_test), color = 'blue')
plt.title('fixed acidity VS sulphates(testing set)')
plt.xlabel('Fixed acidity')
plt.ylabel('Sulphates')
plt.show()
