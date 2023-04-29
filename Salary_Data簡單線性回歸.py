import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x , y,test_size= 0.2,random_state = 0)

##線性回歸 LinearRegression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
#預測結果
y_pred = regressor.predict(x_test)
#繪圖
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color = 'blue')
plt.title('Salary VS Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#test圖
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color = 'blue')
plt.title('Salary VS Experience (testing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
