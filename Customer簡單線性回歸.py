import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv('Customers.csv') 

#查詢有無缺失資料
dataset.isna().sum()

x = dataset.iloc[:, 6].values 
y = dataset.iloc[:, 4].values 

X = x.reshape([-1 , 1])
Y = y.reshape([-1 , 1]) 

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(X , Y,test_size= 0.2,random_state = 0) 
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
#計算殘差 
residuals = y_test - y_pred 
#繪製殘差圖 
import seaborn as sns 
sns.residplot(x_test.flatten(), residuals.flatten(), lowess=True, color="g") 
plt.title("Residuals Plot") 
plt.xlabel("x") 
plt.ylabel("Residuals") 
#計算殘差的平均值、標準差、中位數等統計量 
residuals_mean = np.mean(residuals)
residuals_std = np.std(residuals) 
residuals_median = np.median(residuals) 
#計算殘差是否符合常態分析 P<0.05 
from scipy.stats import shapiro 
_, p_value = shapiro(residuals) 
print("Shapiro-Wilk normality test p-value:",p_value)
