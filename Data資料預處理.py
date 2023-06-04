#提供矩陣數學運算方法
import numpy as np

#提供畫圖方法
import matplotlib.pyplot as plt

#提供讀取資料集的方法
import pandas as pd

#讀取檔案
dataset = pd.read_csv('Data.csv')

#查詢有無缺失資料
dataset.isna().sum()

#自變量和應變量的值
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#missing data 缺失資料處理
from sklearn.impute import SimpleImputer

#用平均值(mean)來用missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean",fill_value=None)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#分類型資料欄位用LabelEncoder標籤編碼、OneHotEncoder虛擬編碼
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#提供虛擬編碼處理的方法
from sklearn.compose import ColumnTransformer

#Categorical data 類別型資料
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])

#使用ColumnTransformer方法進行虛擬編碼
ct=ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder='passthrough')

#轉換成array陣列
X = ct.fit_transform(x)

#做標籤編碼
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#分割訓練集合與測試集合
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0 )

#標準化
from sklearn.preprocessing import StandardScaler

#特徵縮放
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

