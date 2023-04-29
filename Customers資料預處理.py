import numpy as np
import pandas as pd

dataset = pd.read_csv('Customers.csv')

x = dataset.iloc[:,[1,2,3,5,6,7]].values
y = dataset.iloc[:, 4].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=0, strategy='mean',fill_value= None)
imputer = imputer.fit(x[ :,1:3])
x[: ,1:3] = imputer.transform(x[: ,1:3])


imputer2 = SimpleImputer(missing_values=np.nan, strategy="most_frequent", fill_value=None)
imputer2 = imputer2.fit(x[:,3:4])
x[:,3:4] = imputer2.transform(x[:,3:4])

imputer3 = imputer.fit(x[:, 4:5])
x[:, 4:5] = imputer.transform(x[:, 4:5])

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])


ct = ColumnTransformer([("GenderProfession", OneHotEncoder(), [0, 3])], remainder='passthrough')
X = ct.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


