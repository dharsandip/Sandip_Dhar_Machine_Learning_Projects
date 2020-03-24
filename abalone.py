#Abalone Dataset
"""The Abalone Dataset involves predicting the age of abalone given objective measures of 
individuals. It is a multi-class classification problem, but can also be framed as a 
regression. The number of observations for each class is not balanced. There are 4,177 
observations with 8 input variables and 1 output variable. The variable names are as follows:
Sex (M, F, I), Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, 
Shell weight, Rings"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('abalone.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])

ohe = OneHotEncoder()
X_ohe = X[:, 0]
X_ohe = X_ohe.reshape(-1, 1)
X_ohe = ohe.fit_transform(X_ohe).toarray()
X_ohe = X_ohe[:, 1:3]
X = X[:, 1:8]
X = np.append(X_ohe, X, axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

import math
RMSE_error = math.sqrt(np.mean(((y_test - y_pred)**2)))



