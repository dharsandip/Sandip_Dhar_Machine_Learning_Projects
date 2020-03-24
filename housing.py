# Boston House Price Dataset
""""The Boston House Price Dataset involves the prediction of a house price in 
thousands of dollars given details of the house and its neighborhood.
It is a regression problem. The number of observations for each class is balanced. 
There are 506 observations with 13 input variables and 1 output variable"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('housing.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

y_pred_train = regressor.predict(X_train)

# The mean squared error
import math
rmse_of_testset = math.sqrt(np.mean((y_pred - y_test)**2))
rmse_of_trainingset = math.sqrt(np.mean((y_pred_train - y_train)**2))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
Accuracy_of_trainingset = (accuracies.mean())*100
accuracies.std()

