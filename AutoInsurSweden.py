#Swedish Auto Insurance Dataset
"""The Swedish Auto Insurance Dataset involves predicting the total payment for all claims 
in thousands of Swedish Kronor, given the total number of claims. It is a regression 
problem. It is comprised of 63 observations with 1 input variable and one output variable. 
The variable names are as follows:
Number of claims, Total payment for all claims in thousands of Swedish Kronor"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Dataset = pd.read_csv('AutoInsurSweden.csv')
X = Dataset.iloc[:, 0].values
y = Dataset.iloc[:, [1]].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = sc.fit_transform(y_train)


#Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 12, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_pred = sc.inverse_transform(y_pred)
X_test = sc.inverse_transform(X_test)
y_train = sc.inverse_transform(y_train)

import math
# The mean squared error
print("Root Mean squared error for the testset: %.2f" % math.sqrt(np.mean((y_pred - y_test)**2)))

#Plots
plt.scatter(X_test, y_test, s=100, color = 'red')
plt.scatter(X_test, y_pred, s=100, color = 'blue')
plt.title('Total payment for all claims Vs Number of claims for the testset, red = original, blue= predicted')
plt.xlabel('Number of claims')
plt.ylabel('Total payment for all claims')
plt.show()





