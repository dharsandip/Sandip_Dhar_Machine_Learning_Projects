#Banknote Dataset
"""The Banknote Dataset involves predicting whether a given banknote is authentic given a 
number of measures taken from a photograph.It is a binary (2-class) classification problem. 
The number of observations for each class is not balanced. There are 1,372 observations with 
4 input variables and 1 output variable. The variable names are as follows:
Variance of Wavelet Transformed image (continuous)
Skewness of Wavelet Transformed image (continuous)
Kurtosis of Wavelet Transformed image (continuous)
Entropy of image (continuous)
Class (0 for authentic, 1 for inauthentic)"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data_banknote_authentication.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Kernel SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy_of_testset = ((cm[0, 0] + cm[1, 1])/(cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0]))*100

# Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
Accuracy_for_trainingset = accuracies.mean()*100
accuracies.std()

