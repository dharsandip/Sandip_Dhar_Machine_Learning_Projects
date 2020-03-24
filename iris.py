#Iris Flowers Dataset
"""The Iris Flowers Dataset involves predicting the flower species given measurements of 
iris flowers. It is a multi-class classification problem. The number of observations for 
each class is balanced. There are 150 observations with 4 input variables and 1 output 
variable. The variable names are as follows:
Sepal length in cm
Sepal width in cm
Petal length in cm
Petal width in cm
Class (Iris Setosa, Iris Versicolour, Iris Virginica)"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('iris.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy_of_testset = ((cm[0, 0]+cm[1, 1]+cm[2, 2])/
                       (cm[0, 0]+cm[1, 1]+cm[2, 2]+cm[0, 1]+cm[1, 0]+cm[0, 2]+cm[2, 0]+cm[1, 2]+cm[2, 1]))*100

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
Accuracy_of_trainingset = accuracies.mean()*100
accuracies.std()
