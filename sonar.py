#Sonar Dataset
"""The Sonar Dataset involves the prediction of whether or not an object is a mine or a 
rock given the strength of sonar returns at different angles. It is a binary (2-class) 
classification problem. The number of observations for each class is not balanced. 
There are 208 observations with 60 input variables and 1 output variable. The variable 
names are as follows: Sonar returns at different angles, Class (M for mine and R for rock)"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('sonar.csv')
X = dataset.iloc[:, 0:60].values
y = dataset.iloc[:, 60].values

"""from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

#Kernel SVM Classifier
from sklearn.svm import SVC
classifier = SVC(C = 10, kernel = 'rbf', gamma = 0.1, random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy_of_testset = ((cm[0, 0] + cm[1, 1])/(cm[0, 0] + cm[1, 1] + cm[0, 1] + cm[1, 0]))*100

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
Accuracy_of_trainingset = accuracies.mean()*100
accuracies.std()

#Plots
plt.scatter(X_test, y_test, s=20, color = 'red')
plt.scatter(X_test, y_pred, s=20, color = 'blue')
plt.xlabel('Sonar returns at different angles (scaled)')
plt.ylabel('R = Rock, M = Mine')
plt.title('Comparison of results for the testset: Red = original data, Blue = Predicted')
plt.show()
