#E.coli Dataset, also referred to as the “protein localization sites” dataset
"""The dataset describes the problem of classifying E.coli proteins using their amino 
acid sequences in their cell localization sites. That is, predicting how a protein will 
bind to a cell based on the chemical composition of the protein before it is folded.
The dataset is comprised of 336 examples of E.coli proteins and each example is 
described using seven input variables calculated from the proteins amino acid sequence.
Ignoring the sequence name, the input features are described as follows:
mcg: McGeoch’s method for signal sequence recognition.
gvh: von Heijne’s method for signal sequence recognition.
lip: von Heijne’s Signal Peptidase II consensus sequence score.
chg: Presence of charge on N-terminus of predicted lipoproteins.
aac: score of discriminant analysis of the amino acid content of outer membrane and periplasmic proteins.
alm1: score of the ALOM membrane-spanning region prediction program.
alm2: score of ALOM program after excluding putative cleavable signal regions from the sequence.
There are eight classes described as follows:
cp: cytoplasm
im: inner membrane without signal sequence
pp: periplasm
imU: inner membrane, non cleavable signal sequence
om: outer membrane
omL: outer membrane lipoprotein
imL: inner membrane lipoprotein
imS: inner membrane, cleavable signal sequence
The distribution of examples across the classes is not equal and in some cases severely 
imbalanced"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('ecoli.csv')
X = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 7].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)"""

from sklearn.svm import SVC
classifier = SVC(C = 1, kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy_of_testset = ((35+8+4+3+2+8)/68)*100

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = classifier, X = X_train, y=y_train, cv = 10)
Accuracy_of_trainingset = accuracy.mean()*100
accuracy.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

