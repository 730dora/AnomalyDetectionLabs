from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pyod.utils.utility import standardizer
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score

dict = loadmat("shuttle.mat")
x = dict['X']
y = dict['y']

ba_if = 0
auc_if = 0
ba_dif = 0
auc_dif = 0
ba_loda = 0
auc_loda = 0
for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, train_size=0.6)
    train_norm, test_norm = standardizer(x_train, x_test)

    clf = IForest()
    clf.fit(train_norm)
    #train_pred = clf.predict(train_norm)
    test_pred = clf.predict(test_norm)
    #ba_train = balanced_accuracy_score(y_train, train_pred)
    ba_test = balanced_accuracy_score(y_test, test_pred)
    #train_auc = roc_auc_score(y_train, clf.decision_function(train_norm))
    test_auc = roc_auc_score(y_test, clf.decision_function(test_norm))
    ba_if = ba_if + ba_test
    auc_if = auc_if + test_auc

    clf2 = DIF()
    clf2.fit(train_norm)
    #train_pred2 = clf2.predict(train_norm)
    test_pred2 = clf2.predict(test_norm)
    #ba_train2 = balanced_accuracy_score(y_train, train_pred2)
    ba_test2 = balanced_accuracy_score(y_test, test_pred2)
    #train_auc2 = roc_auc_score(y_train, clf2.decision_function(train_norm))
    test_auc2 = roc_auc_score(y_test, clf2.decision_function(test_norm))
    ba_dif = ba_dif + ba_test2
    auc_dif = auc_dif + test_auc2

    clf3 = LODA()
    clf3.fit(train_norm)
    #train_pred3 = clf3.predict(train_norm)
    test_pred3 = clf3.predict(test_norm)
    #ba_train3 = balanced_accuracy_score(y_train, train_pred3)
    ba_test3 = balanced_accuracy_score(y_test, test_pred3)
    #train_auc3 = roc_auc_score(y_train, clf3.decision_function(train_norm))
    test_auc3 = roc_auc_score(y_test, clf3.decision_function(test_norm))
    ba_loda = ba_loda + ba_test3
    auc_loda = auc_loda + test_auc3

mean_ba_if = ba_if/10
mean_auc_if = auc_if/10
mean_ba_dif = ba_dif/10
mean_auc_dif = auc_dif/10
mean_ba_loda = ba_loda/10
mean_auc_loda = auc_loda/10

print("IForest the mean BA and ROC AUC: ", mean_ba_if, ", ", mean_auc_if, "\n")
print("DIF the mean BA and ROC AUC: ", mean_ba_dif, ", ", mean_auc_dif, "\n")
print("LODA the mean BA and ROC AUC: ", mean_ba_loda, ", ", mean_auc_loda)

# this is what i got on one run, it took 16 min so i'll put it here for reference
# IForest the mean BA and ROC AUC:  0.9766220257918488 ,  0.9973120572211004

# DIF the mean BA and ROC AUC:  0.9208822880598234 ,  0.9704719094645637

# LODA the mean BA and ROC AUC:  0.6201696569444367 ,  0.5376402135521122

