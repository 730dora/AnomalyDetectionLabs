from pyod.utils.data import generate_data
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

x_train, x_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1)
plt.scatter(x_train [:,0], x_train[:, 1], c=y_train, cmap= 'Spectral')
plt.show()

clf = KNN(contamination = 0.5)
clf.fit(x_train)
pred_train = clf.predict(x_train)
pred_test =clf.predict(x_test)

# print(pred_train, "\n", pred_test)

tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
tpr = tp/(tp + fn)
tnr = tn/(tn + fp)
ba = (tpr + tnr)/2
print(ba, "\n")

y_score = clf.decision_function(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
#print(fpr, "\n", tpr, "\n", thresholds, "\n", roc_auc)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='random guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()




