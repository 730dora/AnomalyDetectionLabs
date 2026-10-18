from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from scipy.stats import zscore
import numpy as np
from sklearn.metrics import confusion_matrix

x_train, x_test, y_train, y_test = generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1)
z_score = np.abs(zscore(x_train))
threshold = np.quantile(z_score, 0.9)
#mdeci...
y_pred = (z_score > threshold).astype(int)
tn, fp, fn,tp = confusion_matrix(y_train, y_pred).ravel()

tpr = tp/(tp + fn)
tnr = tn / (tn + fp)
ba = (tpr + tnr)/2

print(ba, "\n")
print(y_train)
# print(z_score)
