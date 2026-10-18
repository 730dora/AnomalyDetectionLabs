import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.metrics import confusion_matrix

mean_vector = np.array([2.1, 3.3])
covariance_matrix = np.array([[2.3, 1.1], [1.1, 4.4]])
points = 1000
dimensions = 2
contamination = 0.1
x = np.random.randn(points, dimensions)

cholesky_dec = np.linalg.cholesky(covariance_matrix)
y = np.dot(x, cholesky_dec) + mean_vector


# idk how i should have done the anomalies exactly because
# without them I couldn't have done the confusion matrix so i just
# put the last points targeted as contamination with a given distance
nr = np.int_(points*contamination)
y[-nr:] += 5
y_true = np.zeros(points)
y_true[-nr:] = 1

print(y)
plt.scatter(y [:,0], y[:, 1], c= y_true, cmap= 'PiYG')
plt.show()

z_score = np.max(np.abs(zscore(y)), axis=1)
threshold = np.quantile(z_score, 0.9)

y_pred = (z_score > threshold).astype(int)
tn, fp, fn,tp = confusion_matrix(y_true, y_pred).ravel()

tpr = tp/(tp + fn)
tnr = tn / (tn + fp)
ba = (tpr + tnr)/2

print(ba)