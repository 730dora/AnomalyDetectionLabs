from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
import numpy as np
import sklearn

x_train, x_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1)
plt.scatter(x_train [:,0], x_train[:, 1], c=y_train, cmap= 'Spectral')
plt.show()