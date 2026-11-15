from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
import numpy as np
import matplotlib.pyplot as plt

x, y = make_blobs(n_samples = [500, 500], n_features=2, centers= [(10, 0), (0, 10)])

test_data = np.random.uniform(low = -10, high = 20, size = (1000, 2))

clf = IForest(contamination = 0.02)
clf.fit(x)
#clf.predict(test_data)
#train_scores = clf.decision_scores_
test_scores = clf.decision_function(test_data)
print("the iforest test scores: ", test_scores)
print("\n")

plt.subplot(1, 3, 1)
plt.title("IForest")
plt.scatter(test_data[:, 0], test_data[:, 1], c = test_scores, cmap="plasma")
plt.colorbar()

clf2 = DIF(contamination=0.02, hidden_neurons=[20, 10])
clf2.fit(x)
#clf2.predict(test_data)
test_scores2 = clf2.decision_function(test_data)

plt.subplot(1, 3, 2)
plt.title("DIF")
plt.scatter(test_data[:, 0], test_data[:, 1], c = test_scores2, cmap="plasma")
plt.colorbar()


clf3 = LODA(contamination=0.02,n_bins=20)
clf3.fit(x)
#clf3.predict(test_data)
test_scores3 = clf3.decision_function(test_data)
print("the loda test scores: ", test_scores3)
plt.subplot(1, 3, 3)
plt.title("LODA")
plt.scatter(test_data[:, 0], test_data[:, 1], c = test_scores3, cmap="plasma")
plt.colorbar()
plt.show()

# i'd say that the LODA plot looks almost opposite of the IForest one because based
# on the number of bins it gives the highest anomaly score to the points which are
# the loneliest in a bucket