import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF

x, y = make_blobs([200, 100], 2, cluster_std = [2, 6], centers=[[-10, -10], [10, 10]])
knn = KNN(0.07, 15)
knn.fit(x)
knn_labels = knn.predict(x)

lof = LOF(n_neighbors=15, contamination=0.07)
lof.fit(x)
lof_labels = lof.predict(x)

fig, ax = plt.subplots(1, 2)
ax[0].scatter(x[knn_labels == 0, 0], x[knn_labels == 0, 1], c='blue')
ax[0].scatter(x[knn_labels == 1, 0], x[knn_labels == 1, 1], c='red')
ax[0].set_title('KNN')

ax[1].scatter(x[lof_labels == 0, 0], x[lof_labels == 0, 1], c='blue')
ax[1].scatter(x[lof_labels == 1, 0], x[lof_labels == 1, 1], c='red')
ax[1].set_title('LOF')

plt.show()


