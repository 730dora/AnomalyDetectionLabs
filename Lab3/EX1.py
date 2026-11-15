import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=500, n_features=2, centers=[[0, 0]])
w = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size = 5)
w = w/np.linalg.norm(w, keepdims=True, axis=1)

dataset = np.random.uniform(low=-3, high=3, size=(500, 2))

scores = np.zeros(500)
for i in range(5):
    dot_x = x @ w[i]
    dot_dataset = dataset @ w[i]
    minimul = dot_x.min()-2
    maximul = dot_x.max()+2
    nrbins = 100 # manually changed the bins that are by default 10 so <6 a lot of high scored points 6-15 normal amount
    # and the more bins the narrower so fewer points score as normal
    hist, bin_edges = np.histogram(dot_x, bins = nrbins, range=(minimul, maximul))
    probofbin = hist/500

    for j in range(500):
        point = dot_dataset[j]
        for k in range(nrbins):
            if bin_edges[k] <= point < bin_edges[k + 1]:
                scores[j] = scores[j] + probofbin[k]
                break

anScores = scores/5

plt.scatter(dataset[:, 0], dataset[:, 1], c = anScores, cmap= 'plasma')
plt.colorbar()
plt.show()






