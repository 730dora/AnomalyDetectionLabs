import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score

x_train, x_test, y_train, y_test = generate_data_clusters(400, 200, 2, 2, 0.1)
neighbours = [3, 14, 5, 20, 7]

for i in neighbours:
    clf = KNN(n_neighbors=i, contamination=0.1)
    clf.fit(x_train)
    train_pred = clf.labels_
    test_pred = clf.predict(x_test)

    BAtrain = balanced_accuracy_score(y_train, train_pred)
    BAtest = balanced_accuracy_score(y_test, test_pred)
    print("the train ba for n_neighbors = ", i, " is ", BAtrain)
    print("the test ba for n_neighbors = ", i, " is ", BAtest, "\n")

    fig, ax =plt.subplots(2, 2)
    fig.suptitle(f'for n_neighbors = {i}')
    ax[0, 0].scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='BuPu', edgecolors='k')
    ax[0, 0].set_title('Ground truth labels for training Data')

    ax[0, 1].scatter(x_train[:, 0], x_train[:, 1], c=train_pred, cmap='BuPu', edgecolors='k')
    ax[0, 1].set_title('Predicted labels for training data')

    ax[1, 0].scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='BuPu', edgecolors='k')
    ax[1, 0].set_title('Ground truth labels for test data')

    ax[1, 1].scatter(x_test[:, 0], x_test[:, 1], c=test_pred, cmap='BuPu', edgecolors='k')
    ax[1, 1].set_title('Predicted labels for test data')

    plt.show()
