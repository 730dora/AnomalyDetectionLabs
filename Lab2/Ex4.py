import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pyod.models.knn import KNN
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

data = loadmat('cardio.mat')
x = data['X']
y = data['y'].flatten()

x_train, x_test, y_train, y_test = train_test_split(x, y)
train_norm, test_norm = standardizer(x_train, x_test)
train_scoresList = []
test_scoresList = []
contamination = 0.096

for k in range(30, 121, 10):
    clf = KNN(contamination=0.096, n_neighbors=k)
    clf.fit(train_norm)
    train_pred = clf.predict(train_norm)
    test_pred = clf.predict(test_norm)

    ba_train = balanced_accuracy_score(y_train, train_pred)
    ba_test = balanced_accuracy_score(y_test, test_pred)
    print("the train balanced accuracy on ", k, "nn is ", ba_train)
    print("the test balanced accuracy on ", k, "nn is ", ba_test, "\n")

    train_scoresList.append(clf.decision_scores_)
    test_scoresList.append(clf.decision_function(test_norm))

train_scores = np.column_stack(train_scoresList)
test_scores = np.column_stack(test_scoresList)

norm_train_scores, norm_test_scores = standardizer(train_scores, test_scores)
averageTrainscores = average(norm_train_scores)
maxTrainscores = maximization(norm_train_scores)
averageTestscores = average(norm_test_scores)
maxTestscores = maximization(norm_test_scores)

tresh_avg = np.quantile(averageTrainscores, 1-contamination)
avg_pred = np.where(averageTestscores > tresh_avg, 1, 0)
ba_avg = balanced_accuracy_score(y_test, avg_pred)
print("for the average strategy the threshold is ", tresh_avg, "and BA is ", ba_avg)

tresh_max = np.quantile(maxTrainscores, 1-contamination)
max_pred = np.where(maxTestscores > tresh_max, 1, 0)
ba_max = balanced_accuracy_score(y_test, max_pred)
print("for the average strategy the threshold is ", tresh_max, "and BA is ", ba_max)