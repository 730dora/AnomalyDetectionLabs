import numpy as np
import matplotlib.pyplot as plt

various_val = [(0, 1), (1, 5), (3, 7), (4, 12)]
a = 1
b = 5.5

fig, axes = plt.subplots(1, 4, figsize = (20, 5))
i=0
for mean, var in various_val:

    x = np.random.normal(0, 1, 100)
    noise = np.random.normal(mean, np.sqrt(var), 100)
    y = a * x + b + noise


    x_points = [0, 10, 0, 10]
    y_points = [b, a*10+b, b+15, a*10+b+15]
    x = np.append(x, x_points)
    y = np.append(y, y_points)

    x_mat = np.c_[np.ones(len(x)), x]
    h = x_mat @ np.linalg.inv(x_mat.T @ x_mat) @ x_mat.T

    leverage = np.diag(h)
    print("leerage score for mean = ", mean, " and var = ", var, " the leverage score is ", leverage)

    axes[i].scatter(x, y)
    maxim = np.argmax(leverage)
    axes[i].scatter(x[maxim], y[maxim], edgecolors = 'red')
    i=i+1

plt.tight_layout()
plt.show()