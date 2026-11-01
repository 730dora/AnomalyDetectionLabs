import numpy as np
import matplotlib.pyplot as plt

various_val = [(0, 1), (1, 5), (3, 7), (4, 12)]
a = 1
b = -5.5
c = 3

fig = plt.figure(figsize = (20, 5))
i=0
for mean, var in various_val:
    ax = fig.add_subplot(1, 4, i+1, projection = '3d')
    x1 = np.random.normal(0, 1, 100)
    x2 = np.random.normal(0, 1, 100)
    noise = np.random.normal(mean, np.sqrt(var), 100)
    y = a * x1 + b * x2 + c + noise


    x1_points = [0, 10, 0, 10]
    x2_points = [0, -10, 0, -10]
    y_points = [c, a*10+b*-10+c, c+30, a*10+b*-10+c+30]
    x1 = np.append(x1, x1_points)
    x2 = np.append(x2, x2_points)
    y = np.append(y, y_points)

    x_mat = np.c_[np.ones(len(x1)), x1, x2]
    h = x_mat @ np.linalg.inv(x_mat.T @ x_mat) @ x_mat.T

    leverage = np.diag(h)
    print("leerage score for mean = ", mean, " and var = ", var, " the leverage score is ", leverage)

    ax.scatter(x1, x2, y)
    maxim = np.argmax(leverage)
    ax.scatter(x1[maxim], x2[maxim], y[maxim], edgecolors = 'red', linewidths=3)
    i=i+1

plt.tight_layout()
plt.show()