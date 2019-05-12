import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def return_v(U, x, y):
    U_m = U ** m
    return U_m.T @ x / sum(U_m), U_m.T @ y / sum(U_m)


def return_u_2(v, w, x, y, clusters):
    U = np.zeros((len(x), clusters))
    power = 2.0 / (m - 1)
    dim = range(clusters)
    for k in range(len(x)):
        for i in dim:
            U[k, i] = np.sum([(np.linalg.norm(np.array([x[k], y[k]]) - np.array([v[i], w[i]])) /
                               np.linalg.norm(np.array([x[k], y[k]]) - np.array([v[j], w[j]]))) ** power for j in dim])
    return 1 / U


def return_u_1(v, data, clusters):
    U = np.zeros((len(data), clusters))
    power = 2.0 / (m - 1)
    dim = range(clusters)
    for k in range(len(data)):
        for i in dim:
            U[k, i] = np.sum([(np.linalg.norm(data[k] - v[i]) / np.linalg.norm(data[k] - v[j])) ** power for j in dim])
    return 1 / U


def fit(U, x, y, clusters):
    while True:
        U_previous = U.copy()
        v = return_v(U_previous, x, y)
        U = return_u_2(v[0], v[1], x, y, clusters)
        if np.max(np.absolute(np.subtract(U, U_previous))) < eps:
            break
    return U_previous, return_u_1(v[0], x, clusters)


def plot(U, y, clusters, m):
    for i in range(clusters):
        s = np.argsort(y)
        plt.plot(y[s], U[s, i])
    plt.suptitle('Clusters={}, M={}'.format(clusters, m))
    plt.show()


def MSE(y, y_):
    y = list(y)
    y_ = list(y_)
    return sum((item - element) ** 2 for item, element in zip(y, y_)) / len(y)


if __name__ == "__main__":
    m = 2
    c = [2, 3, 5, 7]
    eps = 1e-3
    with open('data_funct_rule.csv') as f:
        data = [line.strip().split(',') for line in f][1:]
    data_x = [float(item[0]) for item in data]
    data_y = [float(item[1]) for item in data]

    for clusters in c:
        U, U_ = fit(np.random.uniform(0, 2 / clusters, size=(len(data), clusters)), data_x, data_y, clusters)
        plot(U, np.array(data_y), clusters, m)
        F = U_.T * np.array(data_x)
        F = F.T
        slope = np.linalg.inv(F.T @ F) @ F.T @ np.array(data_y)
        print(slope)
        y_hat = U_ @ slope * np.array(data_x)
        print(MSE(data_y, y_hat))
        data_y = np.array(data_y)
        y_scaled = preprocessing.scale(data_y)
        y_hat_scaled = preprocessing.scale(y_hat)
        print(MSE(y_scaled, y_hat_scaled))
        plt.scatter(data_x, data_y, color='g', label='data')
        plt.scatter(data_x, y_hat, color='r', label='degranularized function')
        plt.suptitle('Clusters={}, M={}'.format(clusters, m))
        plt.legend()
        plt.show()
