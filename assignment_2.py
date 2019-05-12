import numpy as np
import matplotlib.pyplot as plt


def get_data(namefile):
    with open(namefile) as f:
        data = [item.strip() for item in f][1:]
    data = np.array([float(item) for item in data])
    np.sort(data)
    return data


def return_v(U):
    U_m = U ** m
    return np.dot(U_m.T, data) / sum(U_m)


def return_u(v):
    U = np.zeros((len(data), clusters))
    power = 2.0 / (m - 1)
    dim = range(clusters)
    for k in range(len(data)):
        for i in dim:
            U[k, i] = np.sum([(np.linalg.norm(data[k] - v[i]) / np.linalg.norm(data[k] - v[j])) ** power for j in dim])
    return 1 / U


def fit(U):
    while True:
        U_previous = U.copy()
        v = return_v(U_previous)
        U = return_u(v)
        if np.max(np.absolute(np.subtract(U, U_previous))) < eps:
            break
    return U_previous


def plot(U):
    for i in range(clusters):
        s = np.argsort(data)
        plt.plot(data[s], U[s, i])
    plt.suptitle('Clusters={}, M={}'.format(clusters, m))
    plt.show()


if __name__ == "__main__":
    m_ = [1.5, 2.1, 3.0]
    c = [3, 4, 6]
    eps = 1e-4
    data = get_data('data.csv')
    for m in m_:
        for clusters in c:
            U = np.random.uniform(0, 2 / clusters, size=(len(data), clusters))
            U = fit(U)
            plot(U)
