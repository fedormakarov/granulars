import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def mapping(x, coef, eps):
    return [coef[0] * (1 - eps[0]) + x * (coef[1] - eps[1]),
            coef[0] * (1 + eps[0]) + x * (coef[1] + eps[1])]


def coverage(x, y, coef, eps):
    temp = (mapping(t, coef, eps) for t in x)
    temp = (t[0] <= y_ <= t[1] for t, y_ in zip(temp, y))
    return sum(temp) / len(x)


def specificity(x, y, coef, eps):
    R = max(y) - min(y)
    temp = (mapping(t, coef, eps) for t in x)
    temp = (1 - min(1, (t[1] - t[0]) / R) for t in temp)
    return sum(temp) / len(x)


def get_q(x, y, coef, eps):
    return coverage(x, y, coef, eps) * specificity(x, y, coef, eps)


def plot_line(e_1, e_2, q):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(e_1, e_2, q)
    plt.show()


def plot_grid(e_1, e_2, q):
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(e_1, e_2, q, cmap=cm.jet, linewidth=0.1)
    ax.set_xlabel('epsilon 1')
    ax.set_ylabel('epsilon 2')
    ax.set_zlabel('Q')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def get_attr_from_static_eps(x, y, coefficients):
    e = 0.6
    l = 10
    e_ = [e * 2 * i / l for i in range(l)]
    e_ = [[e * 2 - item, item] for item in e_]
    q = [get_q(x, y, coefficients, item) for item in e_]
    print(e_[q.index(max(q))])
    return [item[0] for item in e_], [item[1] for item in e_], q


if __name__ == "__main__":
    with open('data_granular_mapping.csv') as f:
        data = [line.strip().split(',') for line in f][1:]

    x = np.array([float(item[0]) for item in data])
    y = np.array([float(item[1]) for item in data])

    reg = LinearRegression().fit(np.array(x).reshape(-1, 1), np.array(y))

    coefficients = reg.intercept_, reg.coef_[0]
    print(coefficients)

    eps = [2e-2, 3e-2]
    A = [[coef - e * coef, coef + e * coef] for coef, e in zip(coefficients, eps)]
    print(A)

    e1, e2, q = get_attr_from_static_eps(x, y, coefficients)
    plot_line(e1, e2, q)

    l = 10
    e_1 = [x / l for x in range(l * 3)]
    e_2 = [x / l for x in range(l * 3)]

    temp = [[item, element, get_q(x, y, coefficients, [item, element])] for item in e_1 for element in e_2]

    e_1 = [x[0] for x in temp]
    e_2 = [x[1] for x in temp]
    q = [x[2] for x in temp]

    plot_grid(e_1, e_2, q)

    max_q = max(q)
    max_index = q.index(max_q)
    print(e_1[max_index], e_2[max_index], (e_1[max_index] + e_2[max_index]) / 2)
    eps = [e_1[max_index], e_2[max_index]]
    print(eps, sum(eps) / 2, 1 / l)

    y_ = [mapping(x_, coefficients, eps) for x_ in x]
    y_lo = [_[0] for _ in y_]
    y_hi = [_[1] for _ in y_]
    y_ = [coefficients[0] + coefficients[1] * x_ for x_ in x]

    plt.plot(x, y_lo, color='r', label='Y-')
    plt.plot(x, y_hi, color='b', label='Y+')
    plt.plot(x, y_, color='g', label='y from LR')
    plt.scatter(x, y, color='y', label='y from data')
    plt.legend()
    plt.show()
