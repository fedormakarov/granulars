import numpy as np


def some_doing(matrix):
    m = np.sum(matrix, axis=0)
    matrix_m = matrix / m
    m_ = np.mean(matrix_m, axis=1)
    return m_


def calculate_ci(matrix):
    m_ = np.linalg.eigvals(matrix)
    m_ = np.max(m_)
    m_ = np.real(m_)
    CI = (m_ - m) / (m - 1)
    return CI


A = np.array([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]])
B1 = np.array([[1, 3, 7], [1 / 3, 1, 5], [1 / 7, 1 / 5, 1]])
B2 = np.array([[1, 1 / 5, 1], [5, 1, 5], [1, 1 / 5, 1]])
B3 = np.array([[1, 5, 9], [1 / 5, 1, 3], [1 / 9, 1 / 3, 1]])

A_ = some_doing(A)
B1_ = some_doing(B1)
B2_ = some_doing(B2)
B3_ = some_doing(B3)

S = np.array([B1_, B2_, B3_]).T

r = np.dot(S, A_)

m = 3

m_ = np.linalg.eigvals(A)
m_ = np.max(m_)
m_ = np.real(m_)
CI = (m_ - m) / (m - 1)
print(CI)

print(calculate_ci(B1))
print(calculate_ci(B2))
print(calculate_ci(B3))
