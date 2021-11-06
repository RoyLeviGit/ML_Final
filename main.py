import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('winequality-red.csv', sep=';')
data = df.to_numpy()
X = preprocessing.scale(data[:, :-1])
y = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)


def sklearn_fit():
    for i in range(-2, 2):
        log_reg = LogisticRegression(C=10**i, max_iter=1e6, penalty='l1', solver='saga')
        log_reg.fit(X_train, y_train)

        print(f'{i}: {log_reg.coef_}')
        print(f'{i}: Accuracy on training data = {log_reg.score(X_train, y_train)}')
        print(f'{i}: Accuracy on test data = {log_reg.score(X_test, y_test)}')


# sklearn_fit()

X_train_1 = np.hstack((np.ones(X_train.shape[0]).reshape(-1, 1), X_train))
y_train_2d = y_train.reshape(-1, 1)
N_train = y_train_2d.shape[0]

X_test_1 = np.hstack((np.ones(X_test.shape[0]).reshape(-1, 1), X_test))
y_test_2d = y_test.reshape(-1, 1)
N_test = y_test_2d.shape[0]


def compute_cost(X_1, y_2d, w, N):
    return (1 / (2 * N)) * ((X_1 @ w - y_2d) ** 2).sum()


def gradient_descent(X_1, y_2d, learning_rate, w, N, num_iters):
    for i in range(num_iters):
        w = w - ((learning_rate / N) * (X_1.T @ (X_1 @ w - y_2d)))
    return w


w_init = np.zeros((X_train_1.shape[1], 1))
print(compute_cost(X_train_1, y_train_2d, w_init, N_train))
print(compute_cost(X_test_1, y_test_2d, w_init, N_test))
w_gd = gradient_descent(X_train_1, y_train_2d, 0.0049, w_init, N_train, 100000)
print(w_gd)
print(compute_cost(X_train_1, y_train_2d, w_gd, N_train))
print(compute_cost(X_test_1, y_test_2d, w_gd, N_test))
