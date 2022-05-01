import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cvxopt as cvx

# Read Data from File
data_set = np.genfromtxt("hw06_data_set_images.csv", delimiter=",")
labels = np.genfromtxt("hw06_data_set_labels.csv").astype(int)

# Training Data
x_training = data_set[:1000, :]
y_training = labels[:1000]

# Test Data
x_test = data_set[1000:2000, :]
y_test = labels[1000:2000]

N = x_training.shape[1]

bin_width = 4

left_borders = np.arange(0, 256, bin_width)
right_borders = np.arange(bin_width, bin_width + 256, bin_width)

H_train = np.asarray([[np.sum((left_borders[bin] <= x) & (x < right_borders[bin])) / N for bin in range(len(left_borders))] for x in x_training])
H_test = np.asarray([[np.sum((left_borders[bin] <= x) & (x < right_borders[bin])) / N for bin in range(len(left_borders))] for x in x_test])

print(H_train[0:5, 0:5])
print(H_test[0:5, 0:5])

def histogram_intersection_kernel(hist_1, hist_2):
    result = np.zeros((hist_1.shape[0], hist_1.shape[0]))

    for i in range(len(hist_1)):
        for j in range((len(hist_2))):
            result[i][j] = np.sum(np.minimum(hist_1[i], hist_2[j]))

    return result

K_train = histogram_intersection_kernel(H_train, H_train)
K_test = histogram_intersection_kernel(H_test, H_train)

print(K_train[0:5, 0:5])
print(K_test[0:5, 0:5])

N_training = x_training.shape[0]

def support_vector_machine(C, y_train, K_train, K_test):
    yyK = np.matmul(y_train[:, None], y_train[None, :]) * K_train

    # set learning parameters
    epsilon = 0.001

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_training, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_training), np.eye(N_training))))
    h = cvx.matrix(np.vstack((np.zeros((N_training, 1)), C * np.ones((N_training, 1)))))
    A = cvx.matrix(1.0 * y_train[None, :])
    b = cvx.matrix(0.0)
                    
    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_training)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))

    f_predicted_training = w0 + np.matmul(K_train, y_train[:, None] * alpha[:, None])
    f_predicted_test = w0 + np.matmul(K_test, y_train[:, None] * alpha[:, None])

    y_predicted_training = 2 * (f_predicted_training > 0.0) - 1
    y_predicted_test = 2 * (f_predicted_test > 0.0) - 1

    return y_predicted_training, y_predicted_test

def calculate_accuracy(y_truth, y_predicted):
    concat = np.concatenate(y_predicted)
    accuracy = (y_truth == concat).sum() / len(y_truth)

    return accuracy

y_predicted_training, y_predicted_test = support_vector_machine(10, y_training, K_train, K_test)

confusion_matrix_train = pd.crosstab(np.reshape(y_predicted_training, N_training), y_training, rownames=["y_predicted"], colnames= ["y_train"])
confusion_matrix_test = pd.crosstab(np.reshape(y_predicted_test, N_training), y_test, rownames=["y_predicted"], colnames=["y_test"])

print(confusion_matrix_train)
print(confusion_matrix_test)

C_list = np.arange(-1, 3.5, 0.5)

training_accuracies = []
test_accuracies = []

for i in C_list:
    y_predicted_training, y_predicted_test = support_vector_machine(10**i, y_training, K_train, K_test)

    training_accuracy = calculate_accuracy(y_training, y_predicted_training)
    test_accuracy = calculate_accuracy(y_test, y_predicted_test)

    training_accuracies.append(training_accuracy)
    test_accuracies.append(test_accuracy)

plt.figure(figsize=(10,4))
plt.xlabel("Regularization Parameter(log(C))")
plt.ylabel("Accuracy")
plt.plot(C_list, training_accuracies, "b.-", label="training", markersize=8)
plt.plot(C_list, test_accuracies, "r.-", label="test", markersize=8)
plt.legend()
plt.show()