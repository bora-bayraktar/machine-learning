import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read Data from File
X = np.genfromtxt("hw03_data_set_images.csv", delimiter=",").reshape(5, 39, 320)
y_truth = np.genfromtxt("hw03_data_set_labels.csv").reshape(5, 39).astype(int)

# Training Data
training_X = X[:, :25, :].reshape(125, 320)
training_y_truth = y_truth[:, :25].reshape(125)

# Test Data
test_X = X[:, 25:, :].reshape(70, 320)
test_y_truth = y_truth[:, 25:].reshape(70)

K = np.max(training_y_truth)        # Class Number
N_training = training_X.shape[0]    # Training Sample Size
N_test = test_X.shape[0]            # Test Sample Size

y_truth = np.zeros((N_training, K)).astype(int)
y_truth[range(N_training), np.array(training_y_truth) - 1] = 1

# Sigmoid Function
def sigmoid(X, W, w0):
    return 1 / (1 + np.exp(-(np.matmul(X, W) + w0)))

# Gradient Functions
def gradient_W(X, y_truth, y_predicted):
    return np.asarray([-np.matmul(y_truth[:,c] - y_predicted[:,c], X) for c in range(K)]).transpose()

def gradient_w0(y_truth, y_predicted):
    return -np.sum(y_truth - y_predicted, axis = 0)

# Setting Learning Parameters
eta = 0.001
epsilon = 0.001

# Initializing Parameters Randomly
W = np.random.uniform(low = -0.01, high = 0.01, size = (training_X.shape[1], K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))

# Learning W and w0 Using Gradient Descent
iteration = 1
objective_values = []
while True:
    y_predicted = sigmoid(training_X, W, w0) 

    objective_values = np.append(objective_values, np.sum(0.5 * ((y_truth - y_predicted) ** 2)))

    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(training_X, y_truth, y_predicted)
    w0 = w0 - eta * gradient_w0(y_truth, y_predicted)

    if np.sqrt(np.sum((w0 - w0_old)) ** 2 + np.sum((W - W_old) ** 2)) < epsilon:
        break

    iteration = iteration + 1

print(W)
print(w0)

# Plotting Objective Function During Iterations
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

# Confusion Matrix for Training Data
training_predictions = np.argmax(y_predicted, axis = 1) + 1

confusion_matrix = pd.crosstab(training_predictions, training_y_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print("-" * 30)
print(confusion_matrix)
print("-" * 30)

# Confustion Matrix for Test Data
test_predictions = sigmoid(test_X, W, w0)
test_predictions = np.argmax(test_predictions, axis=1) + 1

confusion_matrix = pd.crosstab(test_predictions, test_y_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)