import numpy as np
import pandas as pd

# Read Data from File
X = np.genfromtxt("hw02_data_set_images.csv", delimiter=",").reshape(5, 39, 320)
y_truth = np.genfromtxt("hw02_data_set_labels.csv").reshape(5, 39).astype(int)

# Training Data
training_X = X[:, :25, :].reshape(125, 320)
training_y_truth = y_truth[:, :25].reshape(125)

# Test Data
test_X = X[:, 25:, :].reshape(70, 320)
test_y_truth = y_truth[:, 25:].reshape(70)

K = np.max(training_y_truth)        # Class Number
N_training = training_X.shape[0]    # Training Sample Size
N_test = test_X.shape[0]            # Test Sample Size

# Calculating pcd
pcd = np.array([[np.mean(training_X[training_y_truth == (c + 1), i]) for i in range(training_X.shape[1])] for c in range(K)])
print(pcd)

# Calculating Class Priors
class_priors = [np.mean(training_y_truth == (c + 1)) for c in range(K)]
print(class_priors)

# Safelog
def safelog(x):
    return np.log(x + 1e-100)

# Score function
def score(pcd, x):
    return [np.sum(x.T * safelog(pcd[c]) + (1 - x.T) * safelog(1 - pcd[c])) + safelog(class_priors[c]) for c in range(K)]

# Confusion Matrix for Training Data
training_scores = [score(pcd, training_X[i]) for i in range(N_training)]
training_y_predicted = np.argmax(training_scores, axis=1) + 1

training_confusion_matrix = pd.crosstab(training_y_predicted, training_y_truth, rownames=["y_pred"], colnames=["y_truth"])
print(training_confusion_matrix)

# Confusion Matrix for Test Data
test_scores = [score(pcd, test_X[i]) for i in range(N_test)]
test_y_predicted = np.argmax(test_scores, axis=1) + 1

test_confusion_matrix = pd.crosstab(test_y_predicted, test_y_truth, rownames=["y_pred"], colnames=["y_truth"])
print(test_confusion_matrix)