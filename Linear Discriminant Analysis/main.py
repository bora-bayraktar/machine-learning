import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg as linalg
import scipy.stats as st
import scipy.spatial.distance as dt

# Read Data from File
data_set = np.genfromtxt("hw07_data_set_images.csv", delimiter=",")
labels = np.genfromtxt("hw07_data_set_labels.csv").astype(int)

# Training Data
x_training = data_set[:2000, :]
y_training = labels[:2000].astype(int)

# Test Data
x_test = data_set[2000:4000, :]
y_test = labels[2000:4000].astype(int)

K = np.max(y_training)
N = x_training.shape[0]
D = x_training.shape[1]

mean = np.mean(x_training, axis=0, keepdims=True)
means = np.asarray([np.mean(x_training[y_training == c + 1], axis=0, keepdims=True) for c in range(K)])

# Calculate SW and SB
SW = np.sum(np.asarray([np.matmul((x_training[y_training == c + 1] - means[c]).T, x_training[y_training == c + 1] - means[c]) for c in range(K)]), axis=0)
SB = np.sum(np.asarray([np.matmul((means[c] - mean).T, means[c] - mean) * len(x_training[y_training == c + 1]) for c in range(K)]), axis=0)

print(SW[0:4,0:4])
print(SB[0:4,0:4])

# Calcualate eigenvalues and eigenvectors
W = np.matmul(linalg.cho_solve(linalg.cho_factor(SW), np.eye(D)), SB)

values, vectors = linalg.eig(W)
values = np.real(values)
vectors = np.real(vectors)
print(values[0:9])

Z = np.matmul(x_training - np.mean(x_training, axis=0), vectors[:, [0, 1]])

# Plot training data
plt.figure(figsize=(6, 6))
point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])

for c in range(K):
    plt.plot(Z[y_training == c + 1, 0], Z[y_training == c + 1, 1], marker="o", markersize=2, linestyle="none", color=point_colors[c])

plt.legend(["t-shirt-top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"], loc="upper left", markerscale=2)
plt.xlabel("Component#1")
plt.ylabel("COmponent#2")
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.show()

# Plot test data
Z = np.matmul(x_test - np.mean(x_test, axis=0), vectors[:, [0, 1]])

plt.figure(figsize=(6, 6))
point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])

for c in range(K):
    plt.plot(Z[y_test == c + 1, 0], Z[y_test == c + 1, 1], marker="o", markersize=2, linestyle="none", color=point_colors[c])

plt.legend(["t-shirt-top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"], loc="upper left", markerscale=2)
plt.xlabel("Component#1")
plt.ylabel("COmponent#2")
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.show()

# Confusion matrix for training data
Z_training = np.matmul(x_training - np.mean(x_training, axis=0), vectors[:, :9])
training_predictions = st.mode(y_training[np.argsort(dt.cdist(Z_training, Z_training), axis=0)][0:11, :], axis=0)[0][0]
confusion_matrix = pd.crosstab(training_predictions, y_training, rownames=["y_predicted"], colnames=["y_train"])
print(confusion_matrix)

# Confusion matrix for test data
Z_test = np.matmul(x_test - np.mean(x_test, axis=0), vectors[:, :9])
test_predictions = np.concatenate(st.mode(y_training[np.argsort(dt.cdist(Z_test, Z_training))][:, 0:11], axis=1)[0])
confusion_matrix = pd.crosstab(test_predictions, y_test, rownames=["y_predicted"], colnames=["y_test"])
print(confusion_matrix)
