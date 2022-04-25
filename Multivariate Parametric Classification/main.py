import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generating Data
class_means = np.array([[0.0, 4.5], [-4.5, -1.0], [4.5, -1.0], [0.0, -4.0]])
class_covariances = np.array([[[3.2, 0.0], [0.0, 1.2]], [[1.2, 0.8], [0.8, 1.2]], [[1.2, -0.8], [-0.8, 1.2]], [[1.2, 0.0], [0.0, 3.2]]])
class_sizes = np.array([105, 145, 135, 115])

points1 = np.random.multivariate_normal(class_means[0,:], class_covariances[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_covariances[1,:,:], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2,:], class_covariances[2,:,:], class_sizes[2])
points4 = np.random.multivariate_normal(class_means[3,:], class_covariances[3,:,:], class_sizes[3])

X = np.concatenate((points1, points2, points3, points4))
Y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2]), np.repeat(4, class_sizes[3])))

# Plotting the Data
plt.figure(figsize=(7, 7))
plt.plot(points1[:,0], points1[:,1], "r.", markersize=10)
plt.plot(points2[:,0], points2[:,1], "g.", markersize=10)
plt.plot(points3[:,0], points3[:,1], "b.", markersize=10)
plt.plot(points4[:,0], points4[:,1], "m.", markersize=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# Parameter Estimation
K = np.max(Y)
D = X.shape[1]

sample_means = np.array([np.mean(X[Y == c + 1], axis=0) for c in range(K)])
sample_covariances = np.array([np.cov((X[Y == c + 1] - sample_means[c]).T) for c in range(K)])
class_priors = np.array([np.mean(Y == c + 1) for c in range(K)])

print(sample_means)
print(sample_covariances)
print(class_priors)

W = np.array([-0.5 * np.linalg.inv(sample_covariances[c]) for c in range(K)])
w = np.array([np.matmul(np.linalg.inv(sample_covariances[c]), sample_means[c]) for c in range(K)])
w0 = [-0.5 * np.linalg.multi_dot((sample_means[c], np.linalg.inv(sample_covariances[c]), sample_means[c].T)) - D / 2 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(sample_covariances[c])) + np.log(class_priors[c]) for c in range(K)]

# Prediction
predictions = []
for i in range(X.shape[0]):
    scores = [np.linalg.multi_dot((X[i], W[c], X[i].T)) + np.dot(w[c], X[i].T) + w0[c] for c in range(K)]
    predictions.append(scores)

results = np.argmax(predictions, axis = 1) + 1

confusion = pd.crosstab(results, Y, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion)

# Visualization
x1_interval = np.linspace(-11, 10, 2201)
x2_interval = np.linspace(-11, 10, 2201)

x1, x2 = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))

for c in range(K):
    discriminant_values[:,:,c] = W[c, 0, 0] * pow(x1, 2) + W[c, 0, 1] * x1 * x2 + W[c, 1, 0] * x1 * x2 + w[c, 0] * x1 + W[c, 1, 1] * pow(x2, 2) + w[c, 1] * x2 + w0[c]

A = discriminant_values[:,:,0]
B = discriminant_values[:,:,1]
C = discriminant_values[:,:,2]
D = discriminant_values[:,:,3]

A[(A > B) & (A > C) & (A > D)] = np.nan
B[(B > A) & (B > C) & (B > D)] = np.nan
C[(C > A) & (C > B) & (C > D)] = np.nan
D[(D > A) & (D > B) & (D > C)] = np.nan

plt.figure(figsize = (7, 7))
plt.plot(X[Y == 1, 0], X[Y == 1, 1], "r.", markersize = 10)
plt.plot(X[Y == 2, 0], X[Y == 2, 1], "g.", markersize = 10)
plt.plot(X[Y == 3, 0], X[Y == 3, 1], "b.", markersize = 10)
plt.plot(X[Y == 4, 0], X[Y == 4, 1], "m.", markersize = 10)
plt.plot(X[results != Y, 0], X[results != Y, 1], "ko", markersize=12, fillstyle="none")

plt.contourf(x1, x2, B + C + D, levels=0, colors="r", alpha=.3)
plt.contourf(x1, x2, A + C + D, levels=0, colors="g", alpha=.3)
plt.contourf(x1, x2, A + B + D, levels=0, colors="b", alpha=.3)
plt.contourf(x1, x2, A + B + C, levels=0, colors="m", alpha=.3)

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()