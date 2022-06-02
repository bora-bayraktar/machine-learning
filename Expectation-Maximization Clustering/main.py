import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
import scipy.stats as stats

X = np.genfromtxt("hw08_data_set.csv", delimiter=",")
initial_centroids = np.genfromtxt("hw08_initial_centroids.csv", delimiter=",")

class_means = np.array([[+5.0, +5.0],
                        [-5.0, +5.0],
                        [-5.0, -5.0],
                        [+5.0, -5.0],
                        [+5.0, +0.0],
                        [+0.0, +5.0],
                        [-5.0, +0.0],
                        [+0.0, -5.0],
                        [+0.0, +0.0]])

class_covariances = np.array([[[+0.8, -0.6], [-0.6, +0.8]],
                              [[+0.8, +0.6], [+0.6, +0.8]],
                              [[+0.8, -0.6], [-0.6, +0.8]],
                              [[+0.8, +0.6], [+0.6, +0.8]],
                              [[+0.2, +0.0], [+0.0, +1.2]],
                              [[+1.2, +0.0], [+0.0, +0.2]],
                              [[+0.2, +0.0], [+0.0, +1.2]],
                              [[+1.2, +0.0], [+0.0, +0.2]],
                              [[+1.6, +0.0], [+0.0, +1.6]]])

class_sizes = np.array([100, 100, 100, 100, 100, 100, 100, 100, 200])

N = X.shape[0]
K = initial_centroids.shape[0]

plt.figure(figsize=(6,6))
plt.plot(X[:, 0], X[:, 1], 'k.', markersize=6)
plt.xlabel('$x1$')
plt.ylabel('$x2$')
plt.show()

def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis=0)
    return memberships

def calculate_covariances(h_ik, means, X):
    covariances = []
    for c in range(K):
        result = np.zeros((2, 2))
        for i in range(N):
            covariance = np.matmul((X[i] - means[c])[:, None], (X[i] - means[c])[None, :]) * h_ik[c][i]
            result += covariance

        covariances.append(result / np.sum(h_ik[c], axis=0))

    return covariances

means = initial_centroids
memberships = update_memberships(means, X)

priors = [X[memberships == c].shape[0] / N for c in range(K)]

covariances = []
for c in range(K):
    result = np.zeros((2, 2))
    for i in range(X[memberships == c].shape[0]):
        covariance = np.matmul(((X[memberships == c])[i, :] - means[c, :])[:, None], ((X[memberships == c])[i, :] - means[c, :][None, :]))
        result += covariance

    covariances.append(result / X[memberships == c].shape[0])

for i in range(100):
    posterior_probabilites = np.array([stats.multivariate_normal(means[c], covariances[c]).pdf(X) * priors[c] for c in range(K)])
    h_ik = np.vstack([posterior_probabilites[c] / np.sum(posterior_probabilites, axis=0) for c in range(K)])

    means = np.vstack([np.matmul(h_ik[c], X) / np.sum(h_ik[c], axis=0) for c in range(K)])
    priors = np.vstack([np.sum(h_ik[c], axis=0) / N for c in range(K)])
    covariances = calculate_covariances(h_ik, means, X)

memberships = np.argmax(h_ik, axis=0)

print(means)

posterior_probabilites = np.array([stats.multivariate_normal(means[c], covariances[c]).pdf(X) * priors[c] for c in range(K)])
h_ik = np.vstack([posterior_probabilites[c] / np.sum(posterior_probabilites, axis=0) for c in range(K)])

x1 = np.linspace(-8, 8, 1601)  
x2 = np.linspace(-8, 8, 1601)
x, y = np.meshgrid(x1, x2) 
pos = np.dstack((x, y))

cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
plt.figure(figsize=(10, 10))

for c in range(K):
    true_classes = stats.multivariate_normal(class_means[c], class_covariances[c] * 2).pdf(pos)
    predicted_classes = stats.multivariate_normal(means[c], covariances[c] * 2).pdf(pos)

    plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10, color=cluster_colors[c])
    plt.contour(x, y, true_classes, levels=1, linestyles="dashed", colors="k")
    plt.contour(x, y, predicted_classes, levels=1, colors=cluster_colors[c])
    
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()