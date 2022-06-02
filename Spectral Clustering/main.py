import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
import scipy.linalg as linalg

X = np.genfromtxt("hw09_data_set.csv", delimiter=",")

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

plt.figure(figsize=(6, 6))
plt.plot(X[:, 0], X[:, 1], 'k.', markersize=6)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

N = X.shape[0]
delta = 2.0

D = spa.distance_matrix(X, X)
B = np.zeros((N, N)).astype(int)
B[D < delta] = 1
for i in range(N):
    B[i, i] = 0

plt.figure(figsize=(6, 6))

for i in range(N):
    for j in range(i + 1, N):
        if B[i][j] == 1:
            x1 = [X[i, 0], X[j, 0]]
            x2 = [X[i, 1], X[j, 1]]
            plt.plot(x1, x2, "-", color="grey", linewidth=0.5)

plt.plot(X[:, 0], X[:, 1], "k.", markersize=6)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

D = np.zeros((N, N))
for i in range(N):
    D[i, i] = B.sum(axis=1)[i]

I = np.identity(N)
D_inverse_squarred = np.sqrt(np.linalg.matrix_power(D, -1))
L_symmetric = I - np.matmul(D_inverse_squarred, np.matmul(B, D_inverse_squarred))
print(L_symmetric[0:5, 0:5])

eigenvalues, eigenvectors = linalg.eig(L_symmetric)
R = 5
Z = eigenvectors[:, np.argsort(eigenvalues)[1: R + 1]]
print(Z[0:5, 0:5])

K = 9

def update_centroids(memberships, X):
    centroids = np.vstack([np.mean(X[memberships == k, :], axis=0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis=0)
    return(memberships)

indices = [242, 528, 570, 590, 648, 667, 774, 891, 955]
centroids = Z[indices]
memberships = update_memberships(centroids, Z)

while True:
    old_centroids = centroids
    centroids = update_centroids(memberships, Z)
    if np.alltrue(centroids == old_centroids):
        break

    old_memberships = memberships
    memberships = update_memberships(centroids, Z)
    if np.alltrue(memberships == old_memberships):
        break

centroids = update_centroids(memberships, X)

cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
plt.figure(figsize=(6, 6))

for c in range(K):
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=6, color=cluster_colors[c])
    plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize=10, markerfacecolor=cluster_colors[c], markeredgecolor="black")

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
