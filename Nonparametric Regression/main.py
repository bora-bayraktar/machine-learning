import numpy as np
import matplotlib.pyplot as plt

# Import Data
training_data = np.genfromtxt("hw04_data_set_train.csv", delimiter=",")
test_data = np.genfromtxt("hw04_data_set_test.csv", delimiter=",")

# Training Data
x_training = training_data[:, 0]
y_training = training_data[:, 1]

# Test Data
x_test = test_data[:, 0]
y_test = test_data[:, 1]

# Setting Parameters
bin_width = 0.1
origin = 0.0

# Calulating Regressogram
x_max = np.max(x_training)

left_borders = np.arange(origin, x_max, bin_width)
right_borders = np.arange(origin + bin_width, x_max + bin_width, bin_width)
data_interval = np.linspace(origin, x_max, 1601)

regressogram_results = np.array([np.sum(((left_borders[i] < x_training) & (x_training <= right_borders[i])) * y_training) / np.sum((left_borders[i] < x_training) & (x_training <= right_borders[i])) for i in range(len(left_borders))])

# Plotting Training Points
plt.figure(figsize=(10,4))
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.plot(x_training, y_training, "b.", label="training", markersize=8)
plt.legend()

for i in range(len(left_borders)):
    plt.plot([left_borders[i], right_borders[i]], [regressogram_results[i], regressogram_results[i]], "k-")

for i in range(len(left_borders) - 1):
    plt.plot([right_borders[i], right_borders[i]], [regressogram_results[i], regressogram_results[i + 1]], "k-")    

plt.xticks(np.linspace(0.0, 2.0, 9))
plt.yticks(np.linspace(-1.0, 2.0, 7))
plt.show()

# Plotting Test Points
plt.figure(figsize=(10,4))
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.plot(x_test, y_test, "r.", label="test", markersize=8)
plt.legend()

for i in range(len(left_borders)):
    plt.plot([left_borders[i], right_borders[i]], [regressogram_results[i], regressogram_results[i]], "k-")

for i in range(len(left_borders) - 1):
    plt.plot([right_borders[i], right_borders[i]], [regressogram_results[i], regressogram_results[i + 1]], "k-")    

plt.xticks(np.linspace(0.0, 2.0, 9))
plt.yticks(np.linspace(-1.0, 2.0, 7))
plt.show()

# Calculating Root Mean Squared Error (RMSE)
def rmse(N, y_truth, y_pred):
    return np.sqrt(np.sum((y_truth - y_pred)**2) / N)

y_pred = np.array([np.sum((((x - 0.5 * bin_width) < x_training) & (x_training <= (x + 0.5 * bin_width))) * y_training) / np.sum(((x - 0.5 * bin_width) < x_training) & (x_training <= (x + 0.5 * bin_width))) for x in x_test])
print("Regressogram => RMSE is", str(rmse(y_test.shape[0], y_test, y_pred)), "when h is", bin_width)

# Running Mean Smoother
def w(u):
    return np.abs(u) <= 0.5

bin_width_smoother = 0.1
running_mean_smoother = np.array([np.sum(w((x - x_training) / bin_width_smoother) * y_training) / np.sum(w((x - x_training) / bin_width_smoother)) for x in data_interval])

# Plotting Training Points
plt.figure(figsize=(10,4))
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.plot(x_training, y_training, "b.", label="training", markersize=8)
plt.plot(data_interval, running_mean_smoother, "k-")
plt.legend()
plt.xticks(np.linspace(0.0, 2.0, 9))
plt.yticks(np.linspace(-1.0, 2.0, 7))
plt.show()

# Plotting Test Points
plt.figure(figsize=(10,4))
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.plot(x_test, y_test, "r.", label="test", markersize=8)
plt.plot(data_interval, running_mean_smoother, "k-")
plt.legend()
plt.xticks(np.linspace(0.0, 2.0, 9))
plt.yticks(np.linspace(-1.0, 2.0, 7))
plt.show()

# Calculating Root Mean Squared Error (RMSE)
y_pred = np.array([np.sum(w((x - x_training) / bin_width_smoother) * y_training) / np.sum(w((x - x_training) / bin_width_smoother)) for x in x_test])
print("Running Mean Smoother => RMSE is", str(rmse(y_test.shape[0], y_test, y_pred)), "when h is", bin_width_smoother)

# Kernel Smoother
def K(u):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * u**2)

bin_width_kernel = 0.02
kernel_smoother = np.array([np.sum(K((x - x_training) / bin_width_kernel) * y_training) / np.sum(K((x - x_training) / bin_width_kernel)) for x in data_interval])

# Plotting Training Points
plt.figure(figsize=(10,4))
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.plot(x_training, y_training, "b.", label="training", markersize=8)
plt.plot(data_interval, kernel_smoother, "k-")
plt.legend()
plt.xticks(np.linspace(0.0, 2.0, 9))
plt.yticks(np.linspace(-1.0, 2.0, 7))
plt.show()

# Plotting Test Points
plt.figure(figsize=(10,4))
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.plot(x_test, y_test, "r.", label="test", markersize=8)
plt.plot(data_interval, kernel_smoother, "k-")
plt.legend()
plt.xticks(np.linspace(0.0, 2.0, 9))
plt.yticks(np.linspace(-1.0, 2.0, 7))
plt.show()

# Calculating Root Mean Squared Error (RMSE)
y_pred = np.array([np.sum(K((x - x_training) / bin_width_kernel) * y_training) / np.sum(K((x - x_training) / bin_width_kernel)) for x in x_test])
print("Kernel Smoother => RMSE is", str(rmse(y_test.shape[0], y_test, y_pred)), "when h is", bin_width_kernel)