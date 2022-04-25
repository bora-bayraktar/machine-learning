import numpy as np
import matplotlib.pyplot as plt

training_data = np.genfromtxt("hw05_data_set_train.csv", delimiter=",")
test_data = np.genfromtxt("hw05_data_set_test.csv", delimiter=",")

# Training Data
x_training = training_data[:, 0]
y_training = training_data[:, 1]

# Test Data
x_test = test_data[:, 0]
y_test = test_data[:, 1]

N_train = x_training.shape[0]
N_test = x_test.shape[0]

def decision_tree(P):
    # Create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_splits = {}
    node_means = {}

    # Put all training instances into the root node
    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    need_split[1] = True

    # Learning algorithm
    while True:
        # Find nodes that need splitting
        split_nodes = [key for key, value in need_split.items() if value == True]
        # Check whether we reach all terminal nodes
        if len(split_nodes) == 0:
            break
        # Find best split positions for all nodes
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            node_mean = np.mean(y_training[data_indices])

            if len(x_training[data_indices]) <= P:
                is_terminal[split_node] = True
                node_means[split_node] = node_mean
            else:
                is_terminal[split_node] = False

                unique_values = np.sort(np.unique(x_training[data_indices]))
                split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
                split_scores = np.repeat(0.0, len(split_positions))

                for s in range(len(split_positions)):
                    left_indices = data_indices[x_training[data_indices] < split_positions[s]]
                    right_indices = data_indices[x_training[data_indices] >= split_positions[s]]

                    error = 0
                    if len(left_indices) > 0:
                        error += np.sum((y_training[left_indices] - np.mean(y_training[left_indices]))**2)
                    if len(right_indices) > 0:
                        error += np.sum((y_training[right_indices] - np.mean(y_training[right_indices]))**2)

                    split_scores[s] = error / (len(left_indices) + len(right_indices))

                best_split = split_positions[np.argmin(split_scores)]
                node_splits[split_node] = best_split

                # Create left node using the selected split
                left_indices = data_indices[x_training[data_indices] < best_split]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                # Create right node using the selected split
                right_indices = data_indices[x_training[data_indices] >= best_split]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True

    return is_terminal, node_splits, node_means

def make_prediction(is_terminal, node_splits, node_means, value):
    index = 1
    while(True):
        if is_terminal[index] == True:
            return node_means[index]
        else:
            if value <= node_splits[index]:
                index = 2*index
            else:
                index = 2*index + 1

data_points = np.linspace(np.min(x_training), np.max(x_training), 1601)
P = 30
is_terminal, node_splits, node_means = decision_tree(P)
y_pred = [make_prediction(is_terminal, node_splits, node_means, data_points[i]) for i in range(data_points.shape[0])]

# Plotting Training Points
plt.figure(figsize=(10,4))
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.plot(x_training, y_training, "b.", label="training", markersize=8)
plt.plot(data_points, y_pred, "k")
plt.legend()
plt.xticks(np.linspace(0.0, 2.0, 9))
plt.yticks(np.linspace(-1.0, 2.0, 7))
plt.show()

# Plotting Test Points
plt.figure(figsize=(10,4))
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.plot(x_test, y_test, "r.", label="test", markersize=8)
plt.plot(data_points, y_pred, "k")
plt.legend()
plt.xticks(np.linspace(0.0, 2.0, 9))
plt.yticks(np.linspace(-1.0, 2.0, 7))
plt.show()

# Calculating Root Mean Squared Error (RMSE)
def rmse(N, y_truth, y_pred):
    return np.sqrt(np.sum((y_truth - y_pred)**2) / N)

# RMSE of training set
y_pred = [make_prediction(is_terminal, node_splits, node_means, x_training[i]) for i in range(N_train)]
print("RMSE on training set is", str(rmse(y_training.shape[0], y_training, y_pred)), "when P is", P)

# RMSE of test set
y_pred = [make_prediction(is_terminal, node_splits, node_means, x_test[i]) for i in range(N_test)]
print("RMSE on test set is", str(rmse(y_test.shape[0], y_test, y_pred)), "when P is", P)

# Calculatin RMSE of test and training points
P_values = np.arange(10,51,5)
predictions = np.zeros((len(P_values), N_train))
for p in range(len(P_values)):
    is_terminal, node_splits, node_means = decision_tree(P_values[p])
    y_pred = [make_prediction(is_terminal, node_splits, node_means, x_training[i]) for i in range(N_train)]
    predictions[p] = y_pred

rmse_values_training = [rmse(y_training.shape[0], y_training, predictions[i]) for i in range(P_values.shape[0])]

P_values = np.arange(10,51,5)
predictions = np.zeros((len(P_values), N_test))
for p in range(len(P_values)):
    is_terminal, node_splits, node_means = decision_tree(P_values[p])
    y_pred = [make_prediction(is_terminal, node_splits, node_means, x_test[i]) for i in range(N_test)]
    predictions[p] = y_pred

rmse_values_test = [rmse(y_test.shape[0], y_test, predictions[i]) for i in range(P_values.shape[0])]

# Plotting RMSE
plt.figure(figsize=(10,4))
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.plot(P_values, rmse_values_training, "b.-", label="training", markersize=8)
plt.plot(P_values, rmse_values_test, "r.-", label="test", markersize=8)
plt.legend()
plt.show()