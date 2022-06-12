import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn")

# =========================================================================#
#                         Functions for KNN model                          #
# ======================================================================== #


# Euclidean distance
def distance(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    return np.sqrt(sum((x1 - x2)**2))

# Test the functionality of the distance function
sample_data = [[2.7810836,2.550537003],
           [1.465489372,2.362125076],
           [3.396561688,4.400293529],
           [1.38807019,1.850220317],
           [3.06407232,3.005305973],
           [7.627531214,2.759262235],
           [5.332441248,2.088626775],
           [6.922596716,1.77106367],
           [8.675418651,-0.242068655],
           [7.673756466,3.508563011]]

label = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
for data in sample_data:
    print(distance(sample_data[0], data))
# Seem reasonable since the distance between a point and itself is 0

# Neighbor selection
def neighbors_selection(training_set, test, k):
    distances = [(i, distance(test, x)) for i, x in enumerate(training_set)]
    distances.sort(key = lambda x: x[1])
    return [i[0] for i in distances[:k]]

# Testing the functionality of neighbors_selection function
print(neighbors_selection(sample_data, sample_data[0], k=7))

# Prediction
def predict(neighbor_indices, label):
    label = np.array(label)
    neighbor_label = label[neighbor_indices]
    prediction = {}
    for x in neighbor_label:
        if x in prediction:
            prediction[x] += 1
        else:
            prediction[x] = 1
    total = sum(prediction.values())
    prob_pred = {k: v/total for k, v in prediction.items()}
    return prob_pred

print(predict([0, 4, 1, 3, 2, 6, 7], label))

def KNN(training_set, label, test_set, k):
    """

    :param training_set: 2-D array that store training instances
    :param label: 1-D array that store the label for each point in training set
    :param test_set: 2-D array that store testing instances
    :param k: hyperparameter for number of clusters
    :return: predicted label for each point in the datapoint set
    """
    result = []
    for instance in test_set:
        neighbor_indices = neighbors_selection(training_set, instance, k)
        prediction = predict(neighbor_indices, label)
        # Retrieve the label of the prediction dict using .get() method
        result.append(max(prediction, key=prediction.get))

    return np.array(result)

# Testing the functionality of KNN function
print(KNN(sample_data[1:], label, [sample_data[0]], 3))

# ==================================================================== #
#                         Testing the KNN model                        #
# ==================================================================== #

# Here, we will use the Iris dataset from sklearn library
from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target

print(iris_data)
print(iris_label)

# Function to create train-test set
def train_test_split(X, y, test_size, random_seed):
    num = len(y)
    test_len = int(test_size * num)
    np.random.seed(random_seed)
    index = np.random.permutation(num)
    X_train = X[index[:-test_len]]
    X_test = X[index[-test_len:]]
    y_train = y[index[:-test_len]]
    y_test = y[index[-test_len:]]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, 0.2, 29)

print(X_test.shape)
print(y_test.shape)

# test set dimension: 30 x 4 matrix => 4 features and 30 obs
# Choosing k = 10
y_pred = KNN(X_train, y_train, X_test, 10)
print(y_pred)
print(y_test)

# Function to extract the accuracy
def accuracy(y_true, y_pred):
    match = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            match += 1

    return (match / len(y_true)) * 100

print(f'The accuracy score of our KNN model is: {accuracy(y_test, y_pred):.2f} %')



# ==================================================================== #
#                         Tuning the KNN model                         #
# ==================================================================== #

def hyperparameter_tune(X_train, y_train, X_test, y_test, max_k):
    accuracy_score = []
    y_test = y_test
    for i in range(1, max_k):
        y_pred = KNN(X_train, y_train, X_test, i)
        accuracy_score.append(accuracy(y_test, y_pred))

    return np.array(accuracy_score)

accuracy_score = hyperparameter_tune(X_train, y_train, X_test, y_test, 100)
print(accuracy_score)

def hyperparameter_checkboard(fig, ax, accuracy_list, max_k, linewidth = 1, linecolor = "red"):
    x = np.arange(1, max_k)
    y = accuracy_list
    ax.plot(x, y, color = linecolor, linewidth = linewidth, label = "Accuracy")
    ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    return ax

fig, ax = plt.subplots(figsize = (10,5))
hyperparameter_checkboard(fig, ax, accuracy_score, 100)
plt.show()

