import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
plt.style.use("seaborn")
np.random.seed(28)

# Sample data for our implementation, which will be easily illustrated in the R2 space
sample_data = np.array([[2.7810836,2.550537003],
           [1.465489372,2.362125076],
           [3.396561688,4.400293529],
           [1.38807019,1.850220317],
           [3.06407232,3.005305973],
           [7.627531214,2.759262235],
           [5.332441248,2.088626775],
           [6.922596716,1.77106367],
           [8.675418651,-0.242068655],
           [7.673756466,3.508563011]])

label = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


# Function to find the distributions for each class that most fits to the given data through MAP

def class_division(X, y):
    """

    :param X: Nd-array represents the features set
    :param y: Nd-array represents
    :return: a dict that store classes as keys and entries features set as values
    """

    class_dict = {}
    for i, label in enumerate(y):
        if label in class_dict:
            class_dict[label].append(i)
        else:
            class_dict[label] = [i]

    for k, v in class_dict.items():
        class_dict[k] = X[v]
    return class_dict

# Testing functionality of class_division function
class_dict = class_division(sample_data, label)
print(class_dict) # Class_dict comprises keys as class and values from sample_data, which is
# identical to the sample_data

# Function to extract the mean and sd of each class => distribution

# Note that: * means "unpack" the list to make each of its elements as a separate arguments
def summary(dataset):
    """

    :param dataset: Nested lists as input, i.e., in our case is the sample_data
    :return: a nested list that contains the mean and std of each column of the sample_data
    """
    return [[np.mean(column), np.std(column)] for column in zip(*dataset)]

print(summary(sample_data))

# Combining two defined functions above to become a new function called summary_on_class

def summary_on_class(class_dict):
    summary_dict = {}
    for k, v in class_dict.items():
        summary_dict[k] = summary(v)
    return summary_dict

print(summary_on_class(class_dict))

# Let's make some visualization to grasp a better understanding of these moments we get
x1, x2 = np.random.RandomState(8).multivariate_normal([2.4190554339999997, 2.8336963796], [(0.833648422388659, 0), (0, 0.8664248811868022)], 10000).T
df = pd.DataFrame({"x1":x1,"x2":x2})
sns.jointplot(data=df, x='x1', y='x2', kind='kde', color='skyblue')
plt.show()

# ==================================================================== #
#                         Naive Bayes model                            #
# ==================================================================== #

# Create function to define our prior belief

def prior_belief(class_dict, y):
    """

    :param class_dict: a dict store information about labels and features
    :param y: the label
    :return: a dict that stores labels and our prior specification which is simply
            count number of data points corresponding to each class to create a categorical distribution
    """
    prior_dict = {}
    total = len(y)
    for k, v in class_dict.items():
        prior_dict[k] = len(v) / total

    return prior_dict

print(prior_belief(class_dict, label))

def likelihood(class_dict, test_set):
    """

    :param class_dict: a dict store information about labels and features
    :param test_set: a test set
    :return: the likelihood that a specific values belong to a class in test set using
            likelihood function with parameters from summary_on_class function
    """

    likelihood_dict = {}
    feature_params = summary_on_class(class_dict)
    for k in feature_params.keys():
        value = feature_params[k]
        for i, feature in enumerate(value):
            if k in likelihood_dict:
                likelihood_dict[k] *= norm(feature[0], feature[1]).pdf(test_set[i])
            else:
                likelihood_dict[k] = norm(feature[0], feature[1]).pdf(test_set[i])

    return likelihood_dict

# Testing the functionality of the Likelihood function

print(likelihood(class_dict, [5.33244125,  2.08862677]))

# Function to make prediction

def predict(train_set, label, test_set):
    class_dict = class_division(train_set, label)
    class_prob = prior_belief(class_dict, label)
    likelihood_dict = likelihood(class_dict, test_set)
    pred = {k: class_prob[k] * likelihood_dict[k] for k in class_prob}

    return max(pred.keys(), key = lambda k: pred[k])

# Testing the functionality of predict function
# Since we know that sample_data at index 5 belonging to class 1, then
print(predict(sample_data, label, sample_data[5]))
assert predict(sample_data, label, sample_data[5]) == 1

def NBGaussian(train_set, label, test_set):
    prediction = []
    for val in test_set:
        prediction.append(predict(train_set, label, val))
    return np.array(prediction)


# ==================================================================== #
#                        Testing Naive Bayes model                     #
# ==================================================================== #

# We will use a conventional multiclass dataset (wine dataset) from sklearn
# The dataset has three classes, labeled as 0 to 2 and thirteen feature columns

from sklearn import datasets

wine_X, wine_y = datasets.load_wine(return_X_y=True)

# Using the function train_test_split from KNN post
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

X_train, X_test, y_train, y_test = train_test_split(wine_X, wine_y, 0.2, 28)
assert X_train.shape[1] == 13

# Perform prediction
print('Begin predicting')
y_pred = NBGaussian(X_train, y_train, X_test)
print(y_pred)
print(y_test)
print(y_pred == y_test)

# Different implementation from our accuracy score before
def accuracy_score(y_true, y_pred):
    return (sum(y_true == y_pred)/len(y_true)) * 100

print(accuracy_score(y_test, y_pred))

# Examining the result from Sklearn GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred_sklearn = model.predict(X_test)

print("Table 1: Compare results between Sklearn and our implementation \n")
print(pd.DataFrame({'Sklearn':y_pred_sklearn,
                    'Our':y_pred,
                    'True class':y_test}))
print("\n")

print("Accuracy score between two models \n")
print(pd.DataFrame([{'Accuracy of Sklearn model':accuracy_score(y_test, y_pred_sklearn),
                    'Accuracy of Our model':accuracy_score(y_test, y_pred)}]))

