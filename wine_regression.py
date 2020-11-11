import numpy as np
import time


# Perform feature scaling using mean and standard deviation
def feature_scaling(X):
    m = X.shape[0]
    means = np.mean(X, axis=0)
    X = X - means
    sd = np.std(X, axis=0)
    X = X / sd
    return X


def compute_cost_and_gradient(theta, X, Y):
    # m = number of training examples
    m = X.shape[0]
    # compute cost
    J = np.sum(np.power(np.dot(X, theta) - Y, 2)) / (2 * m)

    # compute gradient
    gradients = (np.dot((np.dot(X, theta) - Y).T, X).T) / m

    return J, gradients


# One step of gradient descent, logging can be turned on or off via flag
def perform_gradient_descent(theta, X, Y, alpha, iterations, logging=False):
    for i in range(iterations + 1):
        start = time.time()
        J, gradients = compute_cost_and_gradient(theta, X, Y)
        theta = theta - alpha * gradients
        if  i%50 == 0 and logging:
            log_progress(i, iterations, start, J)
    return J, theta


# Log progress per step (optional)
def log_progress(current_iteration, iterations, start_time, J):
    stop = time.time()
    print('Estimated time remaining: ' + str((stop - start_time) * 1000 * (iterations - current_iteration)) + 'ms')
    print('Current Cost: ' + str(J))
    print('Performing gradient descent (step ' + str(current_iteration) + '/' + str(iterations) + ')\n')


# Use learned parameters theta to predict values for test set
# Outputs accuracy for different tolerance values as well as mean average deviation
def predict(theta, X, Y):
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    m = X.shape[0]
    predictions = np.dot(X, theta)
    deviations = np.abs(predictions - Y)

    tolerances = [0.25, 0.5, 1]
    accuracies = [0, 0, 0]

    for x in tolerances:
        for i in range(0, m):
            if deviations[i] <= x:
                accuracies[tolerances.index(x)] += 1
        print('Accuracy (tolerance = ' + str(x) + '): ' + format(float(accuracies[tolerances.index(x)]) / m * 100,
                                                                 '.2f') + '%')

    mad = np.mean(np.abs(predictions - Y))
    print('Mean Average Deviation (MAD) on test set: ' + str(mad) + '\n')


# Importing dataset from CSV file, splitting data into train/test set
filename = 'winequality-red.csv'
raw_file = open(filename, 'rt')
raw_data = np.loadtxt(raw_file, delimiter=';')
(number_examples, feature) = raw_data.shape

# Shuffle Array to prevent bias in the training/test set
np.random.shuffle(raw_data)

data_train = raw_data[0:1119, 0:11]
data_test = raw_data[1119:number_examples, 0:11]

Y_train = raw_data[0:1119, 11:12]
Y_test = raw_data[1119:number_examples, 11:12]

# setting hyperparameters
alpha = 0.1
max_iterations = 1000

# perform feature scaling
data_train = feature_scaling(data_train)
data_test = feature_scaling(data_test)

# Extend both data_train/data_test with X0 = 1, respectively
x0 = np.ones((data_train.shape[0], 1))
data_train = np.append(x0, data_train, axis=1)
x0 = np.ones((data_test.shape[0], 1))
data_test = np.append(x0, data_test, axis=1)

# Set initial values of Theta to 0
theta = np.zeros((data_train.shape[1], 1))

# Perform Gradient Descent using training set
J, final_theta = perform_gradient_descent(theta, data_train, Y_train, alpha, max_iterations, True)

# Performance check using test set
predict(final_theta, data_test, Y_test)
