import numpy as np

X = np.array([
	[1, 2, 4, 7],
	[1, 3, 4, 8],
	[1, 11, 2, 9]
	])

y = np.array([1, 1, 0]);

# Sigmoid activation function
def sigmoid(array):
	return np.divide(1, np.add(1, np.power(np.e, np.multiply(array, -1))))

# All theta matricies initialized
theta_0 = np.random.rand(4, 4)
theta_1 = np.random.rand(5, 1)

def h(X, theta_0, theta_1):
	# First layer
	a_1 = X
	h_1 = sigmoid(a_1)

	# Second layer
	a_2 = np.sum(np.multiply(h_1, theta_0), axis = 1)
	h_2 = sigmoid(a_2)

	# Third layer
	a_3 = np.sum(np.multiply(h_2, theta_1))

	return sigmoid(a_3)

def cost(X, y, theta_0, theta_1):
	return np.power((h(X, theta_0, theta_1) - y), 2)

print(cost(X[2], y[2], theta_0, theta_1))