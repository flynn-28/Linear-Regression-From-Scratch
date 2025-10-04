import numpy as np # import NumPy library for linear algebra
import matplotlib.pyplot as plt # import for graphing

def generate_data(samples = 1000, features = 3):                  # define data generation function with input for samples and features
    X = np.random.randn(samples, features)                        # generate randomized features
    weights = np.arange(1, features + 1).reshape(-1, 1)           # set random weights for target
    bias = np.random.randn()                                      # set a random bias for target
    y = X.dot(weights) + bias + 0.5 * np.random.randn(samples, 1) # generate target with linear equation
    return X, y                                                   # return features and target

def initialize_parameters(features):
    weights = np.zeros((features, 1)) # initialize weights as zeros
    bias = 0.0                        # initializ bias as zero
    return weights, bias              # return weights and bias

# predicts based on weights and bias using a linear model
def predict(X, weights, bias):
    return X.dot(weights) + bias    # return prediction
	
# computes the mean squared error between the predicted value and the actual value
def compute_loss(prediction, actual):
    return np.mean((prediction - actual)**2)    # return loss
	
def train(X, y, learning_rate=0.1, epochs=20):
    samples, features = X.shape                     # get number of samples and features from input shape
    weights, bias = initialize_parameters(features) # initalize weights and bias
    loss_history = []                               # a list to keep track of loss values

    # iterate for each epoch
    for epoch in range(epochs):
        # forward pass
        y_prediction = predict(X, weights, bias) # predict output with current weights and bias
        loss = compute_loss(y_prediction, y)     # compute loss with prediction and actual
        loss_history.append(loss)                # store loss

        # backpropagation
        derivative_weights = (2 / samples) * X.T.dot(y_prediction - y) # change in loss based on weights
        derivative_bias = (2 / samples) * np.sum(y_prediction - y)     # change in loss based on bias

        # update weights and bias
        weights -= learning_rate * derivative_weights 
        bias -= learning_rate * derivative_bias

        # print loss for every tenth of total epochs
        if epoch % (epochs // 10) == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")


    return weights, bias, loss_history  # return updated weights, updated bias, and loss history


X, y = generate_data() # generate a dataset with 1000 samples and 3 features

print('X shape:', X.shape) # print the shape of the features
print('y shape:', y.shape) # print the shape of the target

# Train model
learned_weights, learned_bias, loss_history = train(X, y)

# print results from training
print("-" * 30)
print("Learned weights:", learned_weights.ravel())
print("Learned bias:", learned_bias)

plt.plot(loss_history)     # plot loss history
plt.xlabel("Epoch")        # label the X axis
plt.ylabel("MSE")          # label the Y axis
plt.title("Training Loss") # Add a title to the chart
plt.show()                 # display the chart

predictions = predict(X, learned_weights, learned_bias) # predict y values

plt.scatter(y, predictions, alpha=0.3)                  # plot predictions vs actual, make points translucent
plt.xlabel("Actual values")                             # label the graphs x axis
plt.ylabel("Predicted values")                          # label the graphs y axis
plt.title("Predicted vs Actual")                        # add a title to the graph
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--") # plot a y=x line for refrence
plt.show()

print(f"Line of best fit: y = {learned_weights[0,0]:.4f} * x + {learned_bias:.4f}") # calculate and print the line of best fit
