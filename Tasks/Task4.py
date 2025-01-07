import numpy as np
from numba import njit
import plotly.express as px

@njit
def predict(theta, xs):
    """Predicts output using current theta."""
    return np.dot(xs, theta)

@njit
def J_squared_residual(theta, xs, y):
    """Calculates the squared residual (L2 Loss)."""
    h = predict(theta, xs)
    sr = ((h - y) ** 2).sum()
    return sr

@njit
def gradient_J_squared_residual(theta, xs, y):
    """Calculates the gradient of the squared residual."""
    h = predict(theta, xs)
    grad = np.dot(xs.T, (h - y))
    return grad

@njit
def mini_batch_gradient_descent(data_x, data_y, theta, learning_rate, n_iters, batch_size):
    """Performs mini-batch gradient descent."""
    m = data_x.shape[0]
    j_history = np.zeros(n_iters * (m // batch_size))  # Pre-allocate loss history
    history_idx = 0

    for it in range(n_iters):
        # Shuffle the data
        indices = np.arange(m)
        np.random.shuffle(indices)
        data_x_shuffled = data_x[indices]
        data_y_shuffled = data_y[indices]

        for i in range(0, m, batch_size):
            end_idx = min(i + batch_size, m)
            data_x_batch = data_x_shuffled[i:end_idx]
            data_y_batch = data_y_shuffled[i:end_idx]

            # Compute gradient and update theta
            grad = gradient_J_squared_residual(theta, data_x_batch, data_y_batch)
            theta -= (learning_rate / batch_size) * grad

            # Compute loss for the batch
            j = J_squared_residual(theta, data_x, data_y)
            j_history[history_idx] = j
            history_idx += 1

    return theta, j_history

# Dataset and initial variables
data_x = np.array([[1.0, 0.5], [1.0, 1.0], [1.0, 2.0]])
data_y = np.array([[1.0], [1.5], [2.5]])
n_features = data_x.shape[1]
theta = np.zeros((n_features, 1))
learning_rate = 0.1
batch_size = 2
n_iters = 10

# Run Mini-Batch Gradient Descent
theta, j_history = mini_batch_gradient_descent(data_x, data_y, theta, learning_rate, n_iters, batch_size)

# Final Loss
final_loss = J_squared_residual(theta, data_x, data_y)
print(f"The final L2 error is: {final_loss:.2f}")

# Predictions
y_pred = predict(theta, data_x)

# L1 Error
l1_error = np.abs(y_pred - data_y).sum()
print(f"The L1 error is: {l1_error:.2f}")

# R^2 Score
u = ((data_y - y_pred) ** 2).sum()
v = ((data_y - data_y.mean()) ** 2).sum()
r_squared = 1 - (u / v)
print(f"R^2: {r_squared:.2f}")

# Plot Loss History
fig = px.line(y=j_history, title="J(theta) - Loss History")
fig.show()
