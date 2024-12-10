import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Gradient calculation for loss
def gradient_of_J(theta, x, y):
    y_hat = theta
    dLdy = (y_hat - y)  # dL / dy_hat
    dy_HatdTheta = 1  # dy_hat / dTheta
    dLdTheta = dLdy * dy_HatdTheta  # chain rule
    return dLdTheta


# Mean Squared Error loss calculation
def calculate_l2_loss_non_vectorized(theta, xs, ys):
    loss = 0.0
    for k in range(ys.shape[0]):
        y_pred = theta
        loss += (y_pred - ys[k]) ** 2

    mean_loss = loss / ys.shape[0]
    return mean_loss


# Gradient descent function with a fixed learning rate
def gradient_descent(initial_theta, x_train, y_train, learning_rate, n_steps):
    theta = np.array([initial_theta])
    m = x_train.shape[0]
    search_history = []

    for steps in range(n_steps):
        gradient_theta_sum = 0.0
        for k in range(m):
            gradient_theta_sum += gradient_of_J(theta, x_train[k], y_train[k])

        mean_gradient = (1 / m) * gradient_theta_sum
        loss = calculate_l2_loss_non_vectorized(theta, x_train, y_train)

        # Update theta
        theta = theta - (learning_rate * mean_gradient)
        search_history.append((theta[0], loss))

    final_loss = calculate_l2_loss_non_vectorized(theta, x_train, y_train)
    return search_history, final_loss


# Hyperparameter: range of learning rates
learning_rates = np.arange(0, 2.02, 0.0005)  # Check every learning rate from 0 to 2 with 0.02 steps increments
initial_theta = 5.5
n_steps = 10
x_train = np.arange(0.0, 1.0, 0.025)
np.random.seed(42)
m = x_train.shape[0]
y_train = 0.4 + x_train * 0.55 + np.random.randn(x_train.shape[0]) * 0.2

best_lr = None
lowest_loss = float('inf')
loss_results = []

# Learning rate search
for lr in learning_rates:
    search_history, final_loss = gradient_descent(initial_theta, x_train, y_train, lr, n_steps)
    loss_results.append((lr, final_loss[0]))  # Extract scalar from 1D array
    print(f"Learning rate {lr:.4f} => Final loss: {final_loss}")

    if final_loss < lowest_loss:
        lowest_loss = final_loss
        best_lr = lr

print(f"Best learning rate: {float(best_lr)} with loss: {lowest_loss}")

# Convert list of tuples to a structured numpy array
loss_results = np.array(loss_results, dtype=[('learning_rate', float), ('final_loss', float)])

# Plot the loss landscape
loss_x = np.arange(-4, 6, 0.01)
loss_y = np.array([calculate_l2_loss_non_vectorized(t, x_train, y_train) for t in loss_x])
fig = px.line(x=loss_x, y=loss_y, title="GD History for Best Learning Rate")

# Visualize GD history for the best learning rate
search_history, _ = gradient_descent(initial_theta, x_train, y_train, best_lr, n_steps)
x_visit, _ = list(zip(*search_history))
x_visit = np.array(x_visit)
y_visit = np.array([calculate_l2_loss_non_vectorized(t, x_train, y_train) for t in x_visit])

fig.add_trace(go.Scatter(x=x_visit, y=y_visit, name='GD history for Best LR',
                         line=dict(color='firebrick', width=8, dash='dot')))
fig.show()

# Plot learning rate vs. final loss
fig_lr = px.line(x=loss_results['learning_rate'], y=loss_results['final_loss'], log_x=True,
                 title="Learning Rate vs. Final Loss",
                 labels={"x": "Learning Rate", "y": "Final Loss"})
fig_lr.show()
