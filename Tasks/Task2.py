import numpy as np
import matplotlib.pyplot as plt


def mse_loss_ab(a, b, x, y):
    f_theta = a * x + b  # f_theta(x) = a*x + b
    return np.mean((f_theta - y) ** 2)

np.random.seed(42)
x_train = np.arange(0.0, 1.0, 0.025)
y_train = 0.4 + x_train * 0.55 + np.random.randn(x_train.shape[0]) * 0.2

a_values = np.linspace(-1, 1, 1000)

b_1 = 0.1

loss_values_b1 = [mse_loss_ab(a, b_1, x_train, y_train) for a in a_values]

b_2 = 2.0
loss_values_b2 = [mse_loss_ab(a, b_2, x_train, y_train) for a in a_values]
# Use a fixed range for x and calculate loss-consistent predictions
x_loss = np.linspace(-1, 1, 500)
y_pred_loss_b1 = a_values[np.argmin(loss_values_b1)] * x_loss + b_1
y_pred_loss_b2 = a_values[np.argmin(loss_values_b2)] * x_loss + b_2

# Plot loss-consistent data
plt.figure(figsize=(12, 6))

# Original training data
plt.scatter(x_train, y_train, label="Training Data", color="blue", alpha=0.6)

# Prediction for b = 0.1
plt.plot(x_loss, y_pred_loss_b1, label=f"Prediction (b = {b_1})", color="orange")

# Prediction for b = 2.0
plt.plot(x_loss, y_pred_loss_b2, label=f"Prediction (b = {b_2})", color="green")

# Labels, title, and legend
plt.title("Consistent Predictions Based on Loss Curve", fontsize=14)
plt.xlabel("$x$", fontsize=12)
plt.ylabel("$y$", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Show plot
plt.show()
