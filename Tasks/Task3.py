import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from Tasks.Task1 import best_loss


#Gradient er innen matematikk en vektor som angir hvor raskt og i hvilken retning en funksjon endrer seg.
# y_hat - y = gradient. for å finne gradient så må vi derivere loss function.
def gradient_of_J(theta, x, y):
    #
    y_hat = theta
    # dL / dy_hat
    dLdy = (y_hat - y)
    # dy_hat / dTheta
    dy_HatdTheta = 1
    # chain rule
    dLdTheta = dLdy * dy_HatdTheta
    return dLdTheta

# avg(mse)
def calculate_l2_loss_non_vectorized(theta, xs, ys):
    loss = 0.0
    for k in range(ys.shape[0]):
        y_pred = theta
        loss += (y_pred - ys[k]) ** 2

    mean_loss = loss / ys.shape[0]
    return mean_loss

#brukes bare i gradient for loss.
initial_theta = 5.5
#x_train = y_train - støy
x_train = np.arange(0.0, 1.0, 0.025)

#
learning_rate = 1
theta = np.array([initial_theta])
np.random.seed(42)
m = x_train.shape[0]
y_train = 0.4 + x_train * 0.55 + np.random.randn(x_train.shape[0]) * 0.2
#print("y_train: ", y_train)

n_steps = 19

print("Running GD with initial theta: {:.2f}, learning rate: {} over {} datapoints for {} steps".format(
    theta.item(),
    learning_rate,
    m,
    n_steps))

search_history = []
#For loop som finner mean gradient og mean loss for alle predicted punktene.
# Oppdaterer også theta ved hjelp av gradient * learningrate.
#legger inn en liste med punkter i search_history(theta, Loss)
for steps in range(n_steps):

    gradient_theta_sum = 0.0
    for k in range(m):
        gradient_theta_sum += gradient_of_J(theta, x_train[k], y_train[k])

    mean_gradient = (1 / m) * gradient_theta_sum
    loss = calculate_l2_loss_non_vectorized(theta, x_train, y_train)

    #print(
     #   "[step {}] theta: {:.2f} => loss: {:.2f}".format(steps, theta.item(), loss.item()))
    #print("[visit] theta: {:.2f} => loss: {:.2f}".format(theta.item(), loss.item()))

    # update theta using GD
    theta = theta - (learning_rate * mean_gradient)
    search_history.append((theta, loss))

# quick helper to generate plots
# liste med x verdier med 0.01 mellomrom fra -4 til 6. 1000 punkter
loss_x = np.arange(-4, 6, 0.01)
#Beregner loss til alle de 1000 punktene, istednfor x_Train med y verdiene.
loss_y = np.array([calculate_l2_loss_non_vectorized(t, x_train, y_train) for t in loss_x])
#xprint("loss_y: ", loss_y)
#plotter en blå linje for generisk loss.
fig = px.line(x=loss_x, y=loss_y, title="GD History : Marks are iterations.")
#Tar alle theta veridene. fra search history.
x_visit, _ = list(zip(*search_history))
x_visit = np.concatenate(x_visit)
print(best_loss)
#Beregner loss til alle theta verdiene våre sammen med y verdiene-
y_visit = np.array([calculate_l2_loss_non_vectorized(t, x_train, y_train) for t in x_visit])
# legge til på samme plot som tidligere.
fig.add_trace(go.Scatter(x=x_visit, y=y_visit, name='GD history',
                         line=dict(color='firebrick', width=8, dash='dot')))

fig.show()