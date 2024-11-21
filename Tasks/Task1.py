import random

def loss_l2(a, x):

    eq1 = a*x+2
    eq2 = -x+3*2

    y_hat = [eq1, eq2]
    y = [4, 2]

    loss_l2 = 0.0

    for y_hat_k, y_k in zip(y_hat, y):
        loss_l2 += (y_hat_k - y_k)**2

    return loss_l2


best_loss = float('inf')
best_model = None

for it in range(1_0000_0000):
    a = random.uniform(-10, 10)
    b = random.uniform(-10, 10)

    loss = loss_l2(a, b)

    if loss < best_loss:
        best_loss = loss
        best_model = (a, b)

        print(f'new  best iter: {it}, loss: {loss}, best loss: {best_loss}')

print(best_model)





