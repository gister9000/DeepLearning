import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


# learning this
def f(x):
    return 2 * x + 1


# dataset
N = 100000
l, r = int(-N/2), int(N/2)
inputs = np.array(range(l, r))

# scaling doesn't allow huge gradients for large dataset
inputs = inputs / N
labels = f(inputs)

X = torch.tensor(inputs)
Y = torch.tensor(labels)

train_ds = TensorDataset(X, Y)
batch_size = 4
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# params init
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# gradient descent
optimizer = optim.SGD([a, b], lr=0.5)

for i in range(11):
    batch_in, batch_out = next(iter(train_dl))

    # linear regression model
    out = a * batch_in + b
    diff = (out - batch_out)

    loss = torch.sum(diff * diff) / diff.numel()
    # pytorch autograd
    loss.backward()

    # calculating gradients explicitly
    anal_a = (1 / batch_size) * torch.sum(2 * batch_in * ((a * batch_in + b) - batch_out))
    anal_b = (1 / batch_size) * torch.sum(2 * ((a * batch_in + b) - batch_out))

    # take step
    optimizer.step()

    print('step: {}, loss:{}, Y_:{}, a:{}, b {}'.format(i, loss, out, a, b))
    print('Pytorch grad a: {}\tPytorch  grad b: {}'.format(a.grad, b.grad))
    print('Analitical grad a: {}\tAnalitical grad b: {}'.format(anal_a, anal_b))

    optimizer.zero_grad()
