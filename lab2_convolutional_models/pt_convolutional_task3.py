import os
from pathlib import Path
import skimage as ski
import skimage.io
import data
import numpy as np
import torch
from torch import nn
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, CrossEntropyLoss
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out_MNIST'

config = {'max_epochs': 3, 'batch_size': 50, 'save_dir': SAVE_DIR, 'weight_decay': 1e-3,
          'lr_policy': {1: {'lr': 1e-1},
                        3: {'lr': 1e-2},
                        5: {'lr': 1e-3},
                        7: {'lr': 1e-4}}}


def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[0]
    num_channels = w.shape[1]
    k = w.shape[2]
    assert w.shape[3] == w.shape[2]
    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = np.math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r + k, c:c + k, :] = w[:, :, :, i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def plot_training_progress(data):
    num_points = len(data)
    x_data = np.linspace(1, num_points, num_points)
    plt.plot(x_data, data)
    save_path = os.path.join('training_plot_task3.png')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)


class CovolutionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            Conv2d(1, 30, kernel_size=5, stride=1, padding=2, bias=True),
            MaxPool2d(kernel_size=2, stride=2),
            ReLU(inplace=True),
            Conv2d(30, 32, kernel_size=5, stride=1, padding=2, bias=True),
            MaxPool2d(kernel_size=2, stride=2),
            ReLU(inplace=True),
        )

        self.linear_layers = nn.Sequential(
            Linear(32 * 7 * 7, 512, bias=True),
            ReLU(inplace=True),
            Linear(512, 10, bias=True),
        )

        self.optimizer = None
        self.loss = None
        self.grads = None
        self.out = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.linear_layers[2]:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                self.linear_layers[2].reset_parameters()

    def forward(self, x):
        self.out = self.conv_layers(torch.Tensor(x))
        self.out = self.out.view(self.out.size(0), -1)
        self.out = self.linear_layers(self.out)
        return self.out

    def backward(self):
        self.grads = self.loss.backward()
        return self.grads.copy()


if __name__ == "__main__":
    np.random.seed(100)
    ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False)
    train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float) / 255
    train_y = ds_train.targets.numpy()
    train_x, valid_x = train_x[:55000], train_x[55000:]
    train_y, valid_y = train_y[:55000], train_y[55000:]
    test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float) / 255
    test_y = ds_test.targets.numpy()
    train_mean = train_x.mean()
    train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
    train_y, valid_y, test_y = (data.dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))
    # data prepared

    net = CovolutionalModel()
    loss = CrossEntropyLoss()

    train_losses, validation_losses = list(), list()
    lr_policy = config['lr_policy']
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    save_dir = config['save_dir']

    num_examples = train_x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    solver_config = lr_policy[1]
    for epoch in range(1, max_epochs + 1):
        if epoch in lr_policy:
            solver_config = lr_policy[epoch]
        cnt_correct = 0

        permutation_idx = np.random.permutation(num_examples)
        train_x = train_x[permutation_idx]
        train_y = train_y[permutation_idx]
        net.optimizer = torch.optim.SGD(net.parameters(), lr=solver_config['lr'], weight_decay=config['weight_decay'])

        for i in range(num_batches):
            net.optimizer.zero_grad()

            batch_x = train_x[i * batch_size:(i + 1) * batch_size, :]
            batch_y = train_y[i * batch_size:(i + 1) * batch_size]
            logits = net.forward(batch_x)

            labels = torch.Tensor(np.argmax(batch_y, 1))
            loss_val = loss(logits, labels.type(torch.LongTensor))
            train_losses.append(loss_val)
            loss_val.backward()

            net.optimizer.step()

            if i % 5 == 0:
                print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i * batch_size, num_examples, loss_val))

        # Scores after each epoch
        yp = np.argmax(logits.detach().numpy(), 1)
        yt = np.argmax(batch_y, 1)
        cnt_correct += (yp == yt).sum()
        print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
        draw_conv_filters(epoch, i, net.conv_layers[0].weight.cpu().detach().numpy(), save_dir)

        # validation test after each epoch too
        # labels = torch.Tensor(np.argmax(test_y, 1))
        # validation_loss = loss(net.forward(test_x), labels.type(torch.LongTensor))
        # validation_losses.append(validation_loss)

    print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
    plot_training_progress(train_losses)

    # compute validation loss
    labels = torch.Tensor(np.argmax(test_y, 1))
    validation_loss = loss(net.forward(test_x), labels.type(torch.LongTensor))
    # compute classification accuracy
    yp = np.argmax(logits.detach().numpy(), 1)
    yt = np.argmax(batch_y, 1)
    cnt_correct += (yp == yt).sum()
    print("Test loss: {}\nTest accuracy: {}\n".format(validation_loss, (cnt_correct / num_examples * 100)))
