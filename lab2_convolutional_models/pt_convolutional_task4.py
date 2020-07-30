import os
import pickle
import torch
from torch import nn
from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Flatten, CrossEntropyLoss
import skimage as ski
import skimage.io
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / 'datasets' / 'cifar-10-batches-py'
SAVE_DIR = Path(__file__).parent / 'out_CIFAR10'

config = {'max_epochs': 4, 'batch_size': 50, 'save_dir': SAVE_DIR, 'weight_decay': 1e-3,
          'lr_policy': {1: {'lr': 1e-1},
                        3: {'lr': 1e-2},
                        5: {'lr': 1e-3},
                        7: {'lr': 1e-4}}}


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def plot_training_progress(data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(SAVE_DIR, 'training_plot_task4.png')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)


def draw_image(img, mean, std):
    img = img.transpose(1, 2, 0)
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()


def evaluate(name, x, y, net, loss, config, show_worst=False):
    print("\nRunning evaluation: ", name)
    batch_size = config['batch_size']
    num_examples = x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    cnt_correct = 0
    loss_avg = 0
    for i in range(num_batches):
        batch_x = x[i * batch_size:(i + 1) * batch_size, :]
        batch_y = y[i * batch_size:(i + 1) * batch_size]
        logits = net.forward(batch_x)
        if show_worst is True:
            pass
            # logits_for_correct = logits.detach().numpy()[batch_y]
            # remember original indices
            # logits_for_correct = np.vstack(logits_for_correct, np.array(range(logits_for_correct.shape[0])))
            # sort
            # logits_for_correct = logits_for_correct.sort()
            # for k in range(20):
            #    draw_image(x[logits_for_correct[k][1]], 1, 1)

            # find lowest probability for correct class
            # most_problematic = np.argmin(logits.detach().numpy()[batch_y])
            # draw_image(x[most_problematic], 1, 1)
        yp = np.argmax(logits.detach().numpy(), 1)
        cnt_correct += (yp == batch_y).sum()
        loss_val = loss.forward(logits, torch.Tensor(batch_y).type(torch.LongTensor))
        loss_avg += loss_val
        # print("step %d / %d, loss = %.2f" % (i*batch_size, num_examples, loss_val / batch_size))
    valid_acc = cnt_correct / num_examples * 100
    loss_avg /= num_batches
    print(name + " accuracy = %.2f" % valid_acc)
    print(name + " avg loss = %.2f\n" % loss_avg)
    return valid_acc, loss_avg


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


class CovolutionalModel(nn.Module):
    """
        conv(32,5) -> relu() -> pool(3,2) -> fc(256) -> relu() -> fc(128) -> relu() -> fc(10)

        conv(16,5) 16 maps, kernel 5x5
        pool(3,2) kernel 3x3, stride 2
    """

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.linear_layers = nn.Sequential(
            Linear(32 * 15 * 15, 200, bias=True),
            ReLU(inplace=True),
            Linear(200, 100, bias=True),
            ReLU(inplace=True),
            Linear(100, 10, bias=True),
        )

        self.optimizer = None
        self.loss = None
        self.grads = None
        self.out = None

    def forward(self, x):
        self.out = self.conv_layers(torch.Tensor(x))
        self.out = self.out.view(self.out.size(0), -1)
        self.out = self.linear_layers(self.out)
        return self.out

    def backward(self):
        self.grads = self.loss.backward()
        return self.grads.copy()


if __name__ == "__main__":
    img_height = 32
    img_width = 32
    num_channels = 3
    num_classes = 10

    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    data_mean = train_x.mean((0, 1, 2))
    data_std = train_x.std((0, 1, 2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    train_x = train_x.transpose(0, 3, 1, 2)
    valid_x = valid_x.transpose(0, 3, 1, 2)
    test_x = test_x.transpose(0, 3, 1, 2)
    # data prepared
    plot_data = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': [], 'lr': []}
    net = CovolutionalModel()
    loss = CrossEntropyLoss()

    draw_conv_filters(0, 0, net.conv_layers[0].weight.detach().numpy(), SAVE_DIR)

    batch_size = config['batch_size']
    save_dir = config['save_dir']

    num_examples = train_x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size

    for epoch in range(config['max_epochs']):
        cnt_correct = 0
        X, Yoh = shuffle_data(train_x, train_y)
        X = torch.FloatTensor(X)
        Yoh = torch.FloatTensor(Yoh)
        net.optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(net.optimizer, gamma=0.95)
        for i in range(num_batches):
            net.optimizer.zero_grad()
            batch_x = train_x[i * batch_size:(i + 1) * batch_size, :]
            batch_y = train_y[i * batch_size:(i + 1) * batch_size]

            logits = net.forward(batch_x)
            loss_val = loss(logits, torch.Tensor(batch_y).type(torch.LongTensor))
            loss_val.backward()

            net.optimizer.step()
            if i % 5 == 0:
                print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i * batch_size, num_examples, loss_val))

        plot_data['lr'] += [scheduler.get_lr()]
        scheduler.step()

        draw_conv_filters(epoch, i, net.conv_layers[0].weight.detach().numpy(), save_dir)
        acc, loss_avg = evaluate("train", train_x, train_y, net, loss, config)
        plot_data['train_loss'] += [loss_val]
        plot_data['train_acc'] += [acc]

        draw_conv_filters(epoch, i, net.conv_layers[0].weight.detach().numpy(), save_dir)
        acc, loss_avg = evaluate("test", test_x, test_y, net, loss, config, show_worst=True)
        plot_data['valid_loss'] += [loss_avg]
        plot_data['valid_acc'] += [acc]

    plot_training_progress(plot_data)
