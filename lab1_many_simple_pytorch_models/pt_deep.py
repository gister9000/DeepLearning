import torch
import numpy as np
import data
import matplotlib.pyplot as plt
import copy


class PTDeep(torch.nn.Module):
    def __init__(self, conf, activation_f, param_lambda=1e-4):
        """Arguments:
            :param conf: network architecture - nr_neurons in each layer
        """
        self.conf = conf
        self.nr_layers = len(conf)
        self.activation_f = activation_f  # activation function, e.g. softmax
        self.D = conf[0]  # nr_inputs
        self.C = conf[self.nr_layers - 1]  # nr_classes
        super(PTDeep, self).__init__()

        # initalize parameters
        w = [torch.nn.Parameter(0.01 * torch.randn(conf[i], conf[i + 1])) for i in range(self.nr_layers - 1)]
        b = [torch.nn.Parameter(torch.zeros(conf[i + 1])) for i in range(self.nr_layers - 1)]
        self.weights = torch.nn.ParameterList(w)
        self.biases = torch.nn.ParameterList(b)

        self.probs = None
        self.loss = None
        self.param_lambda = param_lambda
        self.count = None

    def forward(self, X):
        layer_out = torch.mm(X, self.weights[0]) + self.biases[0]  # N x C
        for i in range(1, self.nr_layers - 1):
            h = self.activation_f(layer_out)  # N x C
            layer_out = torch.mm(h, self.weights[i]) + self.biases[i]  # N x C

        self.probs = torch.nn.functional.softmax(layer_out)  # N x C

    def get_loss(self, X, Y_):
        N = X.shape[0]
        logprobs = torch.log(self.probs)  # N x C
        # regularization
        reg = torch.sum(torch.Tensor([torch.norm(x) for x in self.weights]))
        self.loss = - torch.mean(logprobs[range(N), Y_[range(N)]]) + reg * self.param_lambda

    def count_params(self):
        counter = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
                counter += sum(param.shape)
        self.count = counter
        print("Total count: ", counter)


def train(model, X, Y_, param_niter=20001, param_delta=0.1):
    """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Y_: ground truth [Nx1], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
    """
    N, D = X.shape[0], X.shape[1]
    C = max(Y_) + 1  # nr_classes
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)
    prev_loss, count = None, 0
    for i in range(param_niter):
        model.forward(X)
        model.get_loss(X, Y_)
        model.loss.backward()

        if i % 1 == 0:
            print("iteration {}: loss {}".format(i, model.loss))

        optimizer.step()
        optimizer.zero_grad()

        if prev_loss is not None:  # exit if no move was made for 100 iterations
            if abs(model.loss - prev_loss) < 1e-9:
                count += 1
            else:
                count = 0
            if count > 100:
                break

        prev_loss = model.loss

    return


def train_mb(model, X, Y_, param_niter=20001, param_delta=0.1, batch_size=500):
    """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Y_: ground truth [Nx1], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
    """
    N, D = X.shape[0], X.shape[1]
    C = max(Y_) + 1  # nr_classes
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)
    prev_loss, count = None, 0

    for i in range(param_niter):
        indices = list()  # choose
        while len(indices) < batch_size:
            x = np.random.randint(0, N-1)
            if x not in indices:
                indices.append(x)

        x_train = X[indices]
        y_train = Y_[indices]

        model.forward(x_train)
        model.get_loss(x_train, y_train)
        model.loss.backward()

        if i % 1 == 0:
            print("iteration {}: loss {}".format(i, model.loss))

        optimizer.step()
        optimizer.zero_grad()

        if prev_loss is not None:  # exit if no move was made for 100 iterations
            if abs(model.loss - prev_loss) < 1e-9:
                count += 1
            else:
                count = 0
            if count > 100:
                break

        prev_loss = model.loss

    return


# similar to train, but reports testset performanse after each iteration
def early_stopping_train(model, X, Y_, x_test, y_test, param_niter=20001, param_delta=0.1):
    """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Y_: ground truth [Nx1], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
    """
    best_model, best_accuracy = None, 0

    N, D = X.shape[0], X.shape[1]
    C = max(Y_) + 1  # nr_classes
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)
    prev_loss, count = None, 0
    for i in range(param_niter):
        model.forward(X)
        model.get_loss(X, Y_)
        model.loss.backward()

        if i % 1 == 0:
            print("iteration {}: loss {}".format(i, model.loss))

        optimizer.step()
        optimizer.zero_grad()

        if prev_loss is not None:  # exit if no move was made for 100 iterations
            if abs(model.loss - prev_loss) < 1e-9:
                count += 1
            else:
                count = 0
            if count > 100:
                break

        prev_loss = model.loss

        # evaluate the model on the test dataset
        probs = eval(model, x_test)
        Y = np.argmax(probs, axis=1)
        accuracy, recall, matrix = data.eval_perf_multi(Y, y_test)
        print("Current accuracy on testset: ", accuracy)
        if accuracy > best_accuracy:
            best_model = copy.copy(model)
            best_accuracy = accuracy

    return best_model


def train_adam(model, X, Y_, param_niter=20001, param_delta=0.1):
    """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Y_: ground truth [Nx1], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
    """
    N, D = X.shape[0], X.shape[1]
    C = max(Y_) + 1  # nr_classes
    optimizer = torch.optim.Adam(model.parameters(), lr=param_delta)
    prev_loss, count = None, 0
    for i in range(param_niter):
        model.forward(X)
        model.get_loss(X, Y_)
        model.loss.backward()

        if i % 1 == 0:
            print("iteration {}: loss {}".format(i, model.loss))

        optimizer.step()
        optimizer.zero_grad()

        if prev_loss is not None:  # exit if no move was made for 100 iterations
            if abs(model.loss - prev_loss) < 1e-9:
                count += 1
            else:
                count = 0
            if count > 100:
                break

        prev_loss = model.loss

    return


# adam learner with variable learning rate
def train_variable_adam(model, X, Y_, param_niter=20001, param_delta=0.1):
    """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Y_: ground truth [Nx1], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
    """
    N, D = X.shape[0], X.shape[1]
    C = max(Y_) + 1  # nr_classes
    optimizer = torch.optim.Adam(model.parameters(), lr=param_delta)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1-1e-4)
    prev_loss, count = None, 0
    for i in range(param_niter):
        model.forward(X)
        model.get_loss(X, Y_)
        model.loss.backward()

        if i % 1 == 0:
            print("iteration {}: loss {}".format(i, model.loss))

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if prev_loss is not None:  # exit if no move was made for 100 iterations
            if abs(model.loss - prev_loss) < 1e-9:
                count += 1
            else:
                count = 0
            if count > 100:
                break

        prev_loss = model.loss

    return



def eval(model, X):
    """ Arguments:
     :param X: type: PTLogreg
     :param model: actual datapoints [NxD], type: np.array

    Returns: predicted class probabilites [NxC], type: np.array
    """
    torch_X = torch.Tensor(X)
    det_w, det_b = list(), list()
    [det_w.append(torch.Tensor.detach(x)) for x in model.weights]
    [det_b.append(torch.Tensor.detach(x)) for x in model.biases]

    layer_out = torch.mm(torch_X, det_w[0]) + det_b[0]  # N x C
    for i in range(1, model.nr_layers - 1):
        h = model.activation_f(layer_out)  # N x C
        layer_out = torch.mm(h, det_w[i]) + det_b[i]  # N x C

    model.probs = torch.nn.functional.softmax(layer_out)  # N x C

    return torch.Tensor.numpy(model.probs)  # N x C


def logreg_decfun(model):
    '''
    Wrapper that feeds graph_surface function
    '''

    def classify(X):
        probs = eval(model, X)
        return np.argmax(probs, axis=1)

    return classify


# recreating logreg from pt_logreg module by using [2, 3] architecture
def task1():
    # get the training dataset
    nr_classes, nr_samples = 3, 100
    # nr_classes, nr_samples per class
    X, Y_ = data.sample_gauss_2d(nr_classes, nr_samples)
    N, D = X.shape[0], X.shape[1]
    C = max(Y_) + 1  # nr_classes

    # train the model
    model = PTDeep([D, C], torch.relu)
    train(model, torch.Tensor(X), Y_)

    # evaluate the model on the training dataset
    probs = eval(model, X)
    Y = np.argmax(probs, axis=1)

    # report performance
    accuracy, recall, matrix = data.eval_perf_multi(Y, Y_)
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)

    # plot graph
    decfun = logreg_decfun(model)
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, rect, offset=0.5)

    nr = nr_classes * nr_samples
    data.graph_data(X, Y_.reshape(nr, ), Y.reshape(nr, ), special=[])
    # show the plot
    model.count_params()
    plt.show()


# testing on more difficult datasets and various architectures
def experiment(architecture, data_conf, function):
    # get the training dataset
    nr_components, nr_classes, nr_samples = data_conf
    # nr_classes, nr_samples per class
    X, Y_ = data.sample_gmm_2d(nr_components, nr_classes, nr_samples)
    C = max(Y_) + 1  # nr_classes

    # train the model
    model = PTDeep(architecture, function)
    train(model, torch.Tensor(X), Y_)

    # evaluate the model on the training dataset
    probs = eval(model, X)
    Y = np.argmax(probs, axis=1)

    # report performance
    accuracy, recall, matrix = data.eval_perf_multi(Y, Y_)
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)

    # plot graph
    decfun = logreg_decfun(model)
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, rect, offset=0.5)

    nr = nr_components * nr_samples
    data.graph_data(X, Y_.reshape(nr, ), Y.reshape(nr, ), special=[])
    model.count_params()
    print("Experiment done!\nArchitecture: {}\nData_conf: {}".format(architecture, data_conf))


def do_experiments_relu():
    confs = [[2, 2], [2, 10, 2], [2, 10, 10, 2]]
    data_confs = [[4, 2, 40], [6, 2, 10]]
    for conf in confs:
        for data_conf in data_confs:
            experiment(conf, data_conf, torch.relu)
            print("Function: ReLu")
            plt.show()


def do_experiments_sigmoid():
    confs = [[2, 2], [2, 10, 2], [2, 10, 10, 2]]
    data_confs = [[4, 2, 40], [6, 2, 10]]
    for conf in confs:
        for data_conf in data_confs:
            experiment(conf, data_conf, torch.sigmoid)
            print("Function: sigmoid")
            plt.show()


if __name__ == "__main__":
    np.random.seed(100)
    # task1()  # basic test
    do_experiments_sigmoid()
    do_experiments_relu()
