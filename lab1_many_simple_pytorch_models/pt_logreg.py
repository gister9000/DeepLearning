import torch
import numpy as np
import data
import matplotlib.pyplot as plt


class PTLogreg(torch.nn.Module):
    def __init__(self, D, C, param_lambda=0.05):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        self.D = D  # dimensions
        self.C = C  # nr_classes
        super(PTLogreg, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(D, C, dtype=torch.float), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros(C, dtype=torch.float), requires_grad=True)
        self.probs = None
        self.loss = None
        self.param_lambda = param_lambda

    def forward(self, X):
        scores = torch.mm(X, self.w) + self.b  # N x C
        self.probs = torch.nn.functional.softmax(scores)  # N x C

    def get_loss(self, X, Y_):
        N = X.shape[0]
        logprobs = torch.log(self.probs)  # N x C
        reg = torch.norm(self.w) * self.param_lambda  # regularization
        self.loss = - torch.mean(logprobs[range(N), Y_[range(N)]]) + reg


def train(model, X, Y_, param_niter=2001, param_delta=0.25):
    """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Y_: ground truth [Nx1], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
    """

    N, D = X.shape[0], X.shape[1]
    C = max(Y_) + 1  # nr_classes
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)

    for i in range(param_niter):
        model.forward(X)
        model.get_loss(X, Y_)
        model.loss.backward()

        if i % 50 == 0:
            print("iteration {}: loss {}".format(i, model.loss))

        optimizer.step()
        optimizer.zero_grad()

    return


def eval(model, X):
    """ Arguments:
     :param X: type: PTLogreg
     :param model: actual datapoints [NxD], type: np.array

    Returns: predicted class probabilites [NxC], type: np.array
    """
    torch_X = torch.Tensor(X)
    det_w = torch.Tensor.detach(model.w)
    det_b = torch.Tensor.detach(model.b)
    out = torch.mm(torch_X, det_w) + det_b  # N x C
    probs = torch.softmax(out, axis=1)  # N x C
    return torch.Tensor.numpy(probs)  # N x C


def logreg_decfun(model):
    '''
    Wrapper that feeds graph_surface function
    '''

    def classify(X):
        probs = eval(model, X)
        return np.argmax(probs, axis=1)

    return classify


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    nr_classes, nr_samples = 3, 100
    # nr_classes, nr_samples per class
    X, Y_ = data.sample_gauss_2d(nr_classes, nr_samples)
    N, D = X.shape[0], X.shape[1]
    C = max(Y_) + 1  # nr_classes

    # train the model
    model = PTLogreg(D, C)
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
    plt.show()
