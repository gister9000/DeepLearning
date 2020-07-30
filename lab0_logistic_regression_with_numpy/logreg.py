import numpy as np
import matplotlib.pyplot as plt
import data


def logreg_train(X, Y, param_niter=10001, param_delta=0.0005):
    '''
    Arguments
        X:  data, np.array NxD
        Y_: labels, np.array Nx1

    Return values
        w, b: parameters
    '''

    # Initialize w using Gaussian distribution N(0,1)
    # For random samples from N(\mu, \sigma^2), use:
    # sigma * np.random.randn(...) + mu
    C = max(Y) + 1  # nr_classes
    N, D = X.shape[0], X.shape[1]  # nr_samples, nr_dimensions
    w = np.random.randn(D, C)  # D x C
    b = np.zeros(C)  # C x 1
    range_N = range(N)  # used many times

    for i in range(param_niter):
        # exponential classification score
        scores = np.dot(X, w) + b  # N x C
        expscores = np.exp(scores - np.amax(scores, axis=1, keepdims=True))  # N x C
        sumexp = np.sum(expscores, axis=1).reshape(N, 1)  # N x 1

        # logarithm of class probabilities
        probs = expscores / sumexp  # N x C
        logprobs = np.log(probs)  # N x C

        # loss
        loss = -(1 / N) * np.sum(logprobs[range_N, Y[range_N]])  # scalar

        if i % 200 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # gradients matrix in regards to classification scores
        dL_dS = np.array(probs)  # N x C
        dL_dS[range_N, Y[range_N]] -= 1  # N x C

        # gradient matrix in regards to weights
        grad_W = np.dot(np.transpose(dL_dS), X)  # D x C
        # gradient vector in regards to b
        grad_b = np.sum(dL_dS, axis=0)  # 1 x C

        # take step
        w -= param_delta * np.transpose(grad_W)
        b -= param_delta * grad_b

    return w, b


def logreg_classify(X, w, b):
    '''
    Arguments
      X:    data, np.array NxD
      w, b: parameters

    Return value
      np.array N x C containing probablitites for each sample/class
    '''
    scores = np.dot(X, w) + b  # N x C
    expscores = np.exp(scores - np.amax(scores, axis=1, keepdims=True))  # N x C
    sumexp = np.sum(expscores, axis=1).reshape(X.shape[0], 1)  # N x 1
    return expscores / sumexp  # probabilities


def logreg_decfun(w, b):
    '''
    Wrapper that feeds graph_surface function
    '''

    def classify(X):
        probs = logreg_classify(X, w, b)
        return np.argmax(probs, axis=1)

    return classify


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    nr_classes, nr_samples = 3, 100
    # nr_classes, nr_samples per class
    X, Y_ = data.sample_gauss_2d(nr_classes, nr_samples)

    # train the model
    w, b = logreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = logreg_classify(X, w, b)
    Y = np.argmax(probs, axis=1)

    # report performance
    accuracy, recall, matrix = data.eval_perf_multi(Y, Y_)
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)

    # plot graph
    decfun = logreg_decfun(w, b)
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, rect, offset=0.5)

    nr = nr_classes * nr_samples
    data.graph_data(X, Y_.reshape(nr, ), Y.reshape(nr, ), special=[])
    # show the plot
    plt.show()
