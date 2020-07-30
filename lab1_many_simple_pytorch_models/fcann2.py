import numpy as np
import matplotlib.pyplot as plt
import data


def fcann2_train(X, Y, nr_hidden, param_niter=100001, param_delta=0.003, param_lambda=1e-3):
    C = max(Y) + 1  # nr_classes
    N, D = X.shape[0], X.shape[1]  # nr_samples, nr_dimensions
    W1 = np.random.randn(D, nr_hidden)  # C x nr_hidden
    b1 = np.zeros(nr_hidden)  # nr_hidden x 1
    W2 = np.random.randn(nr_hidden, C)  # nr_hidden x C
    b2 = np.zeros(C)  # C x 1
    range_N = range(N)  # used frequently

    for i in range(param_niter):
        # feed forward 1st layer
        s1 = np.dot(X, W1) + b1  # N x nr_hidden
        # ReLu
        h1 = np.maximum(0, s1)  # N x nr_hidden
        # feed forward 2nd layer
        s2 = np.dot(h1, W2) + b2  # N x C
        # stable softmax
        expscores = np.exp(s2 - np.amax(s2, axis=1, keepdims=True))  # N x C
        sumexp = np.sum(expscores, axis=1).reshape(N, 1)  # N x 1
        probs = expscores / sumexp  # N x C
        logprobs = np.log(probs)

        loss = -(1 / N) * np.sum(logprobs[range_N, Y[range_N]])  # scalar
        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # gradients matrix in regards to classification scores
        dL_dS2 = np.array(probs)  # N x C
        dL_dS2[range_N, Y[range_N]] -= 1  # N x C
        # gradients matrix in regards to s1 scores
        dL_dW2 = np.dot(np.transpose(h1), dL_dS2)  # C x H
        dL_db2 = np.sum(dL_dS2, axis=0)  # C x 1

        dL_ds1 = np.dot(dL_dS2, np.transpose(W2))
        dL_ds1[h1 <= 0] = 0  # N x H

        dL_dW1 = np.dot(np.transpose(X), dL_ds1)  # H x D
        dL_db1 = np.sum(dL_ds1, axis=0)  # H x 1

        # take step (with regularization)
        W1 -= param_delta * (dL_dW1 + param_lambda * W1)
        b1 -= param_delta * dL_db1
        W2 -= param_delta * (dL_dW2 + param_lambda * W2)
        b2 -= param_delta * dL_db2

    return W1, b1, W2, b2


def fcann2_classify(X, W1, b1, W2, b2):
    '''
    Arguments
      X:    data, np.array NxD
      w, b: parameters

    Return value
      np.array N x C containing probablitites for each sample/class
    '''

    # feed forward 1st layer
    s1 = np.dot(X, W1) + b1  # N x nr_hidden
    h1 = np.maximum(0, s1)  # N x nr_hidden
    # feed forward 2nd layer
    s2 = np.dot(h1, W2) + b2  # N x C
    # stable softmax
    expscores = np.exp(s2 - np.amax(s2, axis=1, keepdims=True))  # N x C
    sumexp = np.sum(expscores, axis=1).reshape(X.shape[0], 1)  # N x 1
    return expscores / sumexp  # N x C


def fcann2_decfun(W1, b1, W2, b2):
    '''
    Wrapper that feeds graph_surface function
    '''

    def classify(X):
        probs = fcann2_classify(X, W1, b1, W2, b2)
        return np.argmax(probs, axis=1)

    return classify


if __name__ == "__main__":
    np.random.seed(100)

    nr_components, nr_classes, nr_samples = 6, 2, 10
    X, Y_ = data.sample_gmm_2d(nr_components, nr_classes, nr_samples)

    hidden_layer_size = 5
    W1, b1, W2, b2 = fcann2_train(X, Y_, hidden_layer_size)

    probs = fcann2_classify(X, W1, b1, W2, b2)
    Y = np.argmax(probs, axis=1)

    accuracy, recall, matrix = data.eval_perf_multi(Y, Y_)
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)

    decfun = fcann2_decfun(W1, b1, W2, b2)
    rect = (np.min(X, axis=0), np.max(X, axis=0))  # get rect
    data.graph_surface(decfun, rect, offset=0.5)

    nr = nr_components * nr_samples
    data.graph_data(X, Y_.reshape(nr, ), Y.reshape(nr, ), special=[])
    plt.show()
