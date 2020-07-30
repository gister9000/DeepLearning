import numpy as np
import matplotlib.pyplot as plt
import data


# squashes to [0, 1] interval
def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


# extended sigmoid - considers correct classification label
def extended_sigmoid(x, y):
    if y == 1:
        return sigmoid(x)
    else:  # y==0
        return 1 / (1 + np.exp(x))


def binlogreg_train(X, Y, param_niter=1001, param_delta=0.5):
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
    N, D = X.shape[0], X.shape[1]
    w = np.random.randn(D)  # Dx1
    b = 0

    for i in range(param_niter):
        scores = np.dot(X, w) + b  # Nx1
        probs = sigmoid(scores)  # Nx1
        loss = - (1 / N) * np.sum(np.log(probs))  # scalar

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        y_predicted = np.where(probs > 0.5, 1, 0)  # Nx1
        iverson = np.where(y_predicted == Y, 1, 0)  # Nx1
        dL_dscores = probs - iverson  # N x 1

        # gradients
        # grad_w = (1 / N) * np.sum(np.dot(dL_dscores, np.transpose(X)), axis=0)  # D x 1
        grad_w = (1 / N) * np.dot(np.transpose(dL_dscores), X)  # D x 1
        grad_b = (1 / N) * np.sum(dL_dscores, axis=0)  # 1 x 1

        # take step
        w -= param_delta * grad_w
        b -= param_delta * grad_b

    return w, b


def binlogreg_classify(X, w, b):
    '''
    Arguments
      X:    data, np.array NxD
      w, b: parameters
    
    Return value
      probs: class c1 probabilities
    '''
    return sigmoid(np.dot(X, w) + b)


def binlogreg_decfun(w, b):
    '''
    Wrapper that feeds graph_surface function
    '''

    def classify(X):
        return binlogreg_classify(X, w, b)

    return classify


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train tsigmoidhe model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = np.where(probs > 0.5, 1, 0)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nPrecision: ", precision, "\nAP: ", AP)

    # graph the decision surface
    decfun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
