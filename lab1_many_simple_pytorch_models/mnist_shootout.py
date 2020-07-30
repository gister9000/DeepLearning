import torch
import torchvision
import pt_deep
import numpy as np
import data
import matplotlib.pyplot as plt

# MNIST dataset - hand written digits from 0 to 9
# 60k  28x28 pixel images for learning and 10k for testing
dataset_root = '/tmp/mnist'  # change this to your preference
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

x_train, y_train = mnist_train.train_data, mnist_train.train_labels
x_test, y_test = mnist_test.test_data, mnist_test.test_labels
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2]
C = y_train.max().add_(1).item()

x_test_backup = torch.Tensor(x_test)
x_train = x_train.reshape(N, D)
x_test = x_test.reshape(x_test.shape[0], D)


def experiment(architecture, function, param_lambda, niter=500, delta=0.1):
    N = architecture[0]
    # train the model
    model = pt_deep.PTDeep(architecture, function, param_lambda=param_lambda)
    pt_deep.train(model, torch.Tensor(x_train), y_train, param_niter=niter, param_delta=delta)

    # evaluate the model on the test dataset
    probs = pt_deep.eval(model, x_test)
    Y = np.argmax(probs, axis=1)

    # report performance on test set
    accuracy, recall, matrix = data.eval_perf_multi(Y, y_test)
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)
    print("Architecture: {}\nExperiment done!\n".format(architecture))
    return accuracy, recall, matrix, np.max(probs[range(N)][Y[range(N)]], axis=1)


def bullet1():
    architecture = [784, 10]
    # param_lambdas = [0.1 ** i for i in range(2, 5)]
    param_lambdas = [0.1, 1e-4, 1e-8]
    results = list()
    for param_lambda in param_lambdas:
        # accuracy, recall, matrix = experiment(architecture, torch.sigmoid, param_lambda, niter=500)
        accuracy, recall, matrix, probs = experiment(architecture, torch.relu, param_lambda, niter=500)
        results.append([accuracy, recall, matrix])

    print("\n" * 2, "#" * 70)
    for i, param_lambda in enumerate(param_lambdas):
        accuracy, recall, matrix = results[i]
        print("Regularization factor: {}\t accuracy: {}".format(param_lambda, accuracy))
        # print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)


def bullet2():
    global x_train, y_train
    architectures = [[784, 10],
                     [784, 100, 10],
                     [784, 100, 100, 10],
                     [784, 100, 100, 100, 10]]
    results = list()

    for architecture in architectures:
        # deeper models prefer lower learning rate and more iterations
        niter = 40 * len(architecture) ** 2
        delta = 0.84 ** (len(architecture))
        if len(architecture) == 5:
            delta = 0.15
            niter = 2000
        if len(architecture) == 2:
            niter = 300
            delta = 0.85

        # accuracy, recall, matrix = experiment(architecture, torch.sigmoid, param_lambda=1e-4)
        accuracy, recall, matrix, probs = experiment(architecture, torch.relu, param_lambda=1e-4, niter=niter,
                                                     delta=delta)
        results.append([accuracy, recall, matrix, niter, delta, probs])

    print("\n" * 2, "FULL RESULTS")
    max_accuracy = 0
    for i, architecture in enumerate(architectures):
        accuracy, recall, matrix, niter, delta, probs = results[i]
        if accuracy > max_accuracy:
            index_most_accurate = i
            max_accuracy = float(accuracy)
        print("#" * 100)
        print("Architecture: {}\tniter: {}\tdelta: {}".format(architecture, niter, delta))
        print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)
    print("#" * 100)
    print("The most successful model architecture is: ", architectures[index_most_accurate])
    # print("This is strongly affected by my formulas for niter/delta params.")

    # show image which dominate the loss function most accurate model
    # those images have lowest probability for correct class
    most_problematic = np.argmin(results[index_most_accurate][5])
    worst_prob = results[index_most_accurate][5][most_problematic]
    print("Showing most problematic image, probability for correct class: ", worst_prob)
    fig = plt.figure()
    plt.imshow(x_test_backup[most_problematic], cmap=plt.get_cmap('gray'))
    plt.savefig('problematic.png')
    plt.show()


def bullet3():
    global x_train, y_train
    # choose random idices
    indices, not_i = list(), list()
    while len(indices) < int(N / 5):
        x = np.random.randint(0, N - 1)
        if x not in indices:
            indices.append(x)

    for i in range(N):
        if i not in indices:
            not_i.append(i)

    x_train_bullet3 = x_train[indices]
    y_train_bullet3 = y_train[indices]
    x_test_bullet3 = x_train[not_i]
    y_test_bullet3 = y_train[not_i]

    model = pt_deep.PTDeep([784, 100, 10], torch.relu, param_lambda=1e-4)
    model = pt_deep.early_stopping_train(model, x_train_bullet3, y_train_bullet3, x_test_bullet3, y_test_bullet3,
                                         param_niter=300, param_delta=0.4)

    # evaluate the model on the test dataset
    probs = pt_deep.eval(model, x_test_bullet3)
    Y = np.argmax(probs, axis=1)

    # report performance on test set
    accuracy, recall, matrix = data.eval_perf_multi(Y, y_test_bullet3)
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)


def bullet4():
    model = pt_deep.PTDeep([784, 100, 10], torch.relu)
    pt_deep.train_mb(model, x_train, y_train, param_niter=5000, param_delta=0.02, batch_size=500)

    # evaluate the model on the test dataset
    probs = pt_deep.eval(model, x_test)
    Y = np.argmax(probs, axis=1)

    # report performance on test set
    accuracy, recall, matrix = data.eval_perf_multi(Y, y_test)
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)


def bullet5():
    model = pt_deep.PTDeep([784, 100, 10], torch.relu)
    pt_deep.train_adam(model, x_train, y_train, param_niter=1000, param_delta=1e-4)

    # evaluate the model on the test dataset
    probs = pt_deep.eval(model, x_test)
    Y = np.argmax(probs, axis=1)

    # report performance on test set
    accuracy, recall, matrix = data.eval_perf_multi(Y, y_test)
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)


def bullet6():
    model = pt_deep.PTDeep([784, 100, 10], torch.relu)
    pt_deep.train_variable_adam(model, x_train, y_train, param_niter=1000, param_delta=1e-4)

    # evaluate the model on the test dataset
    probs = pt_deep.eval(model, x_test)
    Y = np.argmax(probs, axis=1)

    # report performance on test set
    accuracy, recall, matrix = data.eval_perf_multi(Y, y_test)
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)


def bullet7():
    model = pt_deep.PTDeep([784, 100, 10], torch.relu)

    # evaluate the model on the test dataset
    probs = pt_deep.eval(model, x_test)
    Y = np.argmax(probs, axis=1)

    # report performance on test set
    accuracy, recall, matrix = data.eval_perf_multi(Y, y_test)
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)


def bullet8():
    from sklearn.svm import SVC
    param_svm_c = 1
    param_svm_gamma = 'auto'
    model = SVC(kernel="linear", C=param_svm_c, gamma=param_svm_gamma, probability=True)
    model.fit(x_train, y_train)
    # report performance on test set
    accuracy, recall, matrix = data.eval_perf_multi(model.predict(x_test), y_test)
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)

    model = SVC(kernel="rbf", C=param_svm_c, gamma=param_svm_gamma, probability=True)
    model.fit(x_train, y_train)
    # report performance on test set
    accuracy, recall, matrix = data.eval_perf_multi(model.predict(x_test), y_test)
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)


if __name__ == "__main__":
    np.random.seed(100)

    bullet1()  # test  [784, 10]  with variable regularization factor
    #input("Press ENTER to continue with next test")

    bullet2()  # test 5 different architectures, each with 1 extra layer than previous
    #input("Press ENTER to continue with next test")

    # Regularization doesn't help to increate accuracy on train set, but greatly helps on testset

    bullet3()  # test early stoppage learn
    #input("Press ENTER to continue with next test")

    bullet4()  # stochastic gradient descent
    #input("Press ENTER to continue with next test")

    bullet5()  # test adam learner
    #input("Press ENTER to continue with next test")

    bullet6()  # test adam learner with variable learning rate
    #input("Press ENTER to continue with next test")

    bullet7()  # test untrained model
    #input("Press ENTER to continue with next test")

    bullet8()  # test linear and rbf SVM
