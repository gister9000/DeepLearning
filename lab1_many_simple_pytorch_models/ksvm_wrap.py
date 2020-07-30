import numpy as np
import data
import matplotlib.pyplot as plt
from sklearn.svm import SVC


class ksvm_wrap:
    '''
    Metode:
      __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        construct SVM RBF wrapper and learn the model
        X,Y_:            input, labels
        param_svm_c:     relative data cost
        param_svm_gamma: RBF kernel size
    '''

    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.model = SVC(kernel="rbf", C=param_svm_c, gamma=param_svm_gamma, probability=True)
        self.model.fit(X, Y_)

    # returns predicted classes for X
    def predict(self, X):
        return self.model.predict(X)

    # returns classification scores of X
    def get_scores(self, X):
        return self.model.predict_proba(X)

    # returns data points that are support vectors in form of index in X
    # np.argwhere is deprecated for this usecase so this is done manually
    def get_support(self, X):
        indices = list()
        for i, value in np.ndenumerate(X):
            if value in self.model.support_vectors_:
                indices.append(i)
        return indices


def decfun(model):
    '''
    Wrapper that feeds graph_surface function
    '''

    def classify(X):
        return model.predict(X)

    return classify


if __name__ == "__main__":
    np.random.seed(100)
    # get the training dataset
    nr_components, nr_classes, nr_samples = 6, 2, 10
    # nr_classes, nr_samples per class
    X, Y_ = data.sample_gmm_2d(nr_components, nr_classes, nr_samples)
    N, D = X.shape[0], X.shape[1]
    C = max(Y_) + 1  # nr_classes

    # train the model
    SVM = ksvm_wrap(X, Y_)

    # evaluate the model on the training dataset
    probs = SVM.get_scores(X)
    Y = SVM.predict(X)

    # report performance
    accuracy, recall, matrix = data.eval_perf_multi(Y, Y_)
    print("Accuracy: ", accuracy, "\nRecall: ", recall, "\nMatrix:\n", matrix)

    # plot graph
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(SVM.predict, rect, offset=0.5)

    nr = nr_components * nr_samples
    data.graph_data(X, Y_.reshape(nr, ), Y.reshape(nr, ), special=SVM.get_support(X))
    # show the plot
    plt.show()
