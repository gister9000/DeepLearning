import numpy as np
import torch
from torch.nn import Linear, ReLU, AdaptiveAvgPool1d, AdaptiveAvgPool2d
from torch.utils.data import DataLoader
import data


class BaselineModel(torch.nn.Module):
    """
        avg_pool() -> fc(300, 150) -> ReLU() -> fc(150, 150) -> ReLU() -> fc(150,1)
    """

    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding  # vector representations for words (prelearned)
        self.vector_len = embedding.embedding_dim  # 300
        self.grads = None
        self.logits = None

        self.avg_pool = AdaptiveAvgPool2d((1, self.vector_len))
        self.layers = torch.nn.Sequential(
            Linear(self.vector_len, 150, bias=True),
            ReLU(inplace=True),
            Linear(150, 150, bias=True),
            ReLU(inplace=True),
            Linear(150, 1, bias=True),
        )

    def forward(self, texts, lengths):
        self.logits = self.embedding(texts)  # b_size x T x vector_len (10 x variable x 300)
        self.logits = self.avg_pool(self.logits)  # removing variable dimensions
        self.logits = self.logits.float()  # b_size x 1 x vector_len (10 x 1 x 300)
        self.logits = self.layers(self.logits)  # deep NN
        return self.logits.squeeze()  # squeeze out the meaningless dimension

    def backward(self):
        self.grads = self.loss.backward()
        return self.grads.copy()


def classify(x):
    if torch.sigmoid(torch.tensor(x)) > 0.5:
        return 1
    else:
        return 0


def evaluate(dataset, model, criterion, b_size):
    with torch.no_grad():
        dataloader = DataLoader(dataset=dataset, batch_size=b_size, shuffle=True, collate_fn=data.collate_fn)
        num_examples = dataset.size
        num_batches = num_examples // b_size
        tp, fn, tn, fp, loss_avg = 0, 0, 0, 0, 0
        for j in range(num_batches):
            texts, labels, lengths = next(iter(dataloader))
            logits = model.forward(texts, lengths)
            to_binary = np.vectorize(classify)
            yp = to_binary(logits.numpy())
            Y, Y_ = yp, np.array(labels)
            tp += sum(np.logical_and(Y == Y_, Y_ == 1))
            fn += sum(np.logical_and(Y != Y_, Y_ == 1))
            tn += sum(np.logical_and(Y == Y_, Y_ == 0))
            fp += sum(np.logical_and(Y != Y_, Y_ == 0))

            loss = criterion(logits.reshape(b_size), torch.tensor(labels).float())
            loss_avg += loss

        loss_avg /= num_batches
        accuracy = (tp + tn) / (tp + fn + tn + fp)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_score = 2 * precision * recall / (precision + recall)
        print("Confusion matrix:  ")
        print(np.array([[tn, fp], [fn, tp]]))
        print("F1 score:\t\t", f1_score)
        print("Average loss:\t", float(loss_avg))
        print("Accuracy:\t\t", accuracy)
        return accuracy, loss_avg


def train(model, dataset, optimizer, criterion, b_size=32, clip=0.25, num_epochs=20, validate_set=None, test_set=None):
    dataloader = DataLoader(dataset=dataset, batch_size=b_size, shuffle=True, collate_fn=data.collate_fn)
    num_examples = dataset.size
    num_batches = num_examples // b_size

    for i in range(num_epochs):
        for j in range(num_batches):
            model.zero_grad()
            texts, labels, lengths = next(iter(dataloader))

            logits = model.forward(texts, lengths)
            loss = criterion(logits.squeeze(), torch.tensor(labels).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if j % 100 == 0:
                print("Epoch {}: loss: {}".format(i, loss))

        if validate_set:
            print("Evaluating model on validation dataset after epoch " + str(i))
            evaluate(validate_set, model, criterion, 32)
            print("#" * 60)

    if test_set:
        print("Evaluating model on test dataset after completion ")
        evaluate(test_set, model, criterion, 32)
        print("#" * 60)


if __name__ == "__main__":
    print("Testing dataset functionalities")
    data.test_dataset_functionalities()
    seed = 7052020
    lr = 1e-4
    b_size = 20

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = data.NLPDataset('datasets/sst_train_raw.csv')
    valid_dataset = data.NLPDataset('datasets/sst_valid_raw.csv')
    test_dataset = data.NLPDataset('datasets/sst_test_raw.csv')

    text_vocab = data.Vocab(train_dataset.frequencies)
    label_vocab = data.Vocab({"positive": 1, "negative": 1}, label_vocab=True)
    train_dataset.vocab_x = text_vocab
    train_dataset.vocab_y = label_vocab
    valid_dataset.vocab_x = text_vocab
    valid_dataset.vocab_y = label_vocab
    test_dataset.vocab_x = text_vocab
    test_dataset.vocab_y = label_vocab

    embedding_m = data.get_embedding_matrix(text_vocab)  # this is for not using pretrained vectors: path=None)
    embedding = torch.nn.Embedding.from_pretrained(embedding_m, padding_idx=0, freeze=True)

    model = BaselineModel(embedding)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("#" * 60)
    train(model, train_dataset, optimizer, criterion, b_size=b_size, validate_set=valid_dataset, test_set=test_dataset)