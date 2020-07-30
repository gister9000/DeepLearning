from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class Instance:
    words: []  # list of strings
    label: str  # positive / negative


@dataclass
class NumericalizedInstance:
    words: torch.LongTensor  # encoded words
    label: torch.LongTensor  # 0 / 1


class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.frequencies = dict()
        self.data = list()
        self.x, self.y = list(), list()
        with open(filename, 'r') as f:
            line = f.readline().replace("\n", "")
            while line:
                words, label = line.split(",")
                label = label.replace(" ", "")
                x = words.split(" ")
                instance = Instance(x, label)
                self.data.append(instance)
                self.x.append(x)
                self.y.append(label)
                for word in x:
                    if word in self.frequencies.keys():
                        self.frequencies[word] += 1
                    else:
                        self.frequencies[word] = 1

                line = f.readline().replace("\n", "")
        self.size = len(self.data)
        self.vocab_x = None  # reserved, needs to be set manually after vocab is created based on the data
        self.vocab_y = None  # reserved, needs to be set manually after vocab is created based on the data
        # self.x = torch.tensor(self.x)
        # self.y = torch.tensor(self.y)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        raw = self.data[index]
        x = self.vocab_x.encode(raw.words)
        y = self.vocab_y.encode(raw.label)
        return NumericalizedInstance(x, y)

    def __iter__(self):
        return iter(self.data)


def print_each_n(flist, n):
    for i in range(len(flist)):
        if i % n == 0:
            print(flist[i])


class Vocab:
    def __init__(self, frequencies, max_size=-1, min_freq=5, label_vocab=False):
        self.freq = frequencies
        self.freq_list = list()
        for key in self.freq.keys():
            self.freq_list.append([key, self.freq[key]])
        self.freq_list.sort(key=lambda x: -x[1])
        # print_each_n(self.freq_list, 100)
        self.label_vocab = label_vocab
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.reverse_vocab = {0: "<PAD>", 1: "<UNK>"}
        index = 2

        if label_vocab:
            self.vocab = dict()
            self.reverse_vocab = dict()
            index = 0

        for word, count in self.freq_list:
            if index > max_size != -1:
                break
            if count > min_freq or self.label_vocab:
                self.vocab[word] = index
                self.reverse_vocab[index] = word
                index += 1

    def encode(self, words):
        ret = list()
        if self.label_vocab:
            return torch.tensor([self.vocab[words]])
        for word in words:
            try:
                ret.append(self.vocab[word])
            except KeyError:
                ret.append(self.vocab["<UNK>"])
        return torch.tensor(ret)

    def decode(self, nums):
        ret = list()
        if self.label_vocab:
            return str(self.reverse_vocab[nums])
        for num in nums:
            ret.append(self.reverse_vocab[num])
        return ret


# uses predefined set of values sst_glove
def get_embedding_matrix(v, d=300, path="datasets/sst_glove_6b_300d.txt"):
    words = v.vocab.keys()
    matrix = np.random.randn(len(words), d)
    if path is None or path == "":
        return torch.tensor(matrix)
    with open(path, 'r') as f:
        line = f.readline().replace("\n", "")
        while line:
            word = line.split(" ")[0]
            vector = line.split(" ")[1:]
            if word in words:
                index = v.vocab[word]
                matrix[index] = np.array(vector)
            line = f.readline().replace("\n", "")

    return torch.tensor(matrix)


def collate_fn(batch):
    """
    Arguments:
      batch:
        list of Instances returned by `Dataset.__getitem__`.
    Returns:
      A tensor representing the input batch.
    """

    texts = [item.words for item in batch]
    labels = [int(item.label) for item in batch]
    lengths = torch.tensor([len(text) for text in texts])  # Needed for later

    texts = torch.nn.utils.rnn.pad_sequence(texts, padding_value=0, batch_first=True)

    return texts, labels, lengths


def test_dataset_functionalities():
    dataset = NLPDataset('datasets/sst_train_raw.csv')
    # print("frequencies check:\nthe:", dataset.frequencies['the'], "\nto: ", dataset.frequencies['to'])

    text_vocab = Vocab(dataset.frequencies)
    label_vocab = Vocab({"positive": 1, "negative": 1}, label_vocab=True)
    dataset.vocab_x = text_vocab
    dataset.vocab_y = label_vocab
    instance_text, instance_label = dataset.x[3], dataset.y[3]
    print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")
    print(f"Numericalized text: {text_vocab.encode(instance_text)}")
    print(f"Numericalized label: {label_vocab.encode(instance_label)}")

    embedding_m = get_embedding_matrix(text_vocab)
    print("Loaded vector for 'the':\n", embedding_m[2])

    # embedding = torch.nn.Embedding.from_pretrained(embedding_m, padding_idx=0, freeze=True)

    batch_size = 2  # Only for demonstrative purposes, more otherwise
    shuffle = False  # Only for demonstrative purposes, True otherwise
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    texts, labels, lengths = next(iter(dataloader))
    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")

