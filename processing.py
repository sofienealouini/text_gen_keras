import numpy as np


def get_alphabet(text):
    return sorted(list(set(text)))


def create_dicts(alphabet):
    i = 0
    index_char = {}
    char_index = {}
    for c in alphabet:
        char_index[c] = i
        index_char[i] = c
        i+=1
    return char_index, index_char


def encode(text, char_index, nb_chars, data_type):
    dataset = []
    for c in text:
        code = np.zeros(nb_chars, dtype=data_type)
        code[char_index[c]] = 1
        dataset.append(code)
    return np.array(dataset)


def decode(dataset, index_char):
    txt = ""
    for char in dataset:
        i = np.argmax(char)
        c = index_char[i]
        txt = txt + c
    return txt


def create_data(dataset, seq_length, skip):
    inputs, targets = [], []
    for i in range(0,len(dataset)-seq_length,skip):
        x = list(dataset[i:(i+seq_length)])
        y = dataset[i+seq_length]
        inputs.append(x)
        targets.append(y)
    return np.array(inputs), np.array(targets)


def to_array(texte, line_length):
    res = []
    for i in range(0,len(texte),line_length):
        x = list(texte[i:(i+line_length)])
        res.append(x)
    res = np.array(res)
    return res