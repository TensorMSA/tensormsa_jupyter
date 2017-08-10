import os, pickle
from collections import Counter

def read_data(fname, count, word2idx):
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise("[!] Data %s not found" % fname)

    words = []
    for line in lines:
        words.extend(line.split())

    if len(count) == 0:
        count.append(['<eos>', 0])

    count[0][1] += len(lines)
    count.extend(Counter(words).most_common())

    if len(word2idx) == 0:
        word2idx['<eos>'] = 0
        word2idx['<unk>'] = 1
        word2idx['<pad>'] = 2

    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    data = list()
    for line in lines:
        for word in line.split():
            index = word2idx[word]
            data.append(index)
        data.append(word2idx['<eos>'])

    print("Read %s words from %s" % (len(data), fname))
    return data

def read_txt(input, word2idx):
    data = list()
    for word in input.split() :
        index = word2idx.get(word) if word2idx.get(word) else 1
        data.append(index)
    data.append(word2idx['<eos>'])
    return data

def save_obj(fname, obj) :
    with open(fname, 'wb') as f :
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fname, obj) :
    with open(fname, 'rb') as f :
        return pickle.load(f)