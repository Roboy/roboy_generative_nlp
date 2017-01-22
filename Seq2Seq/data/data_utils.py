# ============================================================== #
#                           Data-Utils                           #
#                                                                #
#                                                                #
# Data utils functions to load processed dataset, decode and     #
# generate random batches                                        #
# ============================================================== #

from random import sample

import numpy as np
import pickle
import os.path

import data_preprocess 


def load_data(path = ''):
    """
    Load processed data from path:
    ----------
    Args:
        path: processed dir

    Returns:
        tuple(metadata, idx_q. idx_a)
    """

    # read data control dictionaries
    with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    # read numpy arrays
    idx_q = np.load(os.path.join(path, 'idx_q.npy'))
    idx_a = np.load(os.path.join(path, 'idx_a.npy'))

    return metadata, idx_q, idx_a


def split_dataset(x, y, ratio = [0.7, 0.15, 0.15] ):
    """
    Split data into train (70%), test (15%) and valid(15%):
    ----------
    Args:
        x: input data
        y: output data
        ratio: split ratio

    Returns:
        tuple((trainX, trainY), (testX,testY), (validX,validY))
    """

    # number of examples
    data_len = len(x)
    lens     = [ int(data_len*item) for item in ratio ]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY   = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX, trainY), (testX, testY), (validX, validY)


def batch_gen(x, y, batch_size):
    """
    Generate batches from dataset:
    ----------
    Args:
        x: input data
        y: output data
        batch_size: size of batch

    Returns:
        yield (x_gen, y_gen)
    """

    # infinite while
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T


def rand_batch_gen(x, y, batch_size):
    """
    Generate batches, by random sampling a bunch of items:
    ----------
    Args:
        x: input data
        y: output data
        batch_size: size of batch

    Returns:
        yield (x_gen, y_gen)
    """

    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T


def encode(text, lookup, maxlen):
    """
    Geeneric encode function, text to ids:
    ----------
    Args:
        text: input sequence as string
        lookup: word to id mappings
        maxlen: max for padding

    Returns:
        array(ids)
    """

    filtered  = data_preprocess.filter_line(text)
    tokenized = [w.strip() for w in filtered.split(' ') if w]

    indices = []
    for word in tokenized:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[data_preprocess.UNKNOWN_SYMBOL])

    if (len(indices) < maxlen):
        indices = indices + [0]*(maxlen - len(tokenized))

    return indices


def decode(sequence, lookup, separator = ' '):
    """
    Geeneric decode function, ids to words
    (0 used for padding, is ignored):
    ----------
    Args:
        sequence: input sequence as ids
        lookup: id to word mappings
        separator: words separator

    Returns:
        array(words)
    """

    return separator.join([ lookup[element] for element in sequence if element ])

