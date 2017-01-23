# ============================================================== #
#                        Data-Preprocess                         #
#                                                                #
#                                                                #
# Process the given file formated as following question on a     #
# line followed by its answer on another line. The data is then  #
# filtered by removed too long or too short sequences, only      #
# keeping the most frequent vocab size words from the data.      #
# the data is saved as q ids & a ids & meta data holding words   #
# ============================================================== #

from __future__ import print_function

import numpy as np

import argparse
import nltk
import itertools
import pickle
import os.path

EN_WHITELIST   = '0123456789abcdefghijklmnopqrstuvwxyz '
EN_BLACKLIST   = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
UNKNOWN_SYMBOL = 'unk'

FLAGS          = None


def read_lines(path):
    """
    Read lines from file:
    ----------
    Args:
        path: path to file to read

    Returns:
        [list of lines]
    """

    return open(path).read().split('\n')[:-1]


def filter_line(line, whitelist = EN_WHITELIST):
    """
    Only keep chars in whitelist:
    ----------
    Args:
        line: line to filter
        whitelist: list of allowed chars

    Returns:
        str of filtered chars
    """

    return ''.join([ ch for ch in line if ch in whitelist ])


def index_(tokenized_sentences, vocab_size):
    """
    Read list of words, create index to word,
    word to index dictionaries:
    ----------
    Args:
        tokenized_sentences: tokenized list
        vocab_size: size of vocab

    Returns:
        tuple(vocab->(word, count), idx2w, w2idx)
    """

    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNKNOWN_SYMBOL] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )

    return index2word, word2index, freq_dist


def filter_unk(qtokenized, atokenized, w2idx):
    """
    Filter based on number of unknowns (words not in vocabulary)
    filter out the worst sentences:
    ----------
    Args:
        qtokenized: questions tokenized
        atokenized: answers tokenized
        w2idx: word to ids

    Returns:
        filtered quesions & answers
    """

    data_len = len(qtokenized)
    filtered_q, filtered_a = [], []

    for qline, aline in zip(qtokenized, atokenized):
        unk_count_q = len([ w for w in qline if w not in w2idx ])
        unk_count_a = len([ w for w in aline if w not in w2idx ])

        if unk_count_a == 0:
            if unk_count_q > 0:
                if unk_count_q / len(qline) > 0.2:
                    pass
            filtered_q.append(qline)
            filtered_a.append(aline)

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((data_len - filt_data_len)*100/data_len)
    print('[INFO    ]\t%s%% unknowns filtered from original data' % filtered)

    return filtered_q, filtered_a


def pad_seq(seq, lookup, maxlen):
    """
    Replace words with indices in a sequence
    replace with unknown if word not in lookup:
    ----------
    Args:
        seq: word
        lookup: word to id
        maxlen: max length

    Returns:
        [list of indices]
    """

    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNKNOWN_SYMBOL])

    return indices + [0]*(maxlen - len(seq))


def zero_pad(qtokenized, atokenized, w2idx):
    """
    create the final dataset : 
    - convert list of items to arrays of indices
    - add zero padding
    ----------
    Args:
        qtokenized: questions tokenized
        atokenized: answers tokenized
        w2idx: word to ids

    Returns:
        ([array_en([indices]), array_ta([indices]))
    """

    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, FLAGS.q_max_size], dtype = np.int32) 
    idx_a = np.zeros([data_len, FLAGS.a_max_size], dtype = np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, FLAGS.q_max_size)
        a_indices = pad_seq(atokenized[i], w2idx, FLAGS.a_max_size)

        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


def filter_data(sequences):
    """
    Filter too long and too short sequences:
    ----------
    Args:
        sequences: list of lines

    Returns:
        tuple(filtered_q, filtered_a)
    """

    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences)//2

    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i+1].split(' '))
        if qlen >= FLAGS.q_min_size and qlen <= FLAGS.q_max_size:
            if alen >= FLAGS.a_min_size and alen <= FLAGS.a_max_size:
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i+1])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print('[INFO    ]\t%s%% long & short scentences filtered from original data' % filtered)

    return filtered_q, filtered_a


def process_data():
    """
    Process input file and save filtered ids
    """

    print('[PROGRESS]\tRead lines from file')
    lines = read_lines(path = FLAGS.file_path)
    lines = [ line.lower() for line in lines ]

    # filter out unnecessary characters
    print('[PROGRESS]\tFilter sequences just keep chars in whitelist')
    lines = [ filter_line(line, EN_WHITELIST) for line in lines ]

    # filter out too long or too short sequences
    print('[PROGRESS]\tFilter too long or too short sequences')
    qlines, alines = filter_data(lines)

    # convert list of [lines of text] into list of [list of words ]
    print('[PROGRESS]\tSegment lines into words')
    qtokenized = [ [w.strip() for w in wordlist.split(' ') if w] for wordlist in qlines ]
    atokenized = [ [w.strip() for w in wordlist.split(' ') if w] for wordlist in alines ]

    # indexing -> idx2w, w2idx 
    print('[PROGRESS]\tIndex words')
    idx2w, w2idx, freq_dist = index_( qtokenized + atokenized, vocab_size = FLAGS.vocab_size)

    # filter out sentences with too many unknowns
    print('[PROGRESS]\tFilter Unknowns words according to vocab size')
    qtokenized, atokenized = filter_unk(qtokenized, atokenized, w2idx)

    print('[PROGRESS]\tZero Padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    print('[PROGRESS]\tSave numpy arrays to disk')
    np.save(os.path.join(FLAGS.output_dir, 'idx_q.npy'), idx_q)
    np.save(os.path.join(FLAGS.output_dir, 'idx_a.npy'), idx_a)

    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'q_max' : FLAGS.q_max_size,
            'q_min' : FLAGS.q_min_size,
            'a_max' : FLAGS.a_max_size,
            'a_min' : FLAGS.a_min_size,
            'freq_dist' : freq_dist
            }

    # write to disk : data control dictionaries
    with open(os.path.join(FLAGS.output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    # count of unknowns
    unk_count = (idx_q == 1).sum() + (idx_a == 1).sum()
    # count of words
    word_count = (idx_q > 1).sum() + (idx_a > 1).sum()

    print('[INFO    ]\tUnknown %: {0}'.format(100 * (unk_count/word_count)))
    print('[INFO    ]\tDataset count : ' + str(idx_q.shape[0]))


def main():
    """
    Process data
    """

    process_data()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Process and filter Q&A line by line to ids sequences')
    parser.add_argument('--file_path', help = 'Path to file', required = True)
    parser.add_argument('--output_dir', help = 'Output dir', required = True)
    parser.add_argument('--vocab_size', help = 'Vocab size', type = int, default = 8000)
    parser.add_argument('--q_max_size', help = 'Question max size', type = int, default = 25)
    parser.add_argument('--q_min_size', help = 'Question min size', type = int, default = 2)
    parser.add_argument('--a_max_size', help = 'Answer max size', type = int, default = 15)
    parser.add_argument('--a_min_size', help = 'Answer min size', type = int, default = 1)
    
    FLAGS, unparsed = parser.parse_known_args()

    main()
