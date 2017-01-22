# ============================================================== #
#                      Seq2Seq-evaluate                          #
#                                                                #
#                                                                #
# Evaluate seq2seq with given processed dataset                  #
# ============================================================== #

from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import glob

from data import data_utils
import seq2seq

FLAGS = None


def evaluate():
    """
    Load processed dataset and trained model for evalutation:
    """

    # load data from pickle and npy files
    metadata, idx_q, idx_a = data_utils.load_data(FLAGS.dataset_dir)
    (trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

    # parameters 
    xseq_len    = trainX.shape[-1]
    yseq_len    = trainY.shape[-1]
    xvocab_size = len(metadata['idx2w'])  
    yvocab_size = xvocab_size

    # build seq2seq model
    model = seq2seq.Seq2Seq(xseq_len = xseq_len,
                            yseq_len = yseq_len,
                            xvocab_size = xvocab_size,
                            yvocab_size = yvocab_size,
                            ckpt_path = FLAGS.ckpt_dir,
                            emb_dim = FLAGS.emb_dim,
                            num_layers = FLAGS.num_layers)

    # load trained model from latest checkpoint
    files = glob.glob(os.path.join(FLAGS.ckpt_dir, '*'))

    if len(files) is not 0:
        sess = model.restore_last_session()
    else:
        print('[ERROR   ]\tNo checkpoints found to load trained model from')
        return

    # generate test batches
    test_batch_gen  = data_utils.batch_gen(testX, testY, FLAGS.batch_size)

    # predict output to current batch
    input_ = test_batch_gen.next()[0]
    predicted = model.predict(sess, input_)

    # print predictions
    replies = []
    for ii, oi in zip(input_.T, predicted):
        q = data_utils.decode(sequence = ii, lookup = metadata['idx2w'], separator = ' ')
        decoded = data_utils.decode(sequence = oi, lookup = metadata['idx2w'], separator = ' ').split(' ')

        if decoded.count('unk') == 0:
            if decoded not in replies:
                print('q : [{0}];\na : [{1}]\n'.format(q, ' '.join(decoded)))
                replies.append(decoded)


def main():
    """
    Evaluate seq2seq
    """

    evaluate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Evaluate seq2seq model given the processed dataset')
    parser.add_argument('--dataset_dir', help = 'Proccesed dataset dir', required = True)
    parser.add_argument('--ckpt_dir', help = 'Checkpoints dir to load trained model from', required = True)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 32)
    parser.add_argument('--num_layers', help = 'Seq2Seq layers number', type = int, default = 3)
    parser.add_argument('--emb_dim', help = 'Embded size', type = int, default = 1024)

    FLAGS, unparsed = parser.parse_known_args()

    main()
