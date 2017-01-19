# ============================================================== #
#                        Seq2Seq-train                           #
#                                                                #
#                                                                #
# Train seq2seq with given processed dataset                     #
# ============================================================== #

import tensorflow as tf
import numpy as np
import argparse

from data import data_utils
import seq2seq

FLAGS = None


def train():
    """
    Load processed dataset and train the model:
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

    # generate random batches
    val_batch_gen   = data_utils.rand_batch_gen(validX, validY, FLAGS.batch_size)
    train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, FLAGS.batch_size)

    # load latest checkpoint if any and train
    sess = model.restore_last_session()
    sess = model.train(train_batch_gen, val_batch_gen)


def main():
    """
    Train seq2seq
    """

    if not tf.gfile.Exists(FLAGS.ckpt_dir):
        tf.gfile.MakeDirs(FLAGS.ckpt_dir)

    train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train seq2seq model given the processed dataset')
    parser.add_argument('--dataset_dir', help = 'Proccesed dataset dir', required = True)
    parser.add_argument('--ckpt_dir', help = 'Checkpoints dir', required = True)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 32)
    parser.add_argument('--num_layers', help = 'Seq2Seq layers number', type = int, default = 3)
    parser.add_argument('--emb_dim', help = 'Embded size', type = int, default = 1024)

    FLAGS, unparsed = parser.parse_known_args()

    main()
