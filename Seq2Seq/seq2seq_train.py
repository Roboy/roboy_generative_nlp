# ============================================================== #
#                        Seq2Seq-train                           #
#                                                                #
#                                                                #
# Train seq2seq with given processed dataset                     #
# ============================================================== #

import tensorflow as tf
import numpy as np
import argparse
import glob
import os

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
                            num_layers = FLAGS.num_layers,
                            lr = FLAGS.lr)

    # generate batches (data is alread shuffled no need
    # to call rand_batch_gen as it takes more time)
    val_batch_gen   = data_utils.batch_gen(validX, validY, FLAGS.batch_size)
    train_batch_gen = data_utils.batch_gen(trainX, trainY, FLAGS.batch_size)

    # load latest checkpoint if any
    files = glob.glob(os.path.join(FLAGS.ckpt_dir, '*'))

    if len(files) is not 0:
        sess = model.restore_last_session()
    else:
        print('[WARNING ]\tNo checkpoints found. Starting from scratch')
        sess  = None

    # calculate number of steps and train
    steps = FLAGS.epochs * (len(trainX) / FLAGS.batch_size)
    model.train(train_batch_gen, val_batch_gen, steps, sess = sess)


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
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 16)
    parser.add_argument('--epochs', help = 'Number of epochs', type = int, default = 500)
    parser.add_argument('--lr', help = 'Learning rate', type = float, default = 0.0001)
    parser.add_argument('--num_layers', help = 'Seq2Seq layers number', type = int, default = 3)
    parser.add_argument('--emb_dim', help = 'Embded size', type = int, default = 1024)

    FLAGS, unparsed = parser.parse_known_args()

    main()
