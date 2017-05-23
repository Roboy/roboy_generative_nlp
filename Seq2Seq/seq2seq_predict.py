# ============================================================== #
#                       Seq2Seq-predict                          #
#                                                                #
#                                                                #
# Predict output sequence to an input text from cmd              #
# ============================================================== #

from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import glob
import sys
import wget
import tarfile

from data import data_utils
import seq2seq

FLAGS = None


def maybe_download_and_extract(ckpt_dir):
    """
    Downloads and extracts twitter checkpoints
    ----------
    Args:
        output_dir: directory where checkpoints are saved
    """

    if not tf.gfile.Exists(ckpt_dir):
        tf.gfile.MakeDirs(ckpt_dir)

    checkpoints = glob.glob(os.path.join(ckpt_dir, '*'))

    if checkpoints:
        return;

    urls = ['https://transfer.sh/9En0w/twitter-checkpoint.tar.gz']

    print('[INFO    ]\tNo train checkpoints found. Downloading them in %s' % ckpt_dir)
    if len(urls) > 0:
        for url in urls:
            filepath = os.path.join(ckpt_dir, 'tmp')
            wget.download(url, out = filepath)
            tar = tarfile.open(filepath)
            tar.extractall(path = ckpt_dir)
            tar.close()
            os.remove(filepath)

        print('\n[INFO    ]\tTwitter checkpoint downloaded successfully')


def predict():
    """
    Load processed dataset and trained model to predict responses:
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

    # load trained model from latest checkpoint
    files = glob.glob(os.path.join(FLAGS.ckpt_dir, '*'))

    if len(files) is not 0:
        sess = model.restore_last_session()
    else:
        print('[ERROR   ]\tNo checkpoints found to load trained model from')
        return

    # start cmd and print responses
    print('[INFO    ]\tCommand line interface started, write your message:')

    while True:
        print("> ", end = "")
        user_message = raw_input()

        if user_message == "exit":
            break

        # # encode input message
        encoded_message  = data_utils.encode(user_message, metadata['w2idx'], int(metadata['q_max']))
        encoded_message  = np.array(encoded_message).reshape((len(encoded_message), 1))
        # # decode output response
        response         = model.predict(sess, encoded_message)
        decoded_response = data_utils.decode(response[0], metadata['idx2w'], separator = ' ')
        print(u"< %s" % decoded_response)


def main():
    """
    Predict seq2seq
    """

    maybe_download_and_extract(FLAGS.ckpt_dir)
    predict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Predict seq2seq model given the processed dataset with cmd')
    parser.add_argument('--dataset_dir', help = 'Proccesed dataset dir', default = '../Datasets/twitter/processed/')
    parser.add_argument('--ckpt_dir', help = 'Checkpoints dir to load trained model from', default = '../Datasets/twitter/checkpoints/')
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 16)
    parser.add_argument('--lr', help = 'Learning rate', type = float, default = 0.0001)
    parser.add_argument('--num_layers', help = 'Seq2Seq layers number', type = int, default = 3)
    parser.add_argument('--emb_dim', help = 'Embded size', type = int, default = 1024)

    FLAGS, unparsed = parser.parse_known_args()

    main()
