# ============================================================== #
#                      Seq2Seq-evaluate                          #
#                                                                #
#                                                                #
# Evaluate seq2seq with given processed dataset                  #
# ============================================================== #

from __future__ import print_function

import os
import sys
import tensorflow as tf

from config import params_setup
from lib import data_utils
from lib.seq2seq_model_utils import create_model, get_predicted_sentence

FLAGS = None


def evaluate(args):
    """
    Load processed dataset and trained model for evalutation:
    """

    with tf.Session() as sess:
        # Create model and load parameters.
        args.batch_size = 1  # We decode one sentence at a time.
        model = create_model(sess, args)

        # Load vocabularies.
        vocab_path = os.path.join(
            args.data_dir, "vocab%d.in" % args.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            predicted_sentence = get_predicted_sentence(
                args, sentence, vocab, rev_vocab, model, sess)
            if isinstance(predicted_sentence, list):
                for sent in predicted_sentence:
                    print("  (%s) -> %s" % (sent['prob'], sent['dec_inp']))
            else:
                print(sentence, ' -> ', predicted_sentence)

            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def main():
    """
    Evaluate seq2seq
    """

    evaluate(FLAGS)


if __name__ == '__main__':

    FLAGS = params_setup()
    main()

