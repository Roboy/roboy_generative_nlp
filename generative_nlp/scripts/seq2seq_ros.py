#!/usr/bin/env python

# ============================================================== #
#                       Seq2Seq-predict                          #
#                                                                #
#                                                                #
# Predict output sequence to an input text from cmd              #
# ============================================================== #

from __future__ import print_function

# ROS libraries
from generative_nlp.srv import *
import rospy

import tensorflow as tf
import numpy as np
import os
import glob
import sys

from data import data_utils
import seq2seq

LOAD_MODEL = True

class GNLP():

    metadata = None
    idx_q = None
    idx_a = None
    trainX = None
    trainY = None
    testX = None
    testY = None
    validX = None
    validY = None
    xseq_len = None
    yseq_len  = None
    xvocab_size  = None
    yvocab_size = None
    model = None
    sess = None

    def load(self):
        """
        Load processed dataset and trained model for evalutation:
        """

        # load data from pickle and npy files
        self.metadata, self.idx_q, self.idx_a = data_utils.load_data(dataset_dir)
        (self.trainX, self.trainY), (self.testX, self.testY), (self.validX, self.validY) = data_utils.split_dataset(self.idx_q, self.idx_a)

        # parameters 
        self.xseq_len    = self.trainX.shape[-1]
        self.yseq_len    = self.trainY.shape[-1]
        self.xvocab_size = len(self.metadata['idx2w'])  
        self.yvocab_size = self.xvocab_size

        # build seq2seq model
        self.model = seq2seq.Seq2Seq(xseq_len = self.xseq_len,
                                yseq_len = self.yseq_len,
                                xvocab_size = self.xvocab_size,
                                yvocab_size = self.yvocab_size,
                                ckpt_path = ckpt_dir,
                                emb_dim = emb_dim,
                                num_layers = num_layers,
                                lr = lr)

        # load trained model from latest checkpoint
        files = glob.glob(os.path.join(ckpt_dir, '*'))

        if len(files) is not 0:
            self.sess = self.model.restore_last_session()
            return True, self.sess
        else:
            rospy.logerr('[ERROR   ]\tNo checkpoints found to load trained model from')
            return False, None

    def train(self,req):
        # load data from pickle and npy files
        self.metadata, self.idx_q, self.idx_a = data_utils.load_data(dataset_dir)
        (self.trainX, self.trainY), (self.testX, self.testY), (self.validX, self.validY) = data_utils.split_dataset(self.idx_q, self.idx_a)

        # parameters 
        self.xseq_len    = self.trainX.shape[-1]
        self.yseq_len    = self.trainY.shape[-1]
        self.xvocab_size = len(self.metadata['idx2w'])  
        self.yvocab_size = self.xvocab_size

        # build seq2seq model
        self.model = seq2seq.Seq2Seq(xseq_len = self.xseq_len,
                                yseq_len = self.yseq_len,
                                xvocab_size = self.xvocab_size,
                                yvocab_size = self.yvocab_size,
                                ckpt_path = ckpt_dir,
                                emb_dim = emb_dim,
                                num_layers = num_layers,
                                lr = lr)

        # generate batches (data is alread shuffled no need
        # to call rand_batch_gen as it takes more time)
        val_batch_gen   = data_utils.batch_gen(self.validX, self.validY, batch_size)
        train_batch_gen = data_utils.batch_gen(self.trainX, self.trainY, batch_size)

        # load latest checkpoint if any
        files = glob.glob(os.path.join(ckpt_dir, '*'))

        if len(files) is not 0:
            self.sess = self.model.restore_last_session()
        else:
            rospy.loginfo('[WARNING ]\tNo checkpoints found. Starting from scratch')
            self.sess  = None

        # calculate number of steps and train
        steps = epochs * (len(self.trainX) / batch_size) 
        self.model.train(train_batch_gen, val_batch_gen, steps, sess = self.sess)
        rospy.loginfo('[INFO ]\tTraining finished with a success')
        return True

    def train_server(self):
        s = rospy.Service('roboy/gnlp_train', seq2seq_train, self.train)
        rospy.loginfo('Training service ready to be called.')

    def evaluate(self,req):
        if self.model == None and LOAD_MODEL:
            rospy.loginfo('[INFO ]\tLoading model')
            model_loaded, self.sess = self.load()
        else:
            model_loaded = True
        if model_loaded:     
            # generate test batches
            test_batch_gen  = data_utils.batch_gen(self.testX, self.testY, batch_size)

            # predict output to current batch
            input_ = test_batch_gen.next()[0]
            predicted = self.model.predict(self.sess, input_)

            # print predictions
            replies = []
            # single_rep = ''
            for ii, oi in zip(input_.T, predicted):
                # single_rep = ''
                q = data_utils.decode(ii, self.metadata['idx2w'], separator = ' ')
                decoded = data_utils.decode(oi, self.metadata['idx2w'], separator = ' ').split(' ')

                if decoded.count('unk') == 0:
                    if decoded not in replies:
                        print('q : [{0}];\na : [{1}]\n'.format(q, ' '.join(decoded)))
                        replies.append(decoded)
                        # for reply in decoded:
                        #     single_rep += ' ' + reply
                        # replies.append(single_rep)
            if len(replies):
                return True
            else:
                return False
        else:
            rospy.logerr('[ERROR   ]\tNo model loaded. Load your model first to call the service.')
            return False

    def eval_server(self):
        s = rospy.Service('roboy/gnlp_eval', seq2seq_eval, self.evaluate)
        rospy.loginfo('Evaluating service ready to be called.')

    def predict(self, req):
        if self.model == None and LOAD_MODEL:
            rospy.loginfo('[INFO ]\tLoading model')
            model_loaded, self.sess = self.load()
        else:
            model_loaded = True
        if model_loaded:  
            # Load message
            rospy.loginfo('[INFO    ]\tService called with following input : %s', req.text_input)
            user_message = req.text_input

            # # encode input message
            encoded_message  = data_utils.encode(user_message, self.metadata['w2idx'], int(self.metadata['q_max']))
            encoded_message  = np.array(encoded_message).reshape((len(encoded_message), 1))
            # # decode output response
            response         = self.model.predict(self.sess, encoded_message)
            decoded_response = data_utils.decode(response[0], self.metadata['idx2w'], separator = ' ')
            rospy.loginfo(u"< %s" % decoded_response)
            print(decoded_response)
            # Return the response
            return True, decoded_response
        else:
            rospy.logerr('[ERROR   ]\tNo model loaded. Load your model first to call the service.')
            return False,''

    def predict_server(self):
        s = rospy.Service('roboy/gnlp_predict', seq2seq_predict, self.predict)
        rospy.loginfo("Prediction service ready to be called.")

def main():
    """
    Predict seq2seq
    """
    if not tf.gfile.Exists(ckpt_dir):
        tf.gfile.MakeDirs(ckpt_dir)
    rospy.init_node('GNLP')
    node = GNLP()
    while not rospy.is_shutdown():
        node.train_server()
        node.eval_server()
        node.predict_server()
        rospy.spin()


if __name__ == '__main__':
    dataset_dir = str(rospy.get_param('/seq2seq/dataset_dir', '/home/emilka/RC@H/roboy_ws/src/generative_nlp/include/Datasets/squad/processed'))
    ckpt_dir = str(rospy.get_param('/seq2seq/ckpt_dir', '/home/emilka/RC@H/roboy_ws/src/generative_nlp/include/Datasets/squad/checkpoints'))
    batch_size = int(rospy.get_param('/seq2seq/batch_size', 32))
    epochs = int(rospy.get_param('/seq2seq/epochs', 500))
    lr = float(rospy.get_param('/seq2seq/lr', 0.0001))
    num_layers = int(rospy.get_param('/seq2seq/num_layers', 3))
    emb_dim = int(rospy.get_param('/seq2seq/emb_dim', 1024))

    main()
