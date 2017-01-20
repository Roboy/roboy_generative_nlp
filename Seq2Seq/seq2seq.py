# ============================================================== #
#                        Seq2Seq-model                           #
#                                                                #
#                                                                #
# Seq2Seq model wrapper for easy training and prediction         #
# ============================================================== #

from __future__ import print_function

import tensorflow as tf
import numpy as np
import os.path

class Seq2Seq(object):

    def __init__(self, xseq_len, yseq_len, 
                xvocab_size, yvocab_size,
                emb_dim, num_layers, ckpt_path,
                lr = 0.0001, epochs = 100000,
                model_name = 'seq2seq_model'):

        # attach these arguments to self
        self.xseq_len   = xseq_len
        self.yseq_len   = yseq_len
        self.ckpt_path  = ckpt_path
        self.epochs     = epochs
        self.lr         = lr
        self.model_name = model_name

        print('[PROGRESS]\tBuilding Seq2Seq Graph')
        self.build(xvocab_size, yvocab_size, emb_dim, num_layers)


    def build(self, xvocab_size, yvocab_size, emb_dim, num_layers):
        """
        Build seq2seq model with dropout:
        ----------
        Args:
            xvocab_size: input vocab size
            yvocab_size: output vocab size
            emb_dim: embded dimension (1024)
            num_layers: number of layers
        """

        tf.reset_default_graph()

        #  encoder inputs : list of indices of length xseq_len
        self.enc_ip = [ tf.placeholder(shape = [None,], 
                        dtype = tf.int64, 
                        name = 'ei_{}'.format(t)) for t in range(self.xseq_len) ]

        #  labels that represent the real outputs
        self.labels = [ tf.placeholder(shape = [None,], 
                        dtype = tf.int64, 
                        name = 'ei_{}'.format(t)) for t in range(self.yseq_len) ]

        #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
        self.dec_ip = [ tf.zeros_like(self.enc_ip[0], dtype = tf.int64, name = 'GO') ] + self.labels[:-1]

        # Basic LSTM cell wrapped in Dropout Wrapper
        self.keep_prob = tf.placeholder(tf.float32)

        # define the basic cell
        basic_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(emb_dim, state_is_tuple = True),
                output_keep_prob = self.keep_prob)

        # stack cells together : n layered model
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([basic_cell]*num_layers, state_is_tuple = True)

        # for parameter sharing between training and testing model
        with tf.variable_scope('decoder') as scope:
            # build the seq2seq model 
            #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
            self.decode_outputs, self.decode_states = tf.nn.seq2seq.embedding_rnn_seq2seq(self.enc_ip,self.dec_ip, stacked_lstm,
                                                xvocab_size, yvocab_size, emb_dim)
            # share parameters
            scope.reuse_variables()
            # testing model, where output of previous timestep is fed as input 
            #  to the next timestep
            self.decode_outputs_test, self.decode_states_test = tf.nn.seq2seq.embedding_rnn_seq2seq(
                self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                feed_previous = True)

        # weighted loss
        loss_weights = [ tf.ones_like(label, dtype = tf.float32) for label in self.labels ]
        self.loss = tf.nn.seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights, yvocab_size)

        # train op to minimize the loss
        self.train_op = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss)


    def get_feed(self, x, y, keep_prob):
        """
        Get the feed dictionary:
        ----------
        Args:
            x: input
            y: output
            keep_prob: dropout probability

        Returns:
            feed dict
        """

        feed_dict = {self.enc_ip[t]: x[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob

        return feed_dict


    def eval_step(self, sess, eval_batch_gen):
        """
        Evaluate one batch:
        ----------
        Args:
            sess: current session
            eval_batch_gen: batch to evalute

        Returns:
            batch loss, dec_op_v, batchX, batchY
        """

        # get batches
        batchX, batchY = eval_batch_gen.next()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob = 1.0)
        loss_v, dec_op_v = sess.run([self.loss, self.decode_outputs_test], feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        # interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1, 0, 2])

        return loss_v, dec_op_v, batchX, batchY


    def eval_batches(self, sess, eval_batch_gen, num_batches):
        """
        Evaluate num of batches:
        ----------
        Args:
            sess: current session
            eval_batch_gen: batch to evalute
            num_batches: num of batches to test

        Returns:
            mean of batches loss
        """

        losses = []
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.eval_step(sess, eval_batch_gen)
            losses.append(loss_v)

        return np.mean(losses)


    def train_batch(self, sess, train_batch_gen):
        """
        Run one batch for training:
        ----------
        Args:
            sess: current session
            train_batch_gen: batch to train with

        Returns:
            batch loss
        """

        # get batches
        batchX, batchY = train_batch_gen.next()
        # build feed
        feed_dict = self.get_feed(batchX, batchY, keep_prob = 0.5)
        _, loss_v = sess.run([self.train_op, self.loss], feed_dict)

        return loss_v


    def train(self, train_set, valid_set, sess = None):
        """
        Train function that runs the train_op 
        in a session evaluates on valid set 
        periodically prints statistics:
        ----------
        Args:
            train_set: training set
            valid_set: validation set
            sess: current session

        Returns:
            trained session
        """

        # we need to save the model periodically
        saver = tf.train.Saver()

        # if no session is given
        if not sess:
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())

        print('[INFO    ]\tTraining started')
        for i in range(self.epochs):
            try:
                self.train_batch(sess, train_set)
                if i and i % (self.epochs//100) == 0:
                    # save model to disk
                    saver.save(sess, os.path.join(self.ckpt_path, self.model_name) + '.ckpt', global_step = i)
                    # evaluate to get validation loss
                    val_loss = self.eval_batches(sess, valid_set, 16)
                    # print stats
                    print('[PROGRESS]\tModel saved to disk at iteration #{}'.format(i))
                    print('[PROGRESS]\tVal loss : {0:.6f}'.format(val_loss))
            except KeyboardInterrupt:
                print('[WARNING ]\tInterrupted by user at iteration {}'.format(i))
                self.session = sess

                return sess


    def restore_last_session(self):
        """
        Restore last session from checkpoints dir:
        ----------
        Returns:
            last session
        """

        saver = tf.train.Saver()
        # create a session
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            print('[INFO    ]\tRestoring latest session..')
            saver.restore(sess, ckpt.model_checkpoint_path)

        # return to user
        return sess


    def predict(self, sess, x):
        """
        Predict output to a given input:
        ----------
        Args:
            sess: current session
            x: input sequence

        Returns:
            predicted sequence
        """

        feed_dict = {self.enc_ip[t]: x[t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        # interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])

        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis = 2)
