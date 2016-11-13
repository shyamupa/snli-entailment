import os
import sys
import tensorflow as tf
import numpy as np
from collections import Counter, defaultdict
from itertools import count
import random
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical, accuracy
import time

V = 10000
dim = 20
k = 100  # opts.lstm_units


def get_params():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lstm', action="store", default=150, dest="lstm_units", type=int)
    parser.add_argument('-epochs', action="store", default=20, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=32, dest="batch_size", type=int)
    parser.add_argument('-emb', action="store", default=100, dest="emb", type=int)
    parser.add_argument('-xmaxlen', action="store", default=120, dest="xmaxlen", type=int)
    parser.add_argument('-ymaxlen', action="store", default=70, dest="ymaxlen", type=int)
    parser.add_argument('-maxfeat', action="store", default=35000, dest="max_features", type=int)
    parser.add_argument('-classes', action="store", default=3, dest="num_classes", type=int)
    parser.add_argument('-sample', action="store", default=1, dest="samples", type=int)
    parser.add_argument('-nopad', action="store", default=False, dest="no_padding", type=bool)
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-load', action="store", default=False, dest="load_save", type=bool)
    parser.add_argument('-verbose', action="store", default=False, dest="verbose", type=bool)
    parser.add_argument('-train', action="store", default="train_all.txt", dest="train")
    parser.add_argument('-test', action="store", default="test_all.txt", dest="test")
    parser.add_argument('-dev', action="store", default="dev.txt", dest="dev")
    opts = parser.parse_args(sys.argv[1:])
    print ("lstm_units", opts.lstm_units)
    print ("epochs", opts.epochs)
    print ("batch_size", opts.batch_size)
    print ("emb", opts.emb)
    print ("samples", opts.samples)
    print ("xmaxlen", opts.xmaxlen)
    print ("ymaxlen", opts.ymaxlen)
    print ("max_features", opts.max_features)
    print ("no_padding", opts.no_padding)
    return opts


class CustomModel:
    def __init__(self, opts, sess, XMAXLEN, YMAXLEN, vocab, batch_size=1000):
        self.dim = 100
        self.sess = sess
        self.h_dim = opts.lstm_units
        self.batch_size = batch_size
        self.vocab_size = len(vocab)
        self.XMAXLEN = XMAXLEN
        self.YMAXLEN = YMAXLEN

    # def last_relevant(output, length):
    #     batch_size = tf.shape(output)[0]
    #     max_length = tf.shape(output)[1]
    #     out_size = int(output.get_shape()[2])
    #     index = tf.range(0, batch_size) * max_length + (length - 1)
    #     flat = tf.reshape(output, [-1, out_size])
    #     relevant = tf.gather(flat, index)
    #     return relevant

    # def repeat(x, n):
    #     '''Repeats a 2D tensor:
    #     if x has shape (samples, dim) and n=2,
    #     the output will have shape (samples, 2, dim)
    #     '''
    #     x = tf.expand_dims(x, 1)
    #     pattern = tf.pack([1, n, 1])
    #     return tf.tile(x, pattern)

    def build_model(self):
        self.x = tf.placeholder(tf.int32, [self.batch_size, self.XMAXLEN], name="premise")
        self.x_length = tf.placeholder(tf.int32, [self.batch_size], name="premise_len")
        self.y = tf.placeholder(tf.int32, [self.batch_size, self.YMAXLEN], name="hypothesis")
        self.y_length = tf.placeholder(tf.int32, [self.batch_size], name="hyp_len")
        self.target = tf.placeholder(tf.float32, [self.batch_size,3], name="label")  # change this to int32 and it breaks.

        # DO NOT DO THIS
        # self.batch_size = tf.shape(self.x)[0]  # batch size
        # self.x_length = tf.shape(self.x)[1]  # batch size
        # print self.batch_size,self.x_length

        self.embed_matrix = tf.get_variable("embeddings", [self.vocab_size, self.dim])
        self.x_emb = tf.nn.embedding_lookup(self.embed_matrix, self.x)
        self.y_emb = tf.nn.embedding_lookup(self.embed_matrix, self.y)

        print self.x_emb, self.y_emb
        with tf.variable_scope("encode_x"):
            self.fwd_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.h_dim, state_is_tuple=True)
            self.x_output, self.x_state = tf.nn.dynamic_rnn(cell=self.fwd_lstm, inputs=self.x_emb, dtype=tf.float32)
            # self.x_output, self.x_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fwd_lstm,cell_bw=self.bwd_lstm,inputs=self.x_emb,dtype=tf.float32)
            print self.x_output
            # print self.x_state
        # print tf.shape(self.x)
        with tf.variable_scope("encode_y"):
            self.fwd_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.h_dim, state_is_tuple=True)
            self.y_output, self.y_state = tf.nn.dynamic_rnn(cell=self.fwd_lstm, inputs=self.y_emb,
                                                            initial_state=self.x_state, dtype=tf.float32)
            # print self.y_output
            # print self.y_state

        self.Y = self.x_output  # its length must be x_length

        # self.h_n = self.last_relevant(self.y_output,self.x_length)   # TODO
        tmp5= tf.transpose(self.y_output, [1, 0, 2])
        self.h_n = tf.gather(tmp5, int(tmp5.get_shape()[0]) - 1)
        print self.h_n

        # self.h_n_repeat = self.repeat(self.h_n,self.x_length)   # TODO
        self.h_n_repeat = tf.expand_dims(self.h_n, 1)
        pattern = tf.pack([1, self.XMAXLEN, 1])
        self.h_n_repeat = tf.tile(self.h_n_repeat, pattern)

        self.W_Y = tf.get_variable("W_Y", shape=[self.h_dim, self.h_dim])
        self.W_h = tf.get_variable("W_h", shape=[self.h_dim, self.h_dim])

        # TODO compute M = tanh(W*Y + W*[h_n...])
        tmp1 = tf.matmul(tf.reshape(self.Y, shape=[self.batch_size * self.XMAXLEN, self.h_dim]), self.W_Y,
                         name="Wy")
        self.Wy = tf.reshape(tmp1, shape=[self.batch_size, self.XMAXLEN, self.h_dim]);
        tmp2 = tf.matmul(tf.reshape(self.h_n_repeat, shape=[self.batch_size * self.XMAXLEN, self.h_dim]), self.W_h)
        self.Whn = tf.reshape(tmp2, shape=[self.batch_size, self.XMAXLEN, self.h_dim], name="Whn");
        self.M = tf.tanh(tf.add(self.Wy, self.Whn), name="M")
        # print "M",self.M

        # use attention
        self.W_att = tf.get_variable("W_att",shape=[self.h_dim,1]) # h x 1
        tmp3 = tf.matmul(tf.reshape(self.M,shape=[self.batch_size*self.XMAXLEN,self.h_dim]),self.W_att)
        # need 1 here so that later can do multiplication with h x L
        self.att = tf.nn.softmax(tf.reshape(tmp3,shape=[self.batch_size,1, self.XMAXLEN],name="att")) # nb x 1 x Xmax
        # print "att",self.att

        # COMPUTE WEIGHTED
        self.r = tf.reshape(tf.batch_matmul(self.att, self.Y, name="r"),shape=[self.batch_size,self.h_dim])  # (nb,1,L) X (nb,L,k) = (nb,1,k)
        # get last step of Y as r which is (batch,k)
        # tmp4 = tf.transpose(self.Y, [1, 0, 2])
        # self.r = tf.gather(tmp4, int(tmp4.get_shape()[0]) - 1)
        # print "r",self.r

        self.W_p, self.b_p= tf.get_variable("W_p", shape=[self.h_dim, self.h_dim]), tf.get_variable("b_p",shape=[self.h_dim],initializer=tf.constant_initializer())
        self.W_x, self.b_x = tf.get_variable("W_x", shape=[self.h_dim, self.h_dim]), tf.get_variable("b_x",shape=[self.h_dim],initializer=tf.constant_initializer())
        self.Wpr = tf.matmul(self.r, self.W_p, name="Wy") + self.b_p
        self.Wxhn = tf.matmul(self.h_n, self.W_x, name="Wxhn") + self.b_x
        self.hstar = tf.tanh(tf.add(self.Wpr, self.Wxhn), name="hstar")
        # print "Wpr",self.Wpr
        # print "Wxhn",self.Wxhn
        # print "hstar",self.hstar

        self.W_pred = tf.get_variable("W_pred", shape=[self.h_dim, 3])
        self.pred = tf.nn.softmax(tf.matmul(self.hstar, self.W_pred), name="pred_layer")
        # print "pred",self.pred,"target",self.target
        correct = tf.equal(tf.argmax(self.pred,1),tf.argmax(self.target,1))
        self.acc = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
        # self.H_n = self.last_relevant(self.en_output)
        self.loss = -tf.reduce_sum(self.target * tf.log(self.pred), name="loss")
        # print self.loss
        self.optimizer = tf.train.AdamOptimizer()
        self.optim = self.optimizer.minimize(self.loss, var_list=tf.trainable_variables())
        _ = tf.scalar_summary("loss", self.loss)

    def train(self,\
              xdata, ydata, zdata, x_lengths, y_lengths,\
              xxdata, yydata, zzdata, xx_lengths, yy_lengths,\
              MAXITER):
        merged_sum = tf.merge_all_summaries()
        # writer = tf.train.SummaryWriter("./logs/%s" % "modeldir", self.sess.graph_def)
        tf.initialize_all_variables().run()
        start_time = time.time()
        for ITER in range(MAXITER):
            # xdata, ydata, zdata, x_lengths, y_lengths = joint_shuffle(xdata, ydata, zdata, x_lengths, y_lengths)
            for i in xrange(0, len(l), self.batch_size):
                x,y,z,xlen,ylen=xdata[i:i + self.batch_size],\
                                ydata[i:i + self.batch_size],\
                                zdata[i:i + self.batch_size],\
                                x_lengths[i:i + self.batch_size],\
                                y_lengths[i:i + self.batch_size]
                feed_dict = {self.x: x,\
                             self.y: y,\
                             self.target: z,\
                             self.x_length:xlen,\
                             self.y_length:ylen}
                att, _ , loss, acc, summ = self.sess.run([self.att,self.optim, self.loss, self.acc, merged_sum],feed_dict=feed_dict)
                # print "att for 0th",att[0]
                print "loss",loss, "acc on train", acc
            total_test_acc=[]
            for i in xrange(0, len(l), self.batch_size):
                x,y,z,xlen,ylen=xxdata[i:i + self.batch_size],\
                                yydata[i:i + self.batch_size],\
                                zzdata[i:i + self.batch_size],\
                                xx_lengths[i:i + self.batch_size],\
                                yy_lengths[i:i + self.batch_size]
                tfeed_dict = {self.x: x,\
                              self.y: y,\
                              self.target: z,\
                              self.x_length:xlen,\
                              self.y_length:ylen}
                att, _ , test_loss, test_acc, summ = self.sess.run([self.att,self.optim, self.loss, self.acc, merged_sum],feed_dict=tfeed_dict)
                total_test_acc.append(test_acc)
            print "acc on test",np.mean(total_test_acc)
        # for x, y, z in zip(xdata, ydata, zdata):
            # print x, y, z
            # feeddict = {self.x: x, self.y: y, self.target: z, self.x_length:x_lengths, self.y_length:y_lengths}
            # self.sess.run([self.optim, self.loss, merged_sum],feed_dict=feeddict);
        elapsed_time = time.time() - start_time
        print "total time",elapsed_time

def joint_shuffle(xdata, ydata, zdata, x_lengths, y_lengths):
    tmp=list(zip(xdata, ydata, zdata, x_lengths, y_lengths))
    random.shuffle(tmp)
    xdata, ydata, zdata, x_lengths, y_lengths = zip(*tmp)
    return xdata, ydata, zdata, x_lengths, y_lengths
if __name__ == "__main__":
    from reader import *
    from myutils import *

    options = get_params()
    train = [l.strip().split('\t') for l in open(options.train)]
    dev = [l.strip().split('\t') for l in open(options.dev)]
    test = [l.strip().split('\t') for l in open(options.test)]
    vocab = get_vocab(train)

    X_train, Y_train, Z_train = load_data(train, vocab)
    X_dev, Y_dev, Z_dev = load_data(dev, vocab)
    X_test, Y_test, Z_test = load_data(test, vocab)
    # print Z_train[1]
    # sys.exit()

    X_train_lengths = [len(x) for x in X_train]
    X_dev_lengths = np.asarray([len(x) for x in X_dev]).reshape(len(X_dev))
    X_test_lengths = np.asarray([len(x) for x in X_test]).reshape(len(X_test))
    # print len(X_test_lengths)

    Y_train_lengths = np.asarray([len(x) for x in Y_train]).reshape(len(Y_train))
    Y_dev_lengths = np.asarray([len(x) for x in Y_dev]).reshape(len(Y_dev))
    Y_test_lengths = np.asarray([len(x) for x in Y_test]).reshape(len(Y_test))
    # print len(Y_test_lengths)

    Z_train = to_categorical(Z_train, nb_classes=options.num_classes)
    Z_dev = to_categorical(Z_dev, nb_classes=options.num_classes)
    Z_test = to_categorical(Z_test, nb_classes=options.num_classes)
    # print Z_train[0]

    XMAXLEN = options.xmaxlen
    YMAXLEN = options.ymaxlen
    MAXITER = 1000
    X_train = pad_sequences(X_train, maxlen=XMAXLEN, value=vocab["unk"], padding='post') ## NO NEED TO GO TO NUMPY , CAN GIVE LIST OF PADDED LIST
    X_dev = pad_sequences(X_dev, maxlen=XMAXLEN, value=vocab["unk"], padding='post')
    X_test = pad_sequences(X_test, maxlen=XMAXLEN, value=vocab["unk"], padding='post')
    Y_train = pad_sequences(Y_train, maxlen=YMAXLEN, value=vocab["unk"], padding='post')
    Y_dev = pad_sequences(Y_dev, maxlen=YMAXLEN, value=vocab["unk"], padding='post')
    Y_test = pad_sequences(Y_test, maxlen=YMAXLEN, value=vocab["unk"], padding='post')
    print X_test.shape, X_test_lengths.shape
    vocab = get_vocab(train)
    with tf.Session() as sess:
        model = CustomModel(options, sess, XMAXLEN, YMAXLEN, vocab, batch_size=200)
        model.build_model()
        model.train(X_train,Y_train,Z_train,X_train_lengths,Y_train_lengths,\
                    X_test,Y_test,Z_test,X_test_lengths,Y_test_lengths,\
                    MAXITER)
