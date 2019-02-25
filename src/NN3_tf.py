import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from nn_utils import read_dataset, sigmoid

LR = 0.01
Xavier = 1


class NN3_tf:
    def __init__(self, dataset, epochs, w=None, print_step=None):
        self.train_x, self.train_y, self.test_x, self.test_y = dataset
        self.neurons = self.train_x.shape[1]
        self.samples = self.train_x.shape[0]
        self.keep_prob = tf.placeholder("float")
        self.error = 0

        self.x = tf.placeholder(tf.float32, shape=[None, self.neurons])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])

        b0 = tf.Variable(tf.zeros([self.samples, 1]), name="bias0", dtype=tf.float32)
        b1 = tf.Variable(tf.zeros([self.samples, 1]), name="bias1", dtype=tf.float32)
        b2 = tf.Variable(tf.zeros([self.samples, 1]), name="bias2", dtype=tf.float32)
        if w is None:
            W0 = tf.Variable(tf.truncated_normal([self.neurons, self.samples], seed=1), name="W0",
                             dtype=tf.float32) * Xavier
            b0 = tf.Variable(tf.zeros([self.samples, 1]), name="bias0", dtype=tf.float32)
            W1 = tf.Variable(tf.truncated_normal([self.samples, self.neurons], seed=0), name="W1",
                             dtype=tf.float32) * Xavier
            W2 = tf.Variable(tf.truncated_normal([self.neurons, 1], seed=0), name="W2", dtype=tf.float32) * Xavier
        else:
            W0 = tf.Variable(w[0], name="W0", dtype=tf.float32)
            W1 = tf.Variable(w[1], name="W1", dtype=tf.float32)
            W2 = tf.Variable(w[2], name="W2", dtype=tf.float32)

        l0 = tf.sigmoid(tf.add(tf.matmul(self.x, W0), b0))
        l1 = tf.sigmoid(tf.add(tf.matmul(l0, W1), b1))
        l1_dropout = tf.nn.dropout(l1, self.keep_prob)
        self.l2 = tf.sigmoid(tf.add(tf.matmul(l1_dropout, W2), b2))

        ### calculate the error
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.l2, labels=self.y))
        beta = 0.005
        # loss = tf.reduce_mean(loss + beta * tf.nn.l2_loss(self.l2))

        ###  decayed learning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LR, global_step,
                                                   100000, 0.96, staircase=True)
        # apply the optimization
        optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

        with tf.Session() as sess:
            self.sess = sess
            # init W & b
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                # run the optimizer
                y_prime, opt, lo = sess.run([self.l2, optimizer, loss],
                                            feed_dict={self.x: self.train_x, self.y: self.train_y, self.keep_prob: 1})

                self.error = np.mean(np.abs(self.train_y - y_prime))
                self.acc = self.calc_accuracy()
                evar = (self.train_y - y_prime).var()
                if (print_step is not None) and ((epoch % print_step == 0) or epoch == epochs):
                    print("{},{},{},{}".format(epoch, self.error, evar, self.acc))
            self.weights = np.array(self.sess.run([W0, W1, W2]))
            self.biasses = np.array(self.sess.run([b0, b1, b2]))

    def predict(self, X1):
        X1.resize((self.samples, self.neurons), refcheck=False)
        result = self.sess.run(self.l2, feed_dict={self.x: X1, self.y: self.test_y, self.keep_prob: 1})
        return result[:self.test_y.shape[0]]

    def calc_accuracy(self):
        y_error = (self.predict(self.test_x) - self.test_y)
        return 1 - np.mean(np.abs(y_error)), np.std(y_error)

    def get_error(self):
        return self.error

    def get_acc(self):
        return self.acc

    def get_weights(self):
        return self.weights

    def get_biasses(self):
        return self.biasses
