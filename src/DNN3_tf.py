import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from nn_utils import read_dataset, sigmoid

size = 500
LR = 0.01


class NN3_tf:
    def __init__(self, train_x, train_y, test_x, test_y, epochs, w=None, print_step=None):
        self.neurons = train_x.shape[1]
        self.samples = train_x.shape[0]
        self.keep_prob = tf.placeholder("float")
        self.error = 0

        Xavier = 0.001
        self.x = tf.placeholder(tf.float32, shape=[None, self.neurons])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        self.W0 = tf.Variable(tf.truncated_normal([self.neurons, self.samples], seed=1), name="self.W0",
                              dtype=tf.float32) * Xavier
        self.b0 = tf.Variable(tf.zeros([self.samples, 1]), name="bias0", dtype=tf.float32)
        self.W1 = tf.Variable(tf.truncated_normal([self.samples, self.neurons], seed=0), name="self.W1",
                              dtype=tf.float32) * Xavier
        self.b1 = tf.Variable(tf.zeros([self.samples, 1]), name="bias1", dtype=tf.float32)
        self.W2 = tf.Variable(tf.truncated_normal([self.neurons, 1], seed=0), name="self.W2", dtype=tf.float32) * Xavier
        self.b2 = tf.Variable(tf.zeros([self.samples, 1]), name="bias2", dtype=tf.float32)

        l0 = tf.sigmoid(tf.add(tf.matmul(self.x, self.W0), self.b0))
        l1 = tf.sigmoid(tf.add(tf.matmul(l0, self.W1), self.b1))
        l1_dropout = tf.nn.dropout(l1, self.keep_prob)
        self.l2 = tf.sigmoid(tf.add(tf.matmul(l1_dropout, self.W2), self.b2))

        ### calculate the error
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.l2, labels=self.y))
        beta = 0.005
        loss = tf.reduce_mean(loss + beta * tf.nn.l2_loss(self.l2))

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
                                            feed_dict={self.x: train_x, self.y: train_y, self.keep_prob: 0.8})

                if (print_step is not None) and ((epoch % print_step == 0) or epoch == epochs):
                    self.error = np.mean(np.abs(train_y - y_prime))
                    evar = (train_y - y_prime).var()
                    print("{},{},{},{}".format(epoch,self.error,evar,self.calc_accuracy(test_x, test_y)))

    def predict(self, X1):
        X1.resize((self.samples, self.neurons), refcheck=False)
        result = self.sess.run(self.l2, feed_dict={self.x: X1, self.y: test_y, self.keep_prob: 1})
        return result[:test_y.shape[0]]

    def calc_accuracy(self, test_x, test_y):
        y_error = (self.predict(test_x) - test_y)
        return 1 - np.mean(np.abs(y_error)), np.std(y_error)

    def get_error(self):
        return self.error



X, Y = read_dataset(180, 500)
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, test_size=0.3, random_state=1)

epochs = 6000

nn3 = NN3_tf(train_x, train_y, test_x, test_y, epochs, print_step=600)
print('EOC')
