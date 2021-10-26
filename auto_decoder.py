import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as plt
mnist = read_data_sets("./MNIST", validation = 0)


hidden_num = 32
images_size = minist.train.images.shape[1]
inputs_ = tf.placeholder(tf.float32, (None, images_size), name="inputs")
targets_ = tf.placeholder(tf.float32, (None, images_size), name ="inputs")


#hidden_layer
encode = tf.layers.dense(inputs_, hidden_num, activation=tf.nn.relu)
logits = tf.layers.dense(encode, images_size, activation=None)

decode = tf.nn.sigmoid(logits, name="outputs")

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_,logits=logits))
opt = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sees:
    sees.run(tf.global_variables_initializer())
    epoches = 50
    batch_size = 64
    for i in range(epoches):
        for li in range(mnist.train.num_example//batch_size):
            batch = mnist.train.next_batch(batch_size)
            cost_loss, _ = sees.run([loss, opt], feed_dict={inputs_:batch[0], targets_:batch[0]})
            print("epoches:{}/{}".format(i, epoches),"loss is {}".format(cost_loss))

    fig, axes = plt.subplots(2, 10, sharex=True, sharey=True, figsize)
