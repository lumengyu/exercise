import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/input_data")
# 数据已经转化成矩阵,打成gz压缩格式

print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.train.images[0])
print(mnist.train.labels[0])
 # [None,784] ==> [784,1] ==> [None,1]
X = tf.placeholder(dtype=tf.float32,shape=[None,784])
y = tf.placeholder(dtype=tf.float32,shape=[None,1])

