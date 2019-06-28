import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 数据已经转化为矩阵，打成gz压缩格式
mnist = input_data.read_data_sets("../data/input_data")
print(mnist)