import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# (55000, 784)
mnist = input_data.read_data_sets("../data/input_data",one_hot=True)
# 数据已经转化成矩阵,打成gz压缩格式
# (55000, 784)
print(mnist.train.images.shape)
# (55000,) [7,3,4...........]
# one-hot: (55000, 10)
print(mnist.train.labels.shape)
print(mnist.train.next_batch(1))

X = tf.placeholder(dtype=tf.float32, shape=[None,784])
# 设置one-hot编码
y = tf.placeholder(dtype=tf.float32, shape=[None,10])

#  X [None,785]  * [784,10]  ==> [None,10]
# 添加隐藏层：让简单神经网络变成深度神经网络
# 假设第一层500个神经元  [None,784]  * [784,500]  ==> [None,500]
W1 = tf.Variable(initial_value=tf.random_normal(shape=[784,500]),dtype=tf.float32)
B1 = tf.Variable(initial_value=tf.random_normal(shape=[500]),dtype=tf.float32)
L1 =  tf.add(tf.matmul(X,W1),B1)
# 想要提高DNN命中率，应该在隐藏层的输出之后追加激活函数（直线转化曲线）
L1 = tf.nn.tanh(L1)
# 假设第二层200个神经元  [None,500]  * [500,200]  ==> [None,200]
W2 = tf.Variable(initial_value=tf.random_normal(shape=[500,200]),dtype=tf.float32)
B2 = tf.Variable(initial_value=tf.random_normal(shape=[200]),dtype=tf.float32)
L2 =  tf.add(tf.matmul(L1,W2),B2)
L2 = tf.nn.tanh(L2)
# 输出层，有多少个类别，则会有多少个输出层
weight = tf.Variable(initial_value=tf.random_normal(shape=[200,10]),dtype=tf.float32)
# 偏置,有多少个神经元就会有多少个偏置
bias = tf.Variable(initial_value=tf.random_normal(shape=[10]),dtype=tf.float32)
# 根据公式 y_predict = X * w + b  生成预测值
# [50,784] ==> [784,10] + [10]==> [50,10]
y_predict = tf.add(tf.matmul(L2,weight),bias)
# 线性回归的用的均方误差,
# loss = tf.reduce_mean(tf.square(y_predict - y))
# 而分类y_predict应该先转化为概率,在求误差
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_predict))
# 采用梯度下降减少误差
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 机器/深度学习都有  回归和分类算法
    # 回归: 预测值(一个样本一个预测值)与真实值的误差
    # 分类: 输出的不是一个标准值,输出的是属于某个类别的概率 （y 目标值必须one-hot编码）
# 求分类的正确率   y_predict [55,10 ]  y [55,10]
a,b = tf.argmax(y_predict,axis=1),tf.argmax(y,axis=1)
# 返回一个bool值矩阵
result = tf.equal(a,b)
result = tf.cast(result,dtype=tf.float32)
# [0.0,1.0] = 1.0 / 2
result = tf.reduce_mean(result)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 55000 样本，如果循环1000次则每次应该获取55个样本
    for i in range(5000):
        # 如果依赖占位符,则运算时必须指定
        X_train,y_train = mnist.train.next_batch(55)
        d = {X: X_train, y: y_train}
        sess.run(train_op,feed_dict=d)
        # 下一个版本求正确率
        print(f'第{i}次的误差为{sess.run(loss,feed_dict=d)},正确率为:{sess.run(result,feed_dict=d)}')



