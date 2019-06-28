import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#多项式的线性回归
# y_predict = X * w + b
# y_predict = X^2 X^3 X^4 可以是
# y_predict = X^2 * w2 + X^3 * w3 + X * w + b

# 自动生成的特征值n
number = 100
X = np.linspace(-3,3,number,dtype='float32')
# 增加一些随机数 y是目标值
y = np.sin(X) + np.random.uniform(-0.5,0.5,number)
#plt.scatter(X,y)
#plt.show()

# [100,2] dot [2,1] ==> [100,1]
# 100个样本
# [100] * [w] + b ==> y_predict [100]
# tensorflow中，训练的过程就是在训练权重 + 偏置，必须用变量存储
w1 = tf.Variable(initial_value = 0.0)
w2 = tf.Variable(initial_value = 0.0)
w3 = tf.Variable(initial_value = 0.0)
b = tf.Variable(initial_value = 0, name ='b',dtype='float32')
# 特征值 * 权重 + 偏置 ==》预测值
y_predict = tf.add(tf.multiply(X,w1),b)
y_predict = tf.add(tf.multiply(tf.pow(X,2),w2),y_predict)
y_predict = tf.add(tf.multiply(tf.pow(X,3),w3),y_predict)
# 获取均方误差
loss = tf.reduce_mean(tf.square(y - y_predict))
# 可以把误差添加到图中
# 把所有监控的tensor整合到一起然后提交到图中
tf.summary.scalar("abc",loss)
merge = tf.summary.merge_all()
# 通过梯度下降来减少误差 步长：
train_op = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 把图保存起来然后tensorflow打开
    fw = tf.summary.FileWriter("../data/",sess.graph)
    for i in range(1000):
        sess.run(train_op)
        fw.add_summary(merge.eval(), i)
        if i % 10 == 0:
            # 两种方式拿到张量的值 sess.run(loss)或者los.eval()
            print(f'第{i}次均方误差为{loss.eval()},偏置为{b.eval()}')

    # 训练结束，采用可视化显示预测值与真实值
    plt.scatter(X,y)
    plt.plot(X,sess.run(y_predict))
    plt.show()




