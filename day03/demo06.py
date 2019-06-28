import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
w = tf.Variable(initial_value = 0.0, name ='w')
b = tf.Variable(initial_value = 0, name ='b',dtype='float32')
# 特征值 * 权重 + 偏置 ==》预测值

y_predict = tf.add(tf.multiply(X,w),b,name='mutiply_add')
# 获取均方误差
loss = tf.reduce_mean(tf.square(y - y_predict))
# 通过梯度下降来减少误差 步长：
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        sess.run(train_op)
        if i % 10 == 0:
            # 两种方式拿到张量的值 sess.run(loss)或者los.eval()
            print(f'第{i}次均方误差为{loss.eval()},权重为{w.eval()},偏置为{b.eval()}')

    # 训练结束，采用可视化显示预测值与真实值
    plt.scatter(X,y)
    plt.plot(X,sess.run(y_predict))
    plt.show()




