import numpy as np

#create sample
print('求权重公式')
t1 = np.arange(12).reshape(3,4)
#create [4,1]
t2 = np.arange(4).reshape(4,1)
#点积 [3,4] [4,1] --> [3,1]
t3 = t1.dot(t2)


print('求误差公式')
# 真实值
y_true = np.arange(10).reshape(10,1)
# 预测值
y_redict = np.arange(10,20).reshape(10,1)
print(y_redict)
# 线性回归中有均方误差公式
#t4 = y_redict - y_true
#t5 = t4*t4
# 对t5求和再求平均
#print(f'误差值{t5.mean()},{t5.sum()/t5.size()}')