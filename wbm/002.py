import tensorflow as tf 
import numpy as np 
import random

#确定答案
var1=8
var2=42
#先生成学习数据
x_data=np.float64(np.random.rand(1,100))
# y_data=np.dot([23],x_data)
y_data=var1*x_data+var2

#构造一个线性模型
b=tf.Variable(tf.zeros([1]))
w=tf.Variable(tf.random_uniform([1],-1.0,1.0))
#w和b都能用，如下：
y=tf.multiply(w,x_data)+b
#y=tf.multiply(b,x_data)

#最小化方差
loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

#初始化变量
init=tf.initialize_all_variables()

#启动图
sess=tf.Session()
sess.run(init)

#拟合一次函数
for i in range(1000):
    sess.run(train)
    if i%25==0:
        # print(sess.run(y))
        print(sess.run(w),sess.run(b))