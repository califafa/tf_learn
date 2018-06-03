import tensorflow as tf 
import numpy as np 
import random

#确定答案
var1=8
var2=42
var3=2
#先生成学习数据
x_data=np.float64(np.random.rand(1,30000))
y_data=np.float64(np.random.rand(1,30000))
# z_data=np.dot([23],x_data)
z_data=var1*x_data+var2+var3*2*(y_data**2)

#构造一个线性模型
m=tf.Variable(tf.zeros([1]))
b=tf.Variable(tf.zeros([1]))
w=tf.Variable(tf.random_uniform([1],-0.1,0.1))
#w和b都能用，如下：
z=tf.multiply(w,x_data)+b+m*2*(y_data**2)
#y=tf.multiply(b,x_data)

#最小化方差
loss=tf.reduce_mean(tf.square(z-z_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

#初始化变量
init=tf.initialize_all_variables()

#启动图
sess=tf.Session()
sess.run(init)

#拟合平面面
for i in range(30001):
    sess.run(train)
    if i%10000==0:
        # print(sess.run(y))
        print(sess.run(w),sess.run(m),sess.run(b))