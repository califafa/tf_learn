import tensorflow as tf
import numpy as np

#create data 

x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3


#create tf structure start

Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))

y=Weights*x_data+biases

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()#初始化所有变量，现已有 tf.global_variables_initializer()，用法一样
#create tf structure end

sess=tf.Session()
sess.run(init)

for step in range(101):
    sess.run(train)
    if step%10==0:
        print(step,sess.run(Weights),sess.run(biases))