import tensorflow as tf 
import numpy as np 

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data=np.float32(np.random.rand(2,100))
y_data=np.dot([0.1,0.2],x_data)+0.3

# 构造一个线性模型
b=tf.Variable(tf.zeros([1]))
w=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y=tf.matmul(w,x_data)+b

# 最小化方差 
loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

# 初始化变量
init=tf.initialize_all_variables()

# 启动图 (graph)
sess=tf.Session()
sess.run(init)

# 拟合平面
for i in range(0,2000):
    sess.run(train)
    if i%20==0:
        print(i,sess.run(w),sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]