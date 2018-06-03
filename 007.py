import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

#定义添加神经层的函数
def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size])) 
        biases=tf.Variable(tf.zeros([1,out_size])+0.1) 
        Wx_plus_b=tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        return outputs

x_data=np.linspace(-1,1,500)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise
with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,activation_function=None)
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
init=tf.initialize_all_variables()
sess=tf.Session()
#writer=tf.train.SummaryWriter("logs\\",sess.graph)
#现为 
writer=tf.summary.FileWriter("logs\\",sess.graph)
sess.run(init)

#可视化start
fig=plt.figure()
plt.xlim((-1.1,1.1))
plt.ylim((-1.1,1.1))
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
#可视化end

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50:

        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        
        try:
            ax.lines.remove(lines[0])
            print('training',i)
        except Exception:
            pass
        prediction_value=sess.run(prediction,feed_dict={xs:x_data})
        lines=ax.plot(x_data,prediction_value,'r-',lw=2)
        plt.pause(0.01)

#运行本代码后，在\logs目录生成一个文件
#命令行 tensorboard --logdir='c:\code\tf\logs' 启动http服务器