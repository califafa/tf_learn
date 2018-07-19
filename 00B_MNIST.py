import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#加载数据
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#定义回归模型
x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,784]))
W_1=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.matmul(tf.matmul(x,W),W_1)+b

#定义损失函数和优化器
y_=tf.placeholder(tf.float32,[None,10])#真实数据结果
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))#y和y_的差距

#采用SGD作为优化器
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

#Train 
for _ in range(10000):
    batch_xs,batch_ys=mnist.train.next_batch(200)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})


#评估模型
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('x:',x,'==',batch_xs[0])
print('y_:',y,'==',batch_ys[0])
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
