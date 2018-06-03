import tensorflow as tf 

input1=tf.placeholder(tf.float32) #placeholder是占位符，现在input1的值是没有的，
input2=tf.placeholder(tf.float32)   #需要等run的时候传进去，详见第9行

output=tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[2.],input2:[3.]}))