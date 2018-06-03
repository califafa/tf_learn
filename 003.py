import tensorflow as tf 

state=tf.Variable(0,name='counter') 
# print(state.name) 
one=tf.constant(3) 

new_value=tf.add(state,one) 
update=tf.assign(state,new_value) 
update2=tf.assign(state,update)

# init=tf.initialize_all_variables() #must have if define variable
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        sess.run(update)
        sess.run(update2)
        sess.run(state)
        print(sess.run(new_value))

#tf.run(参数) 里面“参数”涉及到的变量都会被run