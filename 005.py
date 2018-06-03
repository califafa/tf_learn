import tensorflow as tf 

#激励函数必须是可微分的，因为需要 误差反向传递

#推荐的激励函数
    #少量层结构
        #卷积神经网络-》relu
        #循环神经网络-》relu or tanh

#形如 tf.nn.relu(features,name=None)