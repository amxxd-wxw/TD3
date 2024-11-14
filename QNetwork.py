import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow_probability import distributions as tfd

# QNetwork 类被用于建立批判者的 Q 网络。这里使用了另一种建立网络的方法,通过继承  Model 类并重构 forward 函数来建立网络模型。

class QNetwork(Model):
    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        input_dim = num_inputs + num_actions
        w_init = tf.random_uniform_initializer(-init_w, init_w)
        self.linear1 = Dense(units=hidden_dim, activation=tf.nn.relu,
                             kernel_initializer=w_init, name='q1')
        self.linear2 = Dense(units=hidden_dim, activation=tf.nn.relu,
                             kernel_initializer=w_init, name='q2')
        self.linear3 = Dense(units=1, kernel_initializer=w_init, name='q3')

    def call(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
