import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.python.ops.distributions.normal import Normal
from tensorflow_probability import distributions as tfd

# PolicyNetwork 类用于建立行动者的策略网络。它在建立网络模型的同时,也增加了 evaluate()、get_action(),sample_action() 函数

class PolicyNetwork(Model):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        w_init = tf.random_uniform_initializer(-init_w, init_w)
        self.linear1 = Dense(units=hidden_dim, activation=tf.nn.relu, kernel_initializer=w_init, name='policy1')
        self.linear2 = Dense(units=hidden_dim, activation=tf.nn.relu, kernel_initializer=w_init, name='policy2')
        self.linear3 = Dense(units=hidden_dim, activation=tf.nn.relu, kernel_initializer=w_init, name='policy3')
        # 修正后的 output_linear 层
        self.output_linear = Dense(
            units=num_actions,
            activation=None,  # 设置为 None 而不是初始化器
            kernel_initializer=w_init,
            bias_initializer=w_init,
            name='policy_output'
        )
        self.action_range = action_range
        self.num_actions = num_actions

    def call(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)
        output = tf.nn.tanh(self.output_linear(x)) # 这里的输出范围是 [-1, 1]
        return output

    def evaluate_action(self, state, eval_noise_scale):
        state = state.astype(np.float32)
        action = self.call(state)
        action = self.action_range * action
        normal = tfd.Normal(loc=0.0, scale=1.0)
        eval_noise_clip = 2 * eval_noise_scale
        noise = normal.sample(action.shape)
        noise = tf.clip_by_value(noise * eval_noise_scale, -eval_noise_clip, eval_noise_clip)
        action = action + noise
        return action

    def get_action(self, state, explore_noise_scale, greedy=False):
        action = self.call([state])
        action = self.action_range * action.numpy()[0]
        if greedy:
            return action
        # 添加噪声
        normal = tfd.Normal(loc=0.0, scale=1.0)
        noise = normal.sample(action.shape) * explore_noise_scale
        action += noise
        return action.numpy()

    def sample_action(self, ):
        # sample_action() 函数用于在训练开始时产生随机动作
        a = tf.random.uniform([self.num_actions], -1, 1)
        return self.action_range * a.numpy()

    def train(self):
        pass
