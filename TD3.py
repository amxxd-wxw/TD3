import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow_probability import distributions as tfd
import numpy as np
from PolicyNetwork import PolicyNetwork
from QNetwork import QNetwork

#接下来介绍 TD3 类,它是本例子的核心内容
class TD3():
    # 创建回放缓存和网络
    def __init__(self, state_dim, action_dim, replay_buffer, hidden_dim, action_range,
                 policy_target_update_interval=1, q_lr=3e-4, policy_lr=3e-4):
        self.replay_buffer = replay_buffer  # 初始化所有网络
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        #self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.policy_net = PolicyNetwork(num_inputs=state_dim, num_actions=action_dim, hidden_dim=hidden_dim,
                                        action_range=action_range)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim,  action_range)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)
        # 初始化目标网络参数
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = tf.optimizers.Adam(q_lr)
        self.q_optimizer2 = tf.optimizers.Adam(q_lr)
        self.policy_optimizer = tf.optimizers.Adam(policy_lr)

    # target_ini() 函数和 target_soft_update() 函数都用来更新目标网络。
    # 不同之处在于前者是通过硬拷贝直接替换参数,而后者是通过 Polyak 平均进行软更新
    def target_ini(self, net, target_net):
        for target_param, param in zip(target_net.trainable_weights,  net.trainable_weights):
            target_param.assign(param)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        for target_param, param in zip(target_net.trainable_weights,  net.trainable_weights):
             target_param.assign(target_param * (1.0 - soft_tau) + param * soft_tau)  # 软更新
        return target_net

    #接下来将介绍关键的 update() 函数。这部分充分体现了 TD3 算法的 3 个关键技术。  在函数的开始部分,我们先从回放缓存中采样数据。
    def update(self, batch_size, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2):  # 更新 TD3 中的所有网 络
        self.update_cnt += 1  # 采样数据
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        reward = reward[:, np.newaxis]  # 扩展维度
        done = done[:, np.newaxis]
        '''
        # 接下来,我们通过给目标动作增加噪声实现了目标策略平滑技术。通过这样跟随动作的变化,
        # 对 Q 值进行平滑,可以使得策略更难利用 Q 函数的拟合差错。这是 TD3 算法中的第三个技术。
        # 技术三: 目标策略平滑。通过给目标动作增加噪声来实现
        '''
        new_next_action = self.target_policy_net.evaluate_action(next_state,
                                    eval_noise_scale=eval_noise_scale ) # 添加了截断的正态噪声
        # 通过批数据的均值和标准差进行标准化
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / np.std(reward,  axis=0)
        '''
        下一个技术是截断的 Double-Q Learning。
        它将同时学习两个 Q 值函数,并且选择较小的 Q 值 来作为贝尔曼误差损失函数中的目标 Q 值。
        通过这种方法可以减轻 Q 值的过估计。这也是 TD3 算法中的第一个技术
        '''
        # 训练 Q 函数
        target_q_input = tf.concat([next_state, new_next_action], 1) # 0 维是样本数量
        # 技术一: 截断的 Double-Q Learning。这里使用了更小的 Q 值作为目标 Q 值
        target_q_min = tf.minimum(self.target_q_net1(target_q_input), self.target_q_net2(target_q_input))
        target_q_value = reward + (1 - done) * gamma * target_q_min # 如果 done==1,则只有  # reward 值
        q_input = tf.concat([state, action], 1) # 处理 Q 网络的输入

        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.square(predicted_q_value1 - target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.q_net1.trainable_weights)
        self.q_optimizer1.apply_gradients(zip(q1_grad, self.q_net1.trainable_weights))

        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.square(predicted_q_value2 - target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.q_net2.trainable_weights)
        self.q_optimizer2.apply_gradients(zip(q2_grad, self.q_net2.trainable_weights))
        '''
        最后一个技术是延迟策略更新技术。这里的策略网络及其目标网络的更新频率比 Q 值网络的更新频率更小。
        论文 (Fujimoto et al., 2018) 中建议每 2 次 Q 值函数更新时进行 1 次策略更新。  
        这也是 TD3 算法中提到的第二个技术。
        '''
        # 训练策略函数
        # 技术二: 延迟策略更新。减少策略更新的频率
        if self.update_cnt:
            with tf.GradientTape() as p_tape:
                new_action = self.policy_net.evaluate_action(  state, eval_noise_scale=0.0 ) # 无噪声,确定性策略梯度
                new_q_input = tf.concat([state, new_action], 1)
                # 实现方法一:
                # predicted_new_q_value = tf.minimum(self.q_net1(new_q_input),self.q_net2(new_q_input))
                # 实现方法二:
                predicted_new_q_value = self.q_net1(new_q_input)
                policy_loss = -tf.reduce_mean(predicted_new_q_value)
            p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
            self.policy_optimizer.apply_gradients(zip(p_grad,  self.policy_net.trainable_weights))
            # 软更新目标网络
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1,  soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2,  soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net,  self.target_policy_net, soft_tau)
