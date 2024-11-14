import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

from ReplayBuffer import ReplayBuffer
from TD3 import TD3
import gym

RANDOM_SEED = 42  # 可以根据需求设置为其他整数值
ENV_ID = "Pendulum-v1"  # 替换为实际使用的 Gym 环境 ID
HIDDEN_DIM = 256  # 可根据需要调整隐藏层大小
replay_buffer = ReplayBuffer(capacity=100000)  # 设置缓存的大小
POLICY_TARGET_UPDATE_INTERVAL = 2  # 每隔2次更新目标网络
Q_LR = 3e-4  # Q 网络的学习率
POLICY_LR = 3e-4  # 策略网络的学习率
TRAIN_EPISODES = 1000  # 可根据需求设置训练轮次数
MAX_STEPS = 200  # 每个 episode 中的最大步数
RENDER = False  # 或 False，是否启用渲染
EXPLORE_STEPS = 10000  # 初始随机探索步数
EXPLORE_NOISE_SCALE = 0.1  # 探索噪声的尺度
BATCH_SIZE = 64  # 每次更新时使用的样本数
UPDATE_ITR = 10  # 每次更新的迭代次数
EVAL_NOISE_SCALE = 0.2  # 评估噪声的尺度
REWARD_SCALE = 1.0  # 奖励缩放因子

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="Enable training mode")
args = parser.parse_args()

# 如下是主要训练代码。这里先创建环境和智能体
# 初始化环境
#env = gym.make(ENV_ID).unwrapped
env = gym.make(ENV_ID, render_mode="human").unwrapped

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

action_range = env.action_space.high # 缩放动作 [-action_range, action_range]
# 设置随机种子,以便复现效果
env.reset(seed=RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
# 初始化回放缓存
# replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
# 初始化智能体
#agent = TD3(state_dim, action_dim, action_range, HIDDEN_DIM, replay_buffer,  POLICY_TARGET_UPDATE_INTERVAL, Q_LR, POLICY_LR)
agent = TD3(state_dim, action_dim, action_range=1.0, hidden_dim=HIDDEN_DIM, replay_buffer=replay_buffer, policy_target_update_interval=POLICY_TARGET_UPDATE_INTERVAL, q_lr=Q_LR, policy_lr=POLICY_LR)

t0 = time.time()

'''
在开始片段之前,需要做一些初始化操作。
这里训练时间受总运行步数的限制,而不是最大片段迭代数。
由于网络建立的方式不同,这种方式需要在使用前额外调用一次函数。
'''

args.train = True

# 训练循环
if args.train:
    frame_idx = 0
    all_episode_reward = []
    # 这里需要进行一次额外的调用,以使内部函数进行一些初始化操作,让其可以正常使用
    # model.forward 函数
    # state = env.reset()[0].astype(np.float32)
    state = env.reset()
    if isinstance(state, tuple):  # 检查是否为 tuple
        state = state[0]
    state = tf.convert_to_tensor(state, dtype=tf.float32)
    state = tf.expand_dims(state, axis=0)  # 添加批次维度

    agent.policy_net(state)

    agent.target_policy_net(state)

    '''
    训练刚开始的时候,会先由智能体进行随机采样。通过这种方式可以采集到足够多的用于更新的数据。
    在那之后,智能体还是和往常一样与环境进行交互并采集数据,再进行存储和更新。
    '''
    for episode in range(TRAIN_EPISODES):
        state = env.reset()[0].astype(np.float32)
        episode_reward = 0
        for step in range(MAX_STEPS):
            if RENDER:
                env.render()
            if frame_idx > EXPLORE_STEPS:
                action = agent.policy_net.get_action(state, EXPLORE_NOISE_SCALE)
            else:
                action = agent.policy_net.sample_action()

            #next_state, reward, done, _ = env.step(action)
            # 修改为：
            next_state, reward, done, _ = env.step(action)[:4]  # 只获取前四个值

            next_state = next_state.astype(np.float32)
            done = 1 if done is True else 0
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            frame_idx += 1
            if len(replay_buffer) > BATCH_SIZE:
                for i in range(UPDATE_ITR):
                    agent.update(BATCH_SIZE, EVAL_NOISE_SCALE, REWARD_SCALE)
            if done:
                break
        '''
        最终,我们提供了一些可视化训练过程所需的函数,并将训练的模型进行存储。
        '''
        if episode == 0:
            all_episode_reward.append(episode_reward)
        else:
            all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
        print('Training | Episode: {}/{} | Episode Reward: {:.4f} | Running Time:  {:.4f}'.format(episode+1, TRAIN_EPISODES, episode_reward, time.time() - t0) )
    agent.save()
    plt.plot(all_episode_reward)
    if not os.path.exists('image'):
        os.makedirs('image')
    plt.savefig(os.path.join('image', 'td3.png'))

print("Training mode:", args.train)
