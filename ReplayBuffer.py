# 导入必要库
import numpy as np
import random
from tensorflow_probability import distributions as tfd

# ReplayBuffer 类用来建立一个回放缓存,它的主要函数是 push() 和 sample() 函数。

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # 堆叠各元素
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

