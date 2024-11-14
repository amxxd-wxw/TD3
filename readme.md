# TD3 强化学习算法实现

本项目实现了双延迟深度确定性策略梯度（TD3）算法，这是一种用于连续动作空间的无模型、异策略的 Actor-Critic 算法。此实现基于 Python，使用 TensorFlow 和 OpenAI 的 Gym 库。

## 项目文件结构

- **main.py**：训练和运行 TD3 智能体的主脚本。
- **PolicyNetwork.py**：定义策略网络（Policy Network），用于 TD3 算法中的动作选择。
- **QNetwork.py**：定义 Q 网络，用于估算期望奖励。
- **ReplayBuffer.py**：实现回放缓存，存储经验元组 `(state, action, reward, next_state, done)`。
- **TD3.py**：TD3 算法的核心文件，包括 actor 和 critic 网络以及训练步骤。
- **test.py**：测试脚本，用于验证 TensorFlow 的安装和环境设置。

## 依赖环境

- Python 3.6 或更高版本
- TensorFlow 2.x
- NumPy
- Matplotlib
- Gym (OpenAI Gym)

## 安装步骤

1. 克隆本仓库：
   ```bash
   git clone <repository-url>
   cd <repository-directory>
