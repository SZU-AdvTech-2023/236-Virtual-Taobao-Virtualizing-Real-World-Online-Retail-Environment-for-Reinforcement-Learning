import csv

import gym
import math
import torch
import random
import virtualTB
import time, sys
import configparser
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym import wrappers
from ddpg import DDPG
from copy import deepcopy
from collections import namedtuple

FLOAT = torch.FloatTensor
LONG = torch.LongTensor

# 经验：环境状态转移与反馈的单元
Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))  # 转换函数：状态、动作、掩码、下一个状态和奖励


# 经验回放类：包含经验存放、经验抽取和样本数量
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# 噪声模型：用于添加探索性的噪声到智能体的动作中
class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
    # 重置噪声
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    # 生成噪声
    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


env = gym.make('VirtualTB-v0')  # 创建一个自定义gym环境（该环境在VirtualTaobao包__init__.py中注册好了）。

# 确保各模型的随机数是可重复的
env.seed(0)  # 设置环境的随机种子
np.random.seed(0)  # 设置 NumPy 的随机种子
torch.manual_seed(0)  # 设置 PyTorch 的随机种子

agent = DDPG(gamma=0.95, tau=0.001, hidden_size=128,
             num_inputs=env.observation_space.shape[0], action_space=env.action_space)

memory = ReplayMemory(1000000)  # 经验回放缓冲区：记住历史状态作为以后的经验样本

ounoise = OUNoise(env.action_space.shape[0])
param_noise = None

rewards = []
total_numsteps = 0
updates = 0

# 训练次数：指定智能体训练100000次
for i_episode in range(100000):
    """
    训练过程如下：
    1.初始化操作：重置环境的状态、设置奖励为 0
    2.开始训练
        a.智能体选择动作
        b.环境根据动作输出下一个状态、奖励和反馈（存入”历史经验“）
        c.智能体统计操作次数以及累计奖励
        d.
    """

    # 重置环境状态，初始化累积奖励
    state = torch.Tensor([env.reset()])
    episode_reward = 0

    # 开始训练
    while True:
        # 智能体选择动作
        action = agent.select_action(state, ounoise, param_noise)

        # 与环境互动，获取下一个状态、奖励和结束标志
        next_state, reward, done, _ = env.step(action.numpy()[0])
        total_numsteps += 1
        episode_reward += reward

        # 将动作、掩码、下一个状态和奖励存储到经验回放内存中
        action = torch.Tensor(action)
        mask = torch.Tensor([not done])
        next_state = torch.Tensor([next_state])
        reward = torch.Tensor([reward])

        memory.push(state, action, mask, next_state, reward)

        state = next_state

        # 如果经验回放内存中有足够的数据，执行参数更新
        if len(memory) > 128:
            for _ in range(5):
                transitions = memory.sample(128)  # 在每次循环中，从经验回放内存中随机抽样128个经验样本。
                batch = Transition(*zip(*transitions))  # 抽样的经验样本被分组成一个批次（batch），包括状态、动作、掩码、下一个状态和奖励

                value_loss, policy_loss = agent.update_parameters(batch)  # 使用抽样的批次数据，调用agent（DDPG智能体）的update_parameters方法，以更新智能体的参数
                # 得到的 value_loss,policy_loss 都在update_parameters()方法中被反向传播
                updates += 1
        if done:
            break

    rewards.append(episode_reward)

    # 每隔10个训练周期进行性能评估
    if i_episode % 10 == 0:
        episode_reward = 0
        episode_step = 0

        # 在一个小的评估周期内测试智能体的性能
        for i in range(50):
            state = torch.Tensor([env.reset()])
            while True:
                action = agent.select_action(state)

                next_state, reward, done, info = env.step(action.numpy()[0])
                episode_reward += reward
                episode_step += 1

                next_state = torch.Tensor([next_state])

                state = next_state
                if done:
                    break

        # 输出评估结果：训练周期编号、每个评估周期执行的步数（操作次数）、平均奖励、点击通过率
        print("test result:Episode: {}, total numsteps: {}, average reward: {}, CTR: {}".format(i_episode, episode_step,
                                                                                    episode_reward / 50,
                                                                                    episode_reward / episode_step / 10))
        # 将数据保存到csv文件中
        epoch = int(i_episode/10)
        ctr = episode_reward / episode_step / 10
        with open('RL_output.csv', 'a', newline='') as csvfile:
            # 定义CSV写入器
            csv_writer = csv.writer(csvfile)

            # 如果是第一次写入，先写入表头
            if csvfile.tell() == 0:
                csv_writer.writerow(['Epoch', 'CTR'])

            # 写入数据行
            csv_writer.writerow(["{:3d}".format(epoch),  "{:.2f}".format(ctr)])
# 关闭虚拟环境
env.close()
