import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torch
import torch.nn as nn
from virtualTB.model.ActionModel import ActionModel
from virtualTB.model.LeaveModel import LeaveModel
from virtualTB.model.UserModel import UserModel  # 直接导入对应模块中的对象而不是导入模块
from virtualTB.utils import *


# 继承 gym.env 表示一个强化学习环境：它主要包含reset()\step()\render()三个方法
class VirtualTB(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        构造函数：设置环境的一些属性和参数：状态空间、动作空间和一些模型的初始化加载
        """
        self.lst_action = None
        self.cur_user = None
        self.rend_action = None
        self.n_item = 5  # 表示项目的数量，初始化为5
        self.n_user_feature = 88  # 用户特征数量初始化为 88
        self.n_item_feature = 27  # 项目特征数量，初始化为27
        self.max_c = 100  # 最大点击次数，初始化为100
        self.obs_low = np.concatenate(([0] * self.n_user_feature, [0, 0, 0]))
        self.obs_high = np.concatenate(([1] * self.n_user_feature, [29, 9, 100]))
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.int32)  # 状态空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_item_feature,), dtype=np.float32)  # 动作空间
        """
        模型对象实例化：初始化用户模型、动作模型和leave模型
        """
        self.user_model = UserModel()
        self.user_model.load()
        self.user_action_model = ActionModel()
        self.user_action_model.load()
        self.user_leave_model = LeaveModel()
        self.user_leave_model.load()
        self.reset()

    def seed(self, sd=0):
        """
        用于设置随机数生成器的种子，以确保实验的可重复性。
        :param sd:
        :return:
        """
        torch.manual_seed(sd)

    @property
    def state(self):
        """
        环境状态的返回：用户是虚拟淘宝的环境，因此环境特征就是用户特征
        :return: 当前的环境状态:是当前用户特征、上一次行为和累计点击次数的拼接
        """
        return np.concatenate((self.cur_user, self.lst_action, np.array([self.total_c])), axis=-1)

    def __user_generator(self):
        """
        用户生成器：用于生成用户特征，首先使用用户模型（生成器）生成用户数据（经过处理），然后使用leave模型判断用户是否应该离开。
        :return: 返回用户模型（虚拟环境的构建）
        """
        # with shape(n_user_feature,)
        user = self.user_model.generate()  # 得到一个用户特征
        self.__leave = self.user_leave_model.predict(user)  # 判断是否会离开（结束）
        return user

    def step(self, action):
        """
        执行一个动作后下一步模型的变化：返回环境下一个状态、奖励和结束标志
        :param action:
        :return: 执行动作后返回下一个状态、奖励和是否终止
        """
        # Action: tensor with shape (27, )
        self.lst_action = self.user_action_model.predict(FLOAT(self.cur_user).unsqueeze(0), FLOAT([[self.total_c]]),
                                                         FLOAT(action).unsqueeze(0)).detach().numpy()[0]
        reward = int(self.lst_action[0])
        self.total_a += reward
        self.total_c += 1
        self.rend_action = deepcopy(self.lst_action)
        done = (self.total_c >= self.__leave)
        if done:
            self.cur_user = self.__user_generator().squeeze().detach().numpy()
            self.lst_action = FLOAT([0, 0])
        return self.state, reward, done, {'CTR': self.total_a / self.total_c / 10}

    def reset(self):
        """
        重置环境状态：用于在每个回合的开始调用
        :return: 返回环境的状态
        """
        self.total_a = 0
        self.total_c = 0
        self.cur_user = self.__user_generator().squeeze().detach().numpy()
        self.lst_action = FLOAT([0, 0])
        self.rend_action = deepcopy(self.lst_action)
        return self.state

    def render(self, mode='human', close=False):
        """
        渲染：在终端上显示当前环境状态和用户的行为信息。
        :param mode:
        :param close:
        :return:
        """
        print('Current State:')
        print('\t', self.state)
        a, b = np.clip(self.rend_action, a_min=0, a_max=None)
        print('User\'s action:')
        print('\tclick:%2d, leave:%s, index:%2d' % (
            int(a), 'True' if self.total_c > self.max_c else 'False', int(self.total_c)))
        print('Total clicks:', self.total_a)
