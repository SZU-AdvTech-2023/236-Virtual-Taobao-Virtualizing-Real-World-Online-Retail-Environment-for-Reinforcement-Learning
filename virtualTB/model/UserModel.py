import os
import numpy as np
from virtualTB.utils import *


# 用户模型
class UserModel(nn.Module):
    """
    用户模型的构建：用户在虚拟淘宝中就是环境模型。
    该模型继承了nn.Module：PyTorch中构建的神经网络模型的基类
    """

    def __init__(self, instance_dimesion=88, seed_dimesion=128, n_hidden=128, learning_rate=0.001):
        """
        构造函数：初始化用户模型的参数和神经网络结构
        :param instance_dimesion:用户特征的维度，初始为88
        :param seed_dimesion:种子特征的维度，初始为128。生成器的随机输入
        :param n_hidden:隐藏层维度，初始为128
        :param learning_rate:学习率，初始化为0.001
        """
        super(UserModel, self).__init__()
        self.seed_dimesion = seed_dimesion
        """
        构建生成式模型：指定其神经网络和权重函数
        该模型主要包括输入层、输出层和隐藏层三层。其中隐藏层由两个线性层构成
        """
        self.generator_model = nn.Sequential(
            nn.Linear(seed_dimesion, n_hidden),
            nn.LeakyReLU(),  # 激活函数
            nn.Linear(n_hidden, instance_dimesion),
        )
        self.apply = self.generator_model.apply(init_weight)

    def generator(self, z):
        """
        生成用户特征：接收种子特征然后传递到 generator_model 利用生成器模型来生成用户特征
        :param z:
        :return:
        """
        x = self.generator_model(z)
        return self.softmax_feature(x)

    def softmax_feature(self, x):
        """
        用户特征处理函数（正规化处理）：将生成的用户特征进行分段，并对每个分段进行 softmax 操作，然后将这些分段结果进行拼接，返回最终的 softmax 特征。
        借助 softmax 操作，我们可以将结果概率化，各元素之和为 1 .如：Softmax输出的结果是(90%,5%,3%,2%)
        :param x:
        :return:
        """
        features = [None] * 11
        features[0] = x[:, 0:8]
        features[1] = x[:, 8:16]
        features[2] = x[:, 16:27]
        features[3] = x[:, 27:38]
        features[4] = x[:, 38:49]
        features[5] = x[:, 49:60]
        features[6] = x[:, 60:62]
        features[7] = x[:, 62:64]
        features[8] = x[:, 64:67]
        features[9] = x[:, 67:85]
        features[10] = x[:, 85:88]
        entropy = 0
        softmax_feature = FLOAT([])
        for i in range(11):
            softmax_feature = torch.cat((softmax_feature, F.softmax(features[i], dim=1)), dim=-1)
            entropy += -(F.log_softmax(features[i], dim=1) * F.softmax(features[i], dim=1)).sum(dim=1).mean()
        return softmax_feature, entropy

    def generate(self, z=None):
        """
        获取用户特征的独热编码：将分类数据转换为机器学习模型可以理解的格式，通常用于处理分类特征
        """
        if z == None:
            z = torch.rand((1, self.seed_dimesion))
        x = self.generator(z)[0]  # 得到生成的用户特征
        # 将特征进行切割成11个子特征
        features = [None] * 11
        features[0] = x[:, 0:8]
        features[1] = x[:, 8:16]
        features[2] = x[:, 16:27]
        features[3] = x[:, 27:38]
        features[4] = x[:, 38:49]
        features[5] = x[:, 49:60]
        features[6] = x[:, 60:62]
        features[7] = x[:, 62:64]
        features[8] = x[:, 64:67]
        features[9] = x[:, 67:85]
        features[10] = x[:, 85:88]
        one_hot = FLOAT()
        for i in range(11):  # 遍历11个子特征
            tmp = torch.zeros(features[i].shape)
            # 从子特征中随机提取一个值
            one_hot = torch.cat((one_hot, tmp.scatter_(1, torch.multinomial(features[i], 1), 1)), dim=-1)
        return one_hot

    def load(self, path=None):
        if path == None:
            g_path = os.path.dirname(__file__) + '/../data/generator_model.pt'
            print(g_path)
        try:
            self.generator_model.load_state_dict(torch.load(g_path))
            print("Model loaded successfully!")
        except Exception as e:
            print("Error loading model:", e)
