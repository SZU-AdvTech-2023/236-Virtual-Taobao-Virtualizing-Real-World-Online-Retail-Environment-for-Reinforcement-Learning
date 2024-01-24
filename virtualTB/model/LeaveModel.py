import os
import numpy as np
from virtualTB.utils import *

# 搜索引擎模型：根据用户模型的特征（状态、动作和奖励）返回Page页
class LeaveModel(nn.Module):
    def __init__(self, n_input = 88, n_output = 101, learning_rate = 0.01):
        super(LeaveModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_output)
        )
        self.model.apply(init_weight)
        
    def predict(self, user):
        x = self.model(user)
        page = torch.multinomial(F.softmax(x, dim = 1), 1)
        return page

    def load(self, path = None):
        if path == None:
            g_path = os.path.dirname(__file__) + '/../data/leave_model.pt'
        self.model.load_state_dict(torch.load(g_path))
