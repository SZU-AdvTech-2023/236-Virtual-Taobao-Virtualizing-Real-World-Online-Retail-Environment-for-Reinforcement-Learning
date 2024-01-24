from gym.envs.registration import register

# 将自定义环境注册到Gym中
register(
    id='VirtualTB-v0',  # 标识符
    entry_point='virtualTB.envs:VirtualTB',  # 指定虚拟环境的入口点
)