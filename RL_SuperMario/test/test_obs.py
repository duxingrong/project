# 主要进行环境的预处理
# 1. 彩色->灰度
# 2. 裁剪图像大小
# 3. reward变换,这里老哥已经帮我们做好了,所以不变
# 4. 跳帧

import uuid
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import  matplotlib.pyplot  as plt
import  os
from .my_wrapper import SkipFrame


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')  # 官网可以看出,v2足够完成训练
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env,keep_dim=True)
    env = ResizeObservation(env,shape=(84,84))
    monitor_dir = r'../monitor_log/'  # 定义监控日志目录
    env = Monitor(env, filename=os.path.join(monitor_dir, str(uuid.uuid4())))  # 创建 Monitor 包装器,为每次运行生成一个唯一的目录名称，确保不同实验的日志和视频不会覆盖。
    return env

if __name__ == '__main__':
    env = make_env()
    done = True
    for step in range(4):  # step理解为帧
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())  # 使用给定的动作迈出一步
        plt.figure(figsize=(4, 4))
        plt.imshow(state.squeeze(), cmap="gray")
        plt.title(f"步骤 {step+1}")
        plt.show()
        print(f"已显示图像 {step+1}/4")

    env.close()
