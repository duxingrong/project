from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


# 得到环境
env = gym_super_mario_bros.make('SuperMarioBros-v2') #官网可以看出,v2足够完成训练
print(type(env))# gym.wrappers
env = JoypadSpace(env, SIMPLE_MOVEMENT)
print(type(env))# nes_py.wrappers 也是一个包装器

done = True
for step in range(5000): # step理解为帧
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample()) #使用给定的动作迈出一步
    env.render() #渲染作用

env.close()