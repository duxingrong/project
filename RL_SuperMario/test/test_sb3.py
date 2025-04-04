import gym
from stable_baselines3 import A2C

# 创建环境
env = gym.make("CartPole-v1")
# 导入模型，策略：ActorCriticPolicy
# Critic:评估哪个动作是重要的，value_loss降低，代表评估越来越接近真实环境
# Actor: policy_loss ，用来衡量当前策略（Actor）在选择动作时与“最优”行为之间的差距
model = A2C("MlpPolicy", env, verbose=2,tensorboard_log="../logs")
# 训练
model.learn(total_timesteps=1e7)

# observation 观察空间==环境
obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True) #deterministic:确定性，是一定执行概率最大的动作
    obs, reward, done, info = env.step(action)
    env.render()

    if done: # done为true代表游戏结束了，重新开始
      obs = env.reset()