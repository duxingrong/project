from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
from train import make_env

def main():
    # 路径
    model_dir= r'base_model/best_model.zip'
    # 初始化环境
    env = SubprocVecEnv([make_env for _ in range(1)])
    env = VecFrameStack(env, 4, channels_order='last')  # 帧叠加
    model = PPO.load(model_dir, env=env)
    # 开始玩游戏
    obs = env.reset()
    ep_len = 10000
    for i in range(ep_len):
        obs = obs.copy()  # 复制数组以避免负步长问题
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print('reward:',reward)
        env.render('human')  # 显示游戏画面

        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()












