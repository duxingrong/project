from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env import VecFrameStack
import uuid
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import  os
import gym




class SkipFrame(gym.Wrapper):
    """
    SkipFrame 是可以实现跳帧操作.因为连续的帧变化不大,我们可以跳过n个中间帧而不会丢失太多信息
    第n帧聚合每个跳过帧上累计的奖励
    """

    def __init__(self, env,skip):
        super().__init__(env)
        self._skip = skip

    def step(self,action):
        """重复action , 然后累计奖励.在连续几帧里重复同一个动作不会对状态造成太大影响"""
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info



def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')  # 官网可以看出,v2足够完成训练
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env,keep_dim=True)
    env = ResizeObservation(env,shape=(84,84))
    monitor_dir = r'monitor_log'  # 定义监控日志目录
    env = Monitor(env, filename=os.path.join(monitor_dir, str(uuid.uuid4())))  # 创建 Monitor 包装器,为每次运行生成一个唯一的目录名称，确保不同实验的日志和视频不会覆盖。
    return env

def linear_schedule(initial_value, end_value):
    """
    线性递减函数，从 initial_value 递减到 end_value

    参数:
        initial_value: 初始值
        end_value: 结束值

    返回:
        一个函数，接受当前进度比例 (从 1.0 到 0.0) 作为参数，返回当前值
    """

    def func(progress_remaining: float) -> float:
        """
        参数:
            progress_remaining: 剩余进度比例 (从 1.0 开始递减到 0.0)
        """
        fraction = 1.0 - progress_remaining  # 转换为从 0.0 到 1.0
        return initial_value + fraction * (end_value - initial_value)

    return func



"""
大多数图像环境（如 gym）默认返回 HWC 格式的观测（Height, Width, Channels，例如 (84, 84, 4)）。

但 PyTorch 的卷积神经网络（CNN）默认要求输入是 CHW 格式（Channels, Height, Width，例如 (4, 84, 84)）。

VecTransposeImage 会自动将 HWC 转换为 CHW，例如将形状从 (84, 84, 4) 转换为 (4, 84, 84)。
"""

def main():
    # 变量
    total_timesteps = 1e7
    num_envs = 4 # 进程数量
    model_params = {
        #---核心训练参数---#
        'learning_rate': linear_schedule(3e-4,1e-5),  # 学习率
        'batch_size': 512,  # 随机抽取多少数据
        'n_epochs': 8,  # 更新次数
        'gamma': 0.95,  # 短视或者长远
        'gae_lambda': 0.92,  # 平衡优势估计的偏差和方差


        #---探索和稳定性---#
        'ent_coef': 0.1,  # 更高的熵系数鼓励早期探索
        'clip_range': linear_schedule(0.25,0.1),  # 截断范围
        "target_kl": None,  # 设置KL散度早停阈值
        'n_steps': 512,  # 每个环境每次更新的步数
        "max_grad_norm": 0.5,  # 更严格的梯度裁剪

        #---网络和优化---#
        "vf_coef": 0.75,  # 加强价值函数学习(应对系数奖励)

        'device': 'cuda',

        # log
        'verbose': 1,
        'policy': "CnnPolicy"
    }

    # 环境
    monitor_dir = r'monitor_log'
    os.makedirs(monitor_dir, exist_ok=True)
    env = SubprocVecEnv([lambda:make_env() for _ in range(num_envs)]) #多进程
    env = VecFrameStack(env,4,channels_order='last') #帧堆叠包装器,一个action动作上堆叠几帧
    env = VecTransposeImage(env)  # 显式添加，确保训练环境结构明确

    # 创建评估环境（需与训练环境结构一致）
    eval_env = SubprocVecEnv([lambda: make_env() for _ in range(1)])  # 单进程评估环境
    eval_env = VecFrameStack(eval_env, 4, channels_order='last')
    eval_env = VecTransposeImage(eval_env)  # 显式添加同样的包装

    eval_callback = EvalCallback(
        eval_env,  # 使用显式包装后的评估环境
        best_model_save_path="./base_model/",
        log_path="./call_back_logs/",
        eval_freq=10000 // num_envs
    )

    # 模型
    # model=PPO.load('best_model.zip', env=env, **model_params)
    model = PPO(env=env, **model_params, tensorboard_log="./logs")

    # 训练
    model.learn(total_timesteps=total_timesteps,callback=eval_callback)

if __name__ == '__main__':
    main()
