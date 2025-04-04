# 继承gym.wrapper,实现跳帧的包装器,重写step
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

