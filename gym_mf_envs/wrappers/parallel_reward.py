import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gym_mf_envs.wrappers.utils import merge_dict

class ParallelRewardWrapper(gym.RewardWrapper):
    """
    A Gymnasium RewardWrapper designed for parallelized (vectorized) environments.
    This wrapper merges the rewards returned from each parallel environment instance into
    a single dictionary using a custom utility function `merge_dict`.
    """
    def __init__(self, env: AsyncVectorEnv):
        """
        Initialize the ParallelRewardWrapper.
        """
        super(ParallelRewardWrapper, self).__init__(env)
        self.env = env

    def reward(self, reward):
        """
        Merge the rewards from all parallel environments into a unified dictionary.
        """
        return merge_dict(reward)
