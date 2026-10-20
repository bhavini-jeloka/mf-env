import gymnasium as gym
from gym_mf_envs.envs.grid_envs.envs.base_grid_env import BaseMFEnv


class PopulationRewardWrapper(gym.RewardWrapper):
    """
    A Gymnasium RewardWrapper that simplifies the reward structure from agent-level to population-level.
    This wrapper assumes that all agents in a population receive the same reward, and thus extracts the
    reward of a representative agent (agent_0) per population to use as the population's reward.
    """
    def __init__(self, env: BaseMFEnv):
        """
        Initialize the PopulationRewardWrapper.
        """
        super(PopulationRewardWrapper, self).__init__(env)
        self.env = env

    def reward(self, reward):
        """
        Transforms agent-level rewards into population-level rewards by taking the reward of agent_0
        from each population.
        """
        reward_ = {
            population_id: reward[population_id]["agent_0"]
            for population_id in self.env.get_population_ids()
        }
        return reward_
