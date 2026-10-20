import gymnasium as gym
from gym_mf_envs.envs.grid_envs.envs.base_grid_env import BaseMFEnv
from gym_mf_envs.wrappers.utils import flatten_dict_box_obs_space as flatten_obs_space
from gym_mf_envs.wrappers.utils import flatten_dict_box_obs as flatten_obs


class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    This wrapper is useful for algorithms or agents that require flat (vectorized)
    observations instead of nested dictionary structures.
    """
    def __init__(self, env: BaseMFEnv):
        """
        Initialize the FlattenObservationWrapper.
        """
        super(FlattenObservationWrapper, self).__init__(env)
        self.env = env

        self.observation_space = gym.spaces.Dict(
            {
                population_id: gym.spaces.Dict(
                    {
                        "agent_{}-local-obs".format(idx): flatten_obs_space(
                            self.env.observation_space[population_id]["agent_{}-local-obs".format(idx)])
                        for idx in range(self.env.get_n_agents(population_id))
                    } |
                    {
                        "shared-team-obs": flatten_obs_space(
                            self.env.observation_space[population_id]["shared-team-obs"])
                    }

                )
                for population_id in self.env.get_population_ids()
            } |
            {
                "global-obs": flatten_obs_space(self.env.observation_space["global-obs"])
            })

    def observation(self, obs):
        """
        Flatten the observation dictionary into vector-based observations.
        """
        obs_ = {
                   population_id:

                       {
                           "agent_{}-local-obs".format(idx): flatten_obs(
                               obs[population_id]["agent_{}-local-obs".format(idx)])
                           for idx in range(self.env.get_n_agents(population_id))
                       } |
                       {
                           "shared-team-obs": flatten_obs(obs[population_id]["shared-team-obs"])
                       }

                   for population_id in self.env.get_population_ids()
               } | {
                   "global-obs": flatten_obs(obs["global-obs"])
               }
        return obs_
