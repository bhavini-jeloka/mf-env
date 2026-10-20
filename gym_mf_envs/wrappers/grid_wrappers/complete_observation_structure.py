import gymnasium as gym
from gym_mf_envs.envs.grid_envs.envs.base_grid_env import BaseMFEnv


class CompleteObservationStructureWrapper(gym.ObservationWrapper):
    """
    Observation wrapper that restructures the observation space to provide each agent with:
    - Its own local observation
    - Its team's shared observation
    - The global observation

    This is useful for mean-field setups where agents may require access to hierarchical or 
    fully observable states during training or evaluation.
    """
    def __init__(self, env: BaseMFEnv):
        """
        Initializes the observation wrapper and constructs the new observation space structure.
        """
        super(CompleteObservationStructureWrapper, self).__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Dict({
            pop_id: gym.spaces.Dict(
                {
                    "agent_{}".format(agent_index): gym.spaces.Dict(
                        {
                            "local-obs": self.env.observation_space[pop_id]["agent_{}-local-obs".format(agent_index)],
                            "shared-team-obs": self.env.observation_space[pop_id]["shared-team-obs"],
                            "global-obs": self.env.observation_space["global-obs"],
                        }
                    )
                    for agent_index in range(self.env.get_n_agents(pop_id))
                }
            )
            for pop_id in self.env.observation_space.keys()
        })

    def observation(self, obs):
        """
        Constructs the new observation dictionary from the original observation.

        Each agent receives a dictionary containing:
        - "local-obs": the agent's local observation
        - "shared-team-obs": the team-level shared observation (same across team)
        - "global-obs": the global environment state
        """
        obs_ = {
            population_id: {
                "agent_{}".format(agent_index): {
                    "local-obs": obs[population_id]["agent_{}".format(agent_index)]["local-obs"],
                    "shared-team-obs": obs[population_id]["agent_0"]["shared-team-obs"],
                    "global-obs": obs["global-obs"]
                }
                for agent_index in range(self.env.get_n_agents(population_id))
            }
            for population_id in obs.keys()
        }
        return obs_
