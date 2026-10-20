import gymnasium as gym
from gymnasium.core import WrapperActType, ActType
from gymnasium.spaces import MultiDiscrete, Discrete
from gym_mf_envs.envs.grid_envs.envs.base_grid_env import BaseMFEnv


class MultiDiscreteActionWrapper(gym.ActionWrapper):
    """
    This wrapper enables the environment to accept a flat MultiDiscrete action array where each agent's discrete action
    is concatenated into a single list. Internally, it maps this list back to the per-agent dictionary structure
    expected by the base environment.
    """
    def __init__(self, env: BaseMFEnv):
        """
        Initializes the wrapper by constructing the MultiDiscrete action space from the original per-agent Discrete spaces.
        """
        super(MultiDiscreteActionWrapper, self).__init__(env)
        self.env = env
        base_env = self.env.unwrapped
        action_space_list = [self.env.action_space[pop_id] for pop_id in self.env.action_space.keys()]
        assert all([isinstance(act_space, Discrete) for act_space in action_space_list])
        n_actions_list = [act_space.n for pop_id, act_space in self.env.action_space.items()
                          for _ in range(base_env.get_n_agents(pop_id))]
        self.action_space = MultiDiscrete(n_actions_list)

    def action(self, action: WrapperActType) -> ActType:
        """
        Converts a flat list of agent actions into the nested dictionary format required by the base environment.
        """
        action = list(action)
        base_env = self.env.unwrapped
        action_dict = {population_id: {"agent_{}".format(agent_id): action.pop(0) for agent_id in
                                       range(base_env.get_n_agents(population_id))}
                       for population_id in self.env.action_space.keys()}
        return action_dict
