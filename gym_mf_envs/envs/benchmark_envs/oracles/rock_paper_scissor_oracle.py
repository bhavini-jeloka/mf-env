from typing import Union, List
import numpy as np
from numpy import ndarray
from gym_mf_envs.envs.base_env.base_oracle import BaseMFOracle

class RPSBenchmarkOracle(BaseMFOracle):
    """
    Oracle for Rock-Paper-Scissors finite-population mean-field game.
    Computes reward and transition dynamics given policies and mean-fields.
    """
    def __init__(self, num_populations: int, num_agents: dict[str, int], num_states: dict[str, int],
                 num_actions: dict[str, int], **kwargs):
        """Initialize this object and its environment state."""
        self._num_populations = num_populations
        self._num_agents = num_agents
        self._total_num_agents = sum(num_agents.values())
        self._num_states, self._num_actions = num_states, num_actions
        super().__init__(num_populations, num_agents, num_states, num_actions, **kwargs)

        self._reward_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        self._transition_matrix_dict = self._get_transition_matrix()

        # setup population ratio
        if kwargs.get("rho") is not None:
            self._population_ratio = kwargs.get("rho")
        else:
            self.rho = {key: n_agents / self._total_num_agents for key, n_agents in self._num_agents.items()}

    def reset(self, **kwargs):
        """
        Reset the mf oracle with new environment information, e.g. obstacle locations, etc.
        """
        pass

    def update_mf_info(self, agent_info: dict[str, dict[str, dict[str, Union[int, tuple, ndarray]]]],
                       mf_info: dict[str, dict[str, Union[tuple, ndarray]]]):
        """
        Update the mean-field (ED) based on transition matrices and agent states.
        """
        for pop_id in self.population_id_list:
            num_agents = self._num_agents[pop_id]
            mf_population_flat = np.zeros(self._num_states[pop_id], dtype=np.float32)
            for agent_id in range(self._num_agents[pop_id]):
                agent_state = agent_info[pop_id]["agent_{}".format(agent_id)]["current-state"]
                mf_population_flat[agent_state] += 1

            mf_info[pop_id]["mf_flat"] = mf_population_flat / num_agents

    def get_reward(self, agent_info: dict[str, dict[str, dict[str, Union[int, tuple, ndarray]]]],
                   mf_info: dict[str, dict[str, Union[tuple, ndarray]]]) -> dict[str, dict[str, float]]:
        """
        Get the reward for each agent in each population

        Args:
            agent_info[population_id][agent_id][info_id]: agent information with info_id of agent_id in population_id
            mf_info[population_id][info_id]: mean-field information with info_id of population_id

        Returns:
            rewards[population_id][agent_id]: reward for agent_id in population
        """
        reward_value = mf_info["blue"]["mf_flat"]@self._reward_matrix@mf_info["red"]["mf_flat"].T 
        if abs(reward_value)<1e-5:
            reward_value = 0
        reward_dict = {}
        reward_dict["blue"] = {"agent_{}".format(agent_index): reward_value for agent_index in range(self._num_agents["blue"])}
        reward_dict["red"] = {"agent_{}".format(agent_index): -reward_value for agent_index in range(self._num_agents["red"])}
            
        return reward_dict
            

    def get_population_transition_probability(self, population_id_list: Union[List[str], None],
                                              mf_info: dict[str, dict[str, Union[tuple, ndarray]]]) -> List[ndarray]:
        """
        Get the full transition probability of population_id under the current mean-field for all states and actions

        Args:
            population_id: (str) population id
            mf_info [population_id][info_id]: mean-field information with info_id of population_id

        Returns:
            transition_prob[a][i, j]: transition probability from state i to state j under action a
        """
        
        return self._transition_matrix_dict
    
    def _get_transition_matrix(self) -> List[ndarray]:

        """Build the state transition matrices for all populations."""
        transition_prob = np.zeros((self._num_actions["blue"], self._num_states["blue"], self._num_states["blue"])) # same for blue and red
        
        # stay
        transition_prob[0][0][0] = 1
        transition_prob[0][1][1] = 1
        transition_prob[0][2][2] = 1
        
        # clockwise cyclic motion S0: Rock, S1: Paper, S2: Scissors
        transition_prob[1][0][1] = 1
        transition_prob[1][1][2] = 1
        transition_prob[1][2][0] = 1

        return {pop_id: transition_prob for pop_id in self._num_agents.keys()}

    def get_agent_transition_probability_action(self, population_id, agent_id, action,
                                                agent_info: dict[str, dict[str, dict[str, Union[int, tuple, ndarray]]]],
                                                mf_info: dict[str, dict[str, Union[tuple, ndarray]]]):
        """
        Get the transition probability of agent_id under action in population_id

        Args:
            population_id (str): population id
            agent_id (str): agent id
            action (int): action id
            agent_info [population_id][agent_id][info_id]: agent information with info_id of agent_id in population_id
            mf_info [population_id][info_id]: mean-field information with info_id of population_id

        Returns:
            transition_prob[j]: transition probability from agent's current state to state j under action
        """
        pass
