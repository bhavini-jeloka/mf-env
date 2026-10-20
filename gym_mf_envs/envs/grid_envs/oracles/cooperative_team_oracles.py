import numpy as np
from numpy import ndarray
from typing import List, Union
from gym_mf_envs.envs.grid_envs.oracles.base_grid_oracle import GridMFOracle


class GridNavMFOracle(GridMFOracle):
    """
    Grid-based Oracle for a team based navigation problem with finite population (multiple types of population can exist).
    Populations consist of agents in either alive or dead states at different grid locations.
    """
    def __init__(self, num_populations: int, num_agents: dict[str, int],
                 action_vectors: dict[str, List[Union[tuple, np.ndarray]]], **kwargs):
        """Initialize this object and its environment state."""
        super().__init__(num_populations=num_populations, num_agents=num_agents, action_vectors=action_vectors,
                         **kwargs)

        self._absorbing_targets = kwargs.get("absorbing_targets", {pop_id: True for pop_id in num_agents.keys()})
        self.old_population_at_target = {pop_id: 0 for pop_id in num_agents.keys()}
        self._num_agents = num_agents

    def get_reward(self, agent_info: dict[str, dict[str, dict[str, Union[int, tuple, ndarray]]]],
                   mf_info: dict[str, dict[str, Union[tuple, ndarray]]]) -> dict[str, dict[str, float]]:
        """
        Compute rewards for populations based on the fraction of Blue agents alive at the target.
        """
        rewards = {population_id: {"agent_{}".format(agent_index): 0. for agent_index in
                                   range(self._num_agents[population_id])}
                   for population_id in self._num_agents.keys()}
        for population_id in self._num_agents.keys():
            mf = mf_info[population_id]["mf_flat"]
            population_at_target = np.sum(mf[self._target_pos_states[population_id]])
            old_population_at_target = self.old_population_at_target[population_id]
            reward = (population_at_target) + self._time_penalty[population_id] #(population_at_target-old_population_at_target) + self._time_penalty[population_id]
            
            for agent_index in range(self._num_agents[population_id]):
                rewards[population_id]["agent_{}".format(agent_index)] = reward
            self.old_population_at_target[population_id] = population_at_target
        return rewards
    
    def reset_reward(self):
        """Reset internal reward state."""
        self.old_population_at_target = {pop_id: 0 for pop_id in self._num_agents.keys()}

    def update_mf_info(self, agent_info: dict[str, dict[str, dict[str, Union[int, tuple, ndarray]]]],
                       mf_info: dict[str, dict[str, Union[tuple, ndarray]]]):
        """
        Update mean-field information using local positions and status of the agents.
        """
        for population_id in self._num_agents.keys():
            grid_size = self.get_population_grid_world_simulator(population_id).size
            num_agents = self._num_agents[population_id]
            mf_population_flat = np.zeros(self._num_states[population_id], dtype=np.float32)
            mf_population_3d = np.zeros((grid_size[0], grid_size[1], 1), dtype=np.float32)
            for agent_id in range(self._num_agents[population_id]):
                agent_state = agent_info[population_id]["agent_{}".format(agent_id)]["current-state"]
                agent_pos = agent_info[population_id]["agent_{}".format(agent_id)]["current-location"]
                mf_population_flat[agent_state] += 1
                mf_population_3d[agent_pos[0], agent_pos[1], 0] += 1

            mf_info[population_id]["mf_flat"] = mf_population_flat / num_agents
            mf_info[population_id]["mf_3d"] = mf_population_3d / num_agents

    def get_population_transition_probability(self, population_id_list,
                                              mf_info: dict[str, dict[str, Union[tuple, ndarray]]]):
        """
        Compute transition probabilities for given populations using local agent position.
        """
        transition_prob = {}
        if population_id_list is None:
            population_id_list = self.population_id_list

        for population_id in population_id_list:
            num_states, num_actions = self._num_states[population_id], self._num_actions[population_id]
            transition_prob[population_id] = {"to_state": [[[] for s in range(num_states)] for a in range(num_actions)],
                               "prob": [[[] for s in range(num_states)] for a in range(num_actions)]}
            for state in range(num_states):
                for action in range(num_actions):
                    to_states, probs = self._get_state_action_transition_probability(state, action, population_id, mf_info)
                    transition_prob[population_id]["to_state"][action][state] = to_states
                    transition_prob[population_id]["prob"][action][state] = probs
        return transition_prob

    def get_agent_transition_probability_action(self, population_id, agent_id, action,
                                                agent_info: dict[str, dict[str, dict[str, Union[int, tuple, ndarray]]]],
                                                mf_info: dict[str, dict[str, Union[tuple, ndarray]]]):
        """
        Compute the transition probability for an agent taking a specific action.
        """
        agent_i_info = agent_info[population_id][agent_id]
        state = agent_i_info["current-state"]
        return self._get_state_action_transition_probability(state, action, population_id, mf_info)

    def _get_state_action_transition_probability(self, state: int, action: int, population_id,
                                                 mf_info: dict[str, dict[str, Union[tuple, ndarray]]]):
        """
        Compute the transition for a state-action pair.
        """
        grid_simulator = self.get_population_grid_world_simulator(population_id)
        agent_pos = grid_simulator.get_pos(state)

        new_pos, valid = grid_simulator.get_intended_new_pos(agent_pos, self.action_vectors[population_id][action])
        new_state = grid_simulator.get_pos_state(new_pos)
        if grid_simulator.is_pos_in_obstacle(new_pos):
            mf_s = mf_info[population_id]["mf_flat"][state]
            penetrate_prob = min(1, np.exp(10*(mf_s-0.3)))  # if more than 0.8 population is in the obstacle, the agent can penetrate
            return [state, new_state], [1 - penetrate_prob, penetrate_prob]
        else:
            return [new_state], [1.0]
