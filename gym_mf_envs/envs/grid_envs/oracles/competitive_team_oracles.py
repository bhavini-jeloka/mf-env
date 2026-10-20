import numpy as np
from numpy import ndarray
from typing import List, Union
from gym_mf_envs.envs.grid_envs.oracles.base_grid_oracle import GridMFOracle


class GridBattleMFOracle(GridMFOracle):
    """
    Grid-based Oracle for a two-population battlefield environment with finite population.
    
    This oracle models population interactions through:
    - Local interactions leading to death or resurrection,
    - Position transitions based on actions and obstacles,
    - Sparse rewards based on agents reaching designated targets.

    Populations consist of agents in either alive or dead states at different grid locations.
    """

    def __init__(self,
                 num_populations: int,
                 num_agents_dict: dict[str, int],
                 action_vectors_dict: dict[str, List[tuple]],
                 team_structure_dict: dict[str, List],
                 **kwargs):
        """Initialize this object and its environment state."""
        super().__init__(num_populations=num_populations, num_agents=num_agents_dict,
                         action_vectors=action_vectors_dict,
                         **kwargs)
        """
        Initialize the oracle with grid configuration, obstacles, targets, and transition parameters.
        """
        assert (len(self.population_id_list) == 2 and
                "blue" in self.population_id_list and "red" in self.population_id_list), \
            "Battle oracle currently only supports two populations with blue and red teams"

        # Deactivation penalty ratio for each population
        self._deactivation_penalty_ratio = kwargs.get("deactivation_penalty_ratio",
                                                      {pop_id: 1 for pop_id in num_agents_dict.keys()})

        # Whether the target is absorbing for each population
        self._absorbing_targets = kwargs.get("absorbing_targets", {pop_id: False for pop_id in num_agents_dict.keys()})

        # team_structure[team_id] = [pop_id1, pop_id2, ...]
        self._team_structure_dict = team_structure_dict

        # interaction coefficients
        self.alpha = kwargs.get("alpha", {"blue": 100, 'red': 5})
        self.beta = kwargs.get("beta", {"blue": 0, 'red': 0})
        self.old_reward_from_target = 0
        self.old_reward_from_deactivation = 0
        self.old_population_alive_at_target = 0

    def get_reward(self, agent_info: dict[str, dict[str, dict[str, Union[int, tuple, ndarray]]]],
                   mf_info: dict[str, dict[str, Union[tuple, ndarray]]]) -> dict[str, dict[str, float]]:
        """
        Compute rewards for populations based on the fraction of Blue agents alive at the target.
        """
        reward_dict = {pop_id: {"agent_{}".format(agent_index): 0. for agent_index in range(self._num_agents[pop_id])}
                       for pop_id in self._num_agents.keys()}

        reward = 0
        for pop_id in self.get_population_id_within_team("blue"):
            mf = mf_info[pop_id]["mf_flat"]

            # in target reward
            states_alive_in_target = self.status2state(status_list=[(pos, 1) for pos in self._target_pos[pop_id]],
                                                       population_id=pop_id)

            population_alive_at_target = np.sum([mf[s] for s in states_alive_in_target])
            old_population_alive_at_target = self.old_population_alive_at_target
            reward = 100 * (population_alive_at_target - old_population_alive_at_target)

        for pop_id in self.population_id_list:
            if pop_id == "blue":
                for agent_index in range(self._num_agents[pop_id]):
                    reward_dict[pop_id]["agent_{}".format(agent_index)] = reward
            else:
                for agent_index in range(self._num_agents[pop_id]):
                    reward_dict[pop_id]["agent_{}".format(agent_index)] = -reward

        self.old_population_alive_at_target = population_alive_at_target
        return reward_dict

    def update_mf_info(self, agent_info: dict[str, dict[str, dict[str, Union[int, tuple, ndarray]]]],
                       mf_info: dict[str, dict[str, Union[tuple, ndarray]]]):
        """
        Update mean-field information using local positions and status of the agents.
        """
        for pop_id in self.population_id_list:
            grid_size = self.get_population_grid_world_simulator(pop_id).size
            num_agents = self._num_agents[pop_id]
            mf_population_flat = np.zeros(self._num_states[pop_id], dtype=np.float32)
            mf_population_3d = np.zeros((grid_size[0], grid_size[1], 2), dtype=np.float32)
            for agent_id in range(self._num_agents[pop_id]):
                agent_state = agent_info[pop_id]["agent_{}".format(agent_id)]["current-state"]
                agent_pos = agent_info[pop_id]["agent_{}".format(agent_id)]["current-location"]
                alive = 1 if agent_info[pop_id]["agent_{}".format(agent_id)]["current-alive"] else 0
                mf_population_flat[agent_state] += 1
                mf_population_3d[agent_pos[0], agent_pos[1], alive] += 1

            mf_info[pop_id]["mf_flat"] = mf_population_flat / num_agents
            mf_info[pop_id]["mf_3d"] = mf_population_3d / num_agents

    def reset_reward(self):
        """Reset internal reward state."""
        self.old_population_alive_at_target = 0

    def get_population_transition_probability(self, population_id_list: Union[List[str], None],
                                              mf_info: dict[str, dict[str, Union[tuple, ndarray]]]) -> dict[str, list]:
        """
        Compute transition probabilities for given populations using status transitions and position.
        """
        if population_id_list is None:
            pop_id_list = self.population_id_list
        else:
            pop_id_list = population_id_list

        status_transition_matrix = self.get_status_transition_matrix(mf_info)

        transition_prob_dict = {pop_id: [[self.get_new_state_prob(pop_id=pop_id,
                                                                  status_trans=status_transition_matrix,
                                                                  status_state=s, action=a)
                                          for s in range(self._num_states[pop_id])]
                                         for a in range(self._num_actions[pop_id])]
                                for pop_id in pop_id_list}
        return transition_prob_dict

    def get_new_state_prob(self, pop_id: str, status_trans: dict[str, ndarray],
                           status_state: int, action: int) -> List[List[Union[int, float]]]:
        """
        Get the new state probabilities from current state and action.
        """
        simulator = self.get_population_grid_world_simulator(pop_id)
        pos, alive = self.state2status([status_state], pop_id)[0]
        pos_trans = simulator.get_state_action_transition_probability(simulator.get_pos_state(pos),
                                                                      self.action_vectors[pop_id][action])
        status_trans_s = status_trans[pop_id][:, status_state]

        if alive:
            # if transition to dead then no move
            to_state_list = [self.status2state([(pos, 0)], pop_id)[0]] + \
                            [self.status2state([(simulator.get_pos(new_pos_state), 1)], pop_id)[0]
                             for new_pos_state in pos_trans[0]]
            prob_list = [status_trans_s[0]] + [pos_prob * status_trans_s[1] for pos_prob in pos_trans[1]]
        else:
            to_state_list = [self.status2state([(pos, new_alive)], pop_id)[0] for new_alive in [0, 1]]
            prob_list = [status_prob for status_prob in status_trans_s]

        assert sum(prob_list) == 1
        return [to_state_list, prob_list]

    def get_status_transition_matrix(self, mf_info: dict[str, dict[str, Union[tuple, ndarray]]],
                                     pop_id_list: List[str] = None,
                                     state_list_dict: dict[str, List[int]] = None) -> dict[str, ndarray]:
        """
        Get the status transition matrix for a population.
        """

        advantage = self.get_advantage(mf_info, pop_id_list)
        if pop_id_list is None:
            pop_id_list = self.population_id_list
        else:
            assert all(
                [pop_id in pop_id_list for pop_id in state_list_dict.keys()]), "state_list contains invalid pop_id"

        if state_list_dict is None:
            state_list_dict = {pop_id: range(self._num_states[pop_id]) for pop_id in pop_id_list}

        status_trans_matrix = {
            pop_id: np.array(
                [[1 - np.clip(self.beta[pop_id] * advantage[pop_id][s], 0, 1) if self.isDead(s, pop_id)
                  else np.clip(-self.alpha[pop_id] * advantage[pop_id][s], 0, 1)
                  for s in state_list_dict[pop_id]],
                 [np.clip(self.beta[pop_id] * advantage[pop_id][s], 0, 1) if self.isDead(s, pop_id)
                  else 1 - np.clip(-self.alpha[pop_id] * advantage[pop_id][s], 0, 1)
                  for s in state_list_dict[pop_id]]
                 ]
            )
            for pop_id in pop_id_list
        }

        return status_trans_matrix

    def get_advantage(self, mf_info: dict[str, dict[str, Union[tuple, ndarray]]],
                      pop_id_list: List[str] = None) -> dict[str, ndarray]:
        """
        Get the advantage of each state in the population against other populations.
        """
        if pop_id_list is None:
            pop_id_list = self.population_id_list
        advantage = {pop_id: np.zeros(self._num_states[pop_id]) for pop_id in pop_id_list}
        for pop_id in pop_id_list:
            for state in range(self._num_states[pop_id]):
                advantage[pop_id][state] = self._get_state_advantage(state, pop_id, mf_info)
        return advantage

    def _get_state_advantage(self, state: int, pop_id: str,
                             mf_info: dict[str, dict[str, Union[tuple, ndarray]]]) -> float:
        """
        Return local advantage at a specific state: own population alive minus opponent alive at same position.
        """
        pos, _ = self.state2status([state], pop_id)[0]
        opponent_pop_id = "blue" if pop_id == "red" else "red"
        state_advantage = mf_info[pop_id]["mf_3d"][pos[0], pos[1], 1] - mf_info[opponent_pop_id]["mf_3d"][
            pos[0], pos[1], 1]
        return state_advantage

    def status2state(self, status_list: List[tuple], population_id: str) -> List[int]:
        """Convert list of (pos, alive) to flat state indices."""
        state_list = []
        grid_simulator = self.get_population_grid_world_simulator(population_id)
        for pos, alive in status_list:
            tmp = grid_simulator.get_pos_state(pos=pos)
            state = alive * grid_simulator.get_n_states() + tmp
            state_list.append(state)
        return state_list

    def state2status(self, state_list: List[int], population_id: str) -> List[tuple]:
        """Convert flat state indices to (pos, alive) tuples."""
        grid_simulator = self.get_population_grid_world_simulator(population_id)
        n_pos_states = grid_simulator.get_n_states()
        status_list = [(grid_simulator.get_pos(s % n_pos_states), s // n_pos_states) for s in state_list]
        return status_list

    def get_alive_states(self, population_id: str) -> List[int]:
        """Get list of state indices where agents are alive."""
        return [i for i in range(self._num_states[population_id] // 2, self._num_states[population_id])]

    def get_dead_states(self, population_id: str) -> List[int]:
        """Get list of state indices where agents are dead."""
        return [i for i in range(self._num_states[population_id] // 2)]

    def get_population_id_within_team(self, team_id: str) -> List[str]:
        """Return population IDs for the given team ID."""
        return self._team_structure_dict[team_id]

    def get_num_states_dict(self):
        """Return a dictionary of population IDs to number of total states (alive + dead)."""
        return {population_id: self.get_population_grid_world_simulator(population_id).get_n_states() * 2 for
                population_id in self.population_id_list}

    def isDead(self, state: int, pop_id: str) -> bool:
        """Check if a state index corresponds to a dead status."""
        return state < self._num_states[pop_id] // 2


class VirusMFOracle(GridMFOracle):

    def __init__(self,
                 num_populations: int,
                 num_agents_dict: dict[str, int],
                 action_vectors_dict: dict[str, List[tuple]],
                 team_structure_dict: dict[str, List],
                 **kwargs):
        """Initialize this object and its environment state."""
        super().__init__(num_populations=num_populations, num_agents=num_agents_dict,
                         action_vectors=action_vectors_dict,
                         **kwargs)
        """
        Initialize the oracle with grid configuration, obstacles, targets, and transition parameters.
        """
        assert (len(self.population_id_list) == 2 and
                "blue" in self.population_id_list and "red" in self.population_id_list), \
            "Battle oracle currently only supports two populations with blue and red teams"

        # team_structure[team_id] = [pop_id1, pop_id2, ...]
        self._team_structure_dict = team_structure_dict

        # interaction coefficients
        self.alpha = kwargs.get("alpha", 3)
        self.beta = kwargs.get("beta", 0.5)

        # Number of status states per population
        self.n_status_state = {"blue": 2, "red": 1}

        # Environment structure
        self.n_obstacles_dict, self._obstacle_locations_dict = kwargs.get("n_obstacles_dict"), kwargs.get("obstacle_locations_dict")
        self.n_hospitals_dict, self._hospital_locations_dict = kwargs.get("n_hospitals_dict"), kwargs.get("hospital_locations_dict")



    def reset(self, **kwargs):
        """Reset episode state and return the initial observation and info."""
        maps = kwargs.get("grid_world_map_list")
        assert maps is not None
        for population_id in self._num_agents.keys():
            self.get_population_grid_world_simulator(population_id).reset(new_map=maps[population_id])
        self._num_states = self.get_num_states_dict()
        self._parse_maps(maps)

        self.hospital_indices_dict = {"blue": []}

        for pop_id in ["blue"]:
            for hospital in self._hospital_locations_dict[pop_id]:
                self.hospital_indices_dict[pop_id].append(self.get_population_grid_world_simulator(pop_id).get_pos_state(hospital))

        self._generate_position_transition_matrix()

    def get_reward(self, agent_info: dict[str, dict[str, dict[str, Union[int, tuple, ndarray]]]],
                   mf_info: dict[str, dict[str, Union[tuple, ndarray]]]) -> dict[str, dict[str, float]]:
        """
        Compute rewards for populations based on the fraction of Blue agents alive at the target.
        """
        reward_dict = {pop_id: {"agent_{}".format(agent_index): 0. for agent_index in range(self._num_agents[pop_id])}
                       for pop_id in self._num_agents.keys()}

        mf = mf_info["blue"]["mf_flat"]
        healthy_population = np.sum(mf[self._num_states["blue"] //2 :])
        reward = 10 * healthy_population

        for pop_id in self.population_id_list:
            if pop_id == "blue":
                for agent_index in range(self._num_agents[pop_id]):
                    reward_dict[pop_id]["agent_{}".format(agent_index)] = reward
            else:
                for agent_index in range(self._num_agents[pop_id]):
                    reward_dict[pop_id]["agent_{}".format(agent_index)] = -reward
        return reward_dict

    def update_mf_info(self, agent_info: dict[str, dict[str, dict[str, Union[int, tuple, ndarray]]]],
                       mf_info: dict[str, dict[str, Union[tuple, ndarray]]]):
        """
        Update mean-field information using local positions and status of the agents.
        """
        for pop_id in self.population_id_list:
            grid_size = self.get_population_grid_world_simulator(pop_id).size
            num_agents = self._num_agents[pop_id]
            mf_population_flat = np.zeros(self._num_states[pop_id], dtype=np.float32)

            n_status = self.n_status_state[pop_id]

            mf_population_3d = np.zeros((grid_size[0], grid_size[1], n_status), dtype=np.float32)
            for agent_id in range(self._num_agents[pop_id]):
                agent_state = agent_info[pop_id]["agent_{}".format(agent_id)]["current-state"]
                agent_pos = agent_info[pop_id]["agent_{}".format(agent_id)]["current-location"]
                healthy = 1 if agent_info[pop_id]["agent_{}".format(agent_id)]["current-alive"] else 0
                mf_population_flat[agent_state] += 1
                mf_population_3d[agent_pos[0], agent_pos[1], min(healthy, n_status - 1)] += 1

            mf_info[pop_id]["mf_flat"] = mf_population_flat / num_agents
            mf_info[pop_id]["mf_3d"] = mf_population_3d / num_agents

    def reset_reward(self):
        """Reset internal reward state."""
        pass

    def get_population_transition_probability(self, population_id_list: Union[List[str], None],
                                              mf_info: dict[str, dict[str, Union[tuple, ndarray]]]) -> dict[str, list]:
        """
        Compute transition matrix for each population based on current policy and interaction model.
        """
        transition_matrix_dict = {}
        for pop_id in self._num_states.keys():
            if pop_id == "blue":
                status_transition = self._status_transition_matrix(mf_info)[pop_id]

                tmp = status_transition[:, None, :, None] * self._position_transition_matrix[pop_id]
                tmp = np.reshape(tmp,(self._num_states[pop_id], self._num_states[pop_id], len(self.action_vectors[pop_id])))

            else:
                tmp = self._position_transition_matrix[pop_id]

            transition_matrix_dict[pop_id] = tmp
        return transition_matrix_dict

    def status2state(self, status_list: List[tuple], population_id: str) -> List[int]:
        """Convert list of (pos, alive) to flat state indices."""
        state_list = []
        grid_simulator = self.get_population_grid_world_simulator(population_id)
        for pos, alive in status_list:
            if population_id == "red":
                tmp = grid_simulator.get_pos_state(pos=pos)
                state_list.append(tmp)
            else:
                tmp = grid_simulator.get_pos_state(pos=pos)
                state = alive * grid_simulator.get_n_states() + tmp
                state_list.append(state)
        return state_list

    def state2status(self, state_list: List[int], population_id: str) -> List[tuple]:
        """Convert flat state indices to (pos, alive) tuples."""
        grid_simulator = self.get_population_grid_world_simulator(population_id)
        n_pos_states = grid_simulator.get_n_states()
        status_list = [(grid_simulator.get_pos(s % n_pos_states), s // n_pos_states) for s in state_list]
        return status_list

    def get_population_id_within_team(self, team_id: str) -> List[str]:
        """Return population IDs for the given team ID."""
        return self._team_structure_dict[team_id]


    def _status_transition_matrix(self,
                                  mf_info):  # compute [(s' = 0|p, s, mu, nu), (s' = 1|p, s, mu, nu)] only the alive members contribute for each position
        """
        Compute transition probabilities between dead/alive states for each cell based on local interactions.
        """
        status_transition_matrices = {}

        for pop_id in self._num_states.keys():
            if pop_id == "red":
                status_transition_matrices["red"] = None
                continue


            infected_population_map = mf_info["blue"]["mf_3d"][:, :, 0] + mf_info["red"]["mf_3d"][:, :, 0]
            blue_grid_simulator = self.get_population_grid_world_simulator("blue")
            blue_pos_list = [blue_grid_simulator.get_pos(pos_state)
                             for pos_state in range(blue_grid_simulator.n_pos_state)]
            infected_population = np.array([infected_population_map[pos[0], pos[1]] for pos in blue_pos_list])

            interaction_infect =np.clip(self.alpha * infected_population, 0, 1)
            interaction_heal = np.zeros_like(interaction_infect)
            for hospital_idx in self.hospital_indices_dict["blue"]:
                interaction_heal[hospital_idx] = self.beta

            status_trans = np.zeros((2, 2, self.get_population_grid_world_simulator(pop_id).n_pos_state))
            status_trans[0, 0, :] = 1 - interaction_heal
            status_trans[1, 0, :] = interaction_heal
            status_trans[0, 1, :] = interaction_infect
            status_trans[1, 1, :] = 1 - interaction_infect

            # Normalize heal transitions
            s = status_trans[0, 0, :] + status_trans[1, 0, :]
            status_trans[0, 0, :] /= s
            status_trans[1, 0, :] /= s

            # Normalize infect transitions
            s = status_trans[0, 1, :] + status_trans[1, 1, :]
            status_trans[0, 1, :] /= s
            status_trans[1, 1, :] /= s

            status_transition_matrices[pop_id] = status_trans.reshape((2, -1))  # DoA for each position
        return status_transition_matrices

    def get_num_states_dict(self):
        """
        Returns the number of states for each population.
        """
        return {population_id: self.get_population_grid_world_simulator(population_id).get_n_states() * self.n_status_state[population_id] for
                population_id in self.population_id_list}

    def _generate_position_transition_matrix(self):
        """
        Precompute movement (position) dynamics for each population and state.
        """
        self._position_transition_matrix = {}

        for pop_id in self._num_states.keys():
            grid_simulator = self.get_population_grid_world_simulator(pop_id)
            action_vectors = self.action_vectors[pop_id]
            self._position_transition_matrix[pop_id] = np.zeros(
                (grid_simulator.get_n_states(), self._num_states[pop_id], len(self.action_vectors[pop_id])))

            for state in range(self._num_states[pop_id]):
                pos_state = state % grid_simulator.get_n_states()
                for a, action_vector in enumerate(action_vectors):
                    next_pos_list, prob_list = grid_simulator.get_state_action_transition_probability(pos_state, action_vector)
                    for next_pos, prob in zip(next_pos_list, prob_list):
                        self._position_transition_matrix[pop_id][next_pos, state, a] = 1