import numpy as np
from typing import Tuple, List, Union
from numpy import ndarray
from sklearn.preprocessing import normalize
from gym_mf_envs.envs.base_env.base_oracle_infinite import BaseMFInfiniteOracle
import time


class GridBattleMFOracleInfinite(BaseMFInfiniteOracle):
    """
    Grid-based Oracle for a two-population battlefield environment with infinite population.
    
    This oracle models population interactions through:
    - Local interactions leading to death or resurrection,
    - Position transitions based on actions and obstacles,
    - Sparse rewards based on agents reaching designated targets.

    Populations consist of agents in either alive or dead states at different grid locations.
    """

    def __init__(self, num_populations: int,
                 grid_size: tuple[int],
                 num_actions: dict[str, int],
                 population_color_dict: dict[str, tuple] = {"blue": (0, 0, 255)},
                 n_obstacles_dict: dict[str, int] = {"blue": 2, },
                 obstacle_locations_dict: dict[str, Union[List[tuple], None]] = {"blue": None},
                 n_targets_dict: dict[str, int] = {"blue": 2},
                 target_locations_dict: dict[str, Union[List[tuple], None]] = {"blue": None}, **kwargs):

        """
        Initialize the oracle with grid configuration, obstacles, targets, and transition parameters.
        """

        # State space: each cell has two states (dead and alive)
        num_states = grid_size[0] * grid_size[1] * 2
        self._num_states = {"blue": num_states, "red": num_states}
        self._num_actions = num_actions

        self._num_pos = grid_size[0] * grid_size[1]
        self._grid_size = grid_size

        # Action: right, down, left, up, stay
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0]),
            4: np.array([0, 0])
        }

        # Interaction parameters
        self.alpha = kwargs.get("alpha", {"blue": 100, 'red': 5})
        self.beta = kwargs.get("beta", {"blue": 0, 'red': 0})
        self.old_reward_from_target = 0
        self.old_reward_from_deactivation = 0
        self.old_population_alive_at_target = 0

        # Environment structure
        self.n_obstacles_dict, self._obstacle_locations_dict = n_obstacles_dict, obstacle_locations_dict
        self.n_targets_dict, self._target_locations_dict = n_targets_dict, target_locations_dict

        self.target_indices_dict = {"blue": [], "red": []}
        self.obstacle_indices_dict = {"blue": [], "red": []}

        self.target_test = np.zeros(32)
        self.target_test[16] = 1

        for pop_id in self._num_states.keys():
            for target in self._target_locations_dict[pop_id]:
                self.target_indices_dict[pop_id].append(self._pos2index(target))
            for obstacle in self._obstacle_locations_dict[pop_id]:
                self.obstacle_indices_dict[pop_id].append(self._pos2index(obstacle))

        self._generate_neighbour_matrix()
        self._generate_position_transition_matrix()
        super().__init__(num_populations, self._num_states, self._num_actions)

    def reset(self, **kwargs):
        """Reset any internal state."""
        pass

    def update_mf_info(self, population_id_list: Union[List[str], None],
                       mf_info: dict[str, dict[str, Union[tuple, ndarray]]], policy: dict[str, Union[tuple, ndarray]]):
        """
        Update the mean-field distribution using current policy and transition probabilities.
        """
        transition_matrices = self.get_population_transition_probability(population_id_list, mf_info, policy)
        for pop_id in self.population_id_list:
            mf_info[pop_id] = mf_info[pop_id] @ transition_matrices[pop_id]
            mf_info[pop_id] = normalize(mf_info[pop_id].reshape(1, -1), norm='l1').flatten()

    def get_reward(self, mf_info: dict[str, dict[str, Union[tuple, ndarray]]],
                   policies: dict[str, dict[str, Union[tuple, ndarray]]]) -> dict:
        """
        Compute rewards for populations based on the fraction of Blue agents alive at the target.
        """
        population_alive_at_target = np.sum(
            mf_info["blue"][np.array(self.target_indices_dict["blue"]) + self._num_states["blue"] // 2])
        old_population_alive_at_target = self.old_population_alive_at_target
        reward_value = 100 * (population_alive_at_target - old_population_alive_at_target)
        reward_dict = {}
        reward_dict["blue"] = reward_value
        reward_dict["red"] = -reward_value
        self.old_population_alive_at_target = population_alive_at_target
        return reward_dict

    def reset_reward(self):
        """Reset internal reward state."""
        self.old_population_alive_at_target = 0

    def get_population_transition_probability(self, population_id_list: Union[List[str], None],
                                              mf_info: dict[str, dict[str, Union[tuple, ndarray]]],
                                              policy: dict[str, Union[tuple, ndarray]]) -> List[ndarray]:
        """
        Compute transition matrix for each population based on current policy and interaction model.
        """
        transition_matrix_dict = {}
        for pop_id in self._num_states.keys():
            policy[pop_id] = policy[pop_id].reshape(self._num_states[pop_id], self._num_actions[pop_id])
            policy[pop_id] = normalize(policy[pop_id], axis=1, norm='l1')

            status_transition = self._status_transition_matrix(mf_info)[pop_id]

            tmp = status_transition[:, None, :, None] * self._position_transition_matrix_stack[pop_id]
            tmp = np.reshape(tmp, (self._num_states[pop_id], self._num_states[pop_id], len(self._action_to_direction)))

            transition_matrix = np.einsum('ijk, jk->ij', tmp, policy[pop_id])
            transition_matrix_dict[pop_id] = np.transpose(transition_matrix)

        return transition_matrix_dict

    def convert_obs(self, mf_info):
        """
        Convert mean-field info into a structured grid tensor for visualization or learning.
        Returns a tensor of shape (4, grid_x, grid_y) with channels for blue/red and alive/dead.
        """
        mf_combined = np.zeros((self._grid_size[0], self._grid_size[1], 4), dtype=np.float32)

        for i, pop_id in enumerate(["blue", "red"]):  # Iterate directly over keys
            num_half_states = self._num_states[pop_id] // 2  # Avoid repeated computation
            mf_combined[:, :, 2 * i] = np.reshape(mf_info[pop_id][:num_half_states], self._grid_size).T
            mf_combined[:, :, 2 * i + 1] = np.reshape(mf_info[pop_id][num_half_states:], self._grid_size).T
            assert (abs(np.sum(mf_combined[:, :, 2 * i: 2 * i + 2]) - 1) < 1e-5)

        return mf_combined.T

    def check_mf_termination(self, mf_info):
        """
        Determine whether the mean-field game had terminated.
        """
        agents_dead_blue = np.sum(mf_info["blue"][:self._num_states["blue"] // 2])
        agents_dead_red = np.sum(mf_info["red"][:self._num_states["red"] // 2])
        population_alive_at_target = np.sum(
            mf_info["blue"][np.array(self.target_indices_dict["blue"]) + self._num_states["blue"] // 2])
        done_status = (population_alive_at_target + agents_dead_blue == 1 or
                       agents_dead_red == 1)
        return {"blue": done_status, "red": done_status}

    def _pos2index(self, pos):
        """Convert (x, y) tuple to flattened grid index."""
        return np.ravel_multi_index(pos, self._grid_size)

    def _index2pos(self, idx):
        """Convert flattened index to (x, y) grid position."""
        return np.array(np.unravel_index(idx, self._grid_size))

    def neighbourhood(self, state):
        """
        Get neighbor indices for a given grid cell index.
        """
        pos = self._index2pos(state)
        indices = []
        indices = [state]
        return indices

    def _index2status(self, s):
        """Return (position_index, alive/dead) from global state index."""
        return s % self._num_pos, s // self._num_pos

    def _generate_neighbour_matrix(self):
        """Create neighborhood matrix for each population (used for local interactions)."""
        self.nbhd_matrices = {}

        for pop_id in self._num_states.keys():
            self.nbhd_matrices[pop_id] = np.zeros((self._num_pos, self._num_states[pop_id]))

        for pos in range(self._num_pos):
            neighbor_indices = self.neighbourhood(pos)
            for state in range(self._num_pos, 2 * self._num_pos):
                nbr_pos, nbr_status = self._index2status(state)
                if nbr_pos in neighbor_indices:
                    for pop_id in self._num_states.keys():
                        self.nbhd_matrices[pop_id][pos, state] = 1

    def _status_transition_matrix(self,
                                  mf_info):  # compute [(s' = 0|p, s, mu, nu), (s' = 1|p, s, mu, nu)] only the alive members contribute for each position
        """
        Compute transition probabilities between dead/alive states for each cell based on local interactions.
        """
        status_transition_matrices = {}

        for pop_id in self._num_states.keys():
            if pop_id == "blue":
                diff = self.rho * mf_info["blue"] - (1 - self.rho) * mf_info["red"]
            else:
                diff = (1 - self.rho) * mf_info["red"] - self.rho * mf_info["blue"]

            interaction = self.nbhd_matrices[pop_id] @ diff

            interaction_kill = np.clip(- self.alpha[pop_id] * interaction, 0, 1)
            interaction_ressurect = np.clip(self.beta[pop_id] * interaction, 0, 1)

            status_trans = np.zeros((2, 2, self._num_pos))
            status_trans[0, 0, :] = 1 - interaction_ressurect
            status_trans[1, 0, :] = interaction_ressurect
            status_trans[0, 1, :] = interaction_kill
            status_trans[1, 1, :] = 1 - interaction_kill

            status_transition_matrices[pop_id] = status_trans.reshape((2, -1))  # DoA for each position
        return status_transition_matrices

    def _generate_position_transition_matrix(self):
        """
        Precompute movement (position) dynamics for each population and state.
        """
        self._position_transition_matrix_stack = {}
        self._position_transition_matrix = {}
        self._position_transition_matrix_stay = {}

        for pop_id in self._num_states.keys():
            self._position_transition_matrix[pop_id] = np.zeros(
                (self._num_pos, self._num_states[pop_id], len(self._action_to_direction)))

        for state in range(self._num_pos * 2):
            pos, status = self._index2status(state)
            if status == 0:
                for pop_id in self._num_states.keys():
                    self._position_transition_matrix[pop_id][pos, state, :] = 1

            else:
                for action in range(len(self._action_to_direction)):
                    next_pos = np.clip(self._index2pos(pos) + self._action_to_direction[action], 0,
                                       self._grid_size[0] - 1)
                    next_pos = self._pos2index(next_pos)
                    for pop_id in self._num_states.keys():
                        if np.isin(next_pos, self.obstacle_indices_dict[pop_id]):
                            next_pos = pos
                        self._position_transition_matrix[pop_id][next_pos, state, action] = 1

        # Enforcing Dead agents to stay in place.
        for pop_id in self._num_states.keys():
            self._position_transition_matrix_stay[pop_id] = np.zeros(
                (self._num_pos, self._num_states[pop_id], len(self._action_to_direction)))
            for state in range(self._num_pos * 2):
                pos, status = self._index2status(state)
                self._position_transition_matrix_stay[pop_id][pos, state, :] = 1
            self._position_transition_matrix_stack[pop_id] = np.stack(
                [self._position_transition_matrix_stay[pop_id], self._position_transition_matrix[pop_id]])


class VirusMFOracleInfinite(BaseMFInfiniteOracle):
    """
    Grid-based Oracle for a two-population virus environment with infinite population.

    This oracle models population interactions through:
    - Local interactions leading to infected or healthy,
    - Position transitions based on actions and obstacles,
    - Sparse rewards based on agents reaching designated targets.

    Populations consist of agents in either alive or dead states at different grid locations.
    """

    def __init__(self, num_populations: int,
                 grid_size: tuple[int],
                 num_actions: dict[str, int],
                 population_color_dict: dict[str, tuple] = {"blue": (0, 0, 255)},
                 n_obstacles_dict: dict[str, int] = {"blue": 2, },
                 obstacle_locations_dict: dict[str, Union[List[tuple], None]] = {"blue": None},
                 n_hospitals_dict: dict[str, int] = {"blue": 2},
                 hospital_locations_dict: dict[str, Union[List[tuple], None]] = {"blue": None}, **kwargs):

        """
        Initialize the oracle with grid configuration, obstacles, targets, and transition parameters.
        """

        # State space: each cell has two states (dead and alive)
        num_states_blue, num_states_red = grid_size[0] * grid_size[1] * 2,  grid_size[0] * grid_size[1]
        self._num_states = {"blue": num_states_blue, "red": num_states_red}
        self._num_actions = num_actions

        self._num_pos = grid_size[0] * grid_size[1]
        self._grid_size = grid_size

        # Action: right, down, left, up, stay
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0]),
            4: np.array([0, 0])
        }

        # Interaction parameters
        self.alpha = kwargs.get("alpha", 3)
        self.beta = kwargs.get("beta", 0.5)

        # Environment structure
        self.n_obstacles_dict, self._obstacle_locations_dict = n_obstacles_dict, obstacle_locations_dict
        self.n_hospitals_dict, self._hospital_locations_dict = n_hospitals_dict, hospital_locations_dict

        self.hospital_indices_dict = {"blue": [], "red": []}
        self.obstacle_indices_dict = {"blue": [], "red": []}

        self.target_test = np.zeros(32)
        self.target_test[16] = 1

        for pop_id in self._num_states.keys():
            for hospital in self._hospital_locations_dict[pop_id]:
                self.hospital_indices_dict[pop_id].append(self._pos2index(hospital))
            for obstacle in self._obstacle_locations_dict[pop_id]:
                self.obstacle_indices_dict[pop_id].append(self._pos2index(obstacle))

        self._generate_position_transition_matrix()
        super().__init__(num_populations, self._num_states, self._num_actions)

    def reset(self, **kwargs):
        """Reset any internal state."""
        seed = kwargs.get("seed", None)
        np.random.default_rng(seed)

    def update_mf_info(self, population_id_list: Union[List[str], None],
                       mf_info: dict[str, dict[str, Union[tuple, ndarray]]], policy: dict[str, Union[tuple, ndarray]]):
        """
        Update the mean-field distribution using current policy and transition probabilities.
        """
        transition_matrices = self.get_population_transition_probability(population_id_list, mf_info, policy)
        for pop_id in self.population_id_list:
            mf_info[pop_id] = mf_info[pop_id] @ transition_matrices[pop_id]
            mf_info[pop_id] = normalize(mf_info[pop_id].reshape(1, -1), norm='l1').flatten()

    def get_reward(self, mf_info: dict[str, dict[str, Union[tuple, ndarray]]],
                   policies: dict[str, dict[str, Union[tuple, ndarray]]]) -> dict:
        """
        Compute rewards for populations based on the fraction of Blue agents alive at the target.
        """
        healthy_population = np.sum(mf_info["blue"][self._num_states["blue"] // 2:])
        reward_value = 10 * healthy_population
        reward_dict = {}
        reward_dict["blue"] = reward_value
        reward_dict["red"] = -reward_value
        return reward_dict

    def reset_reward(self):
        """Reset internal reward state."""
        pass

    def get_population_transition_probability(self, population_id_list: Union[List[str], None],
                                              mf_info: dict[str, dict[str, Union[tuple, ndarray]]],
                                              policy: dict[str, Union[tuple, ndarray]]) -> List[ndarray]:
        """
        Compute transition matrix for each population based on current policy and interaction model.
        """
        transition_matrix_dict = {}
        for pop_id in self._num_states.keys():
            policy[pop_id] = policy[pop_id].reshape(self._num_states[pop_id], self._num_actions[pop_id])
            policy[pop_id] = normalize(policy[pop_id], axis=1, norm='l1')

            if pop_id == "blue":
                status_transition = self._status_transition_matrix(mf_info)[pop_id]

                tmp = status_transition[:, None, :, None] * self._position_transition_matrix[pop_id]
                tmp = np.reshape(tmp,
                                 (self._num_states[pop_id], self._num_states[pop_id], len(self._action_to_direction)))

                transition_matrix = np.einsum('ijk, jk->ij', tmp, policy[pop_id])
                transition_matrix_dict[pop_id] = np.transpose(transition_matrix)
            else:
                transition_matrix = np.einsum('ijk, jk->ij',
                                              self._position_transition_matrix[pop_id],
                                              policy[pop_id])
                transition_matrix_dict[pop_id] = np.transpose(transition_matrix)

        return transition_matrix_dict

    def convert_obs(self, mf_info):
        """
        Convert mean-field info into a structured grid tensor for visualization or learning.
        Returns a tensor of shape (4, grid_x, grid_y) with channels for blue/red and alive/dead.
        """
        mf_combined = np.zeros((self._grid_size[0], self._grid_size[1], 3), dtype=np.float32)

        num_half_states = self._num_states["blue"] // 2
        mf_combined[:, :, 0] = np.reshape(mf_info["blue"][:num_half_states], self._grid_size).T
        mf_combined[:, :, 1] = np.reshape(mf_info["blue"][num_half_states:], self._grid_size).T
        assert (abs(np.sum(mf_combined[:, :, 0:2]) - 1) < 1e-5)

        mf_combined[:, :, 2] = np.reshape(mf_info["red"], self._grid_size).T
        assert (abs(np.sum(mf_combined[:, :, 2]) - 1) < 1e-5)

        return mf_combined.T

    def check_mf_termination(self, mf_info):
        """
        Determine whether the mean-field game had terminated.
        """
        agents_infected_blue = np.sum(mf_info["blue"][:self._num_states["blue"] // 2])
        done_status = agents_infected_blue == 1
        return {"blue": done_status, "red": done_status}

    def _pos2index(self, pos):
        """Convert (x, y) tuple to flattened grid index."""
        return np.ravel_multi_index(pos, self._grid_size)

    def _index2pos(self, idx):
        """Convert flattened index to (x, y) grid position."""
        return np.array(np.unravel_index(idx, self._grid_size))

    def neighbourhood(self, state):
        """
        Get neighbor indices for a given grid cell index.
        """
        pos = self._index2pos(state)
        indices = []
        indices = [state]
        return indices

    def _index2status(self, s):
        """Return (position_index, alive/dead) from global state index."""
        return s % self._num_pos, s // self._num_pos

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

            infected_population = mf_info["blue"][0:self._num_states["blue"] // 2] + mf_info["red"]

            interaction_infect =np.clip(self.alpha * infected_population, 0, 1)
            interaction_heal = np.zeros_like(interaction_infect)
            for hospital_idx in self.hospital_indices_dict["blue"]:
                interaction_heal[hospital_idx] = self.beta

            status_trans = np.zeros((2, 2, self._num_pos))
            status_trans[0, 0, :] = 1 - interaction_heal
            status_trans[1, 0, :] = interaction_heal
            status_trans[0, 1, :] = interaction_infect
            status_trans[1, 1, :] = 1 - interaction_infect

            status_transition_matrices[pop_id] = status_trans.reshape((2, -1))  # DoA for each position
        return status_transition_matrices

    def _generate_position_transition_matrix(self):
        """
        Precompute movement (position) dynamics for each population and state.
        """
        self._position_transition_matrix = {}

        for pop_id in self._num_states.keys():
            self._position_transition_matrix[pop_id] = np.zeros(
                (self._num_pos, self._num_states[pop_id], len(self._action_to_direction)))

            for state in range(self._num_states[pop_id]):
                pos, status = self._index2status(state)
                for action in range(len(self._action_to_direction)):
                    next_pos = np.clip(self._index2pos(pos) + self._action_to_direction[action], 0,
                                       self._grid_size[0] - 1)
                    next_pos = self._pos2index(next_pos)
                    if state >= self._num_states[pop_id]:
                        continue
                    if np.isin(next_pos, self.obstacle_indices_dict[pop_id]):
                        next_pos = pos
                    self._position_transition_matrix[pop_id][next_pos, state, action] = 1
