from typing import Tuple, List, Union, Any
import numpy as np
import gymnasium as gym
import pygame
from gymnasium import spaces
import imageio
from pathlib import Path
import random

from gym_mf_envs.envs.base_env.base_env_infinite import BaseMFInfiniteEnv
from gym_mf_envs.envs.grid_envs.oracles import GridBattleMFOracleInfinite, VirusMFOracleInfinite
from gym_mf_envs.utils import flat_dict_values
from copy import copy


class MFBattleFieldInfiniteEnv(BaseMFInfiniteEnv):
    """
    Mean-field battlefield environment with infinite population.
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10,
    }

    def __init__(self, grid_size: tuple[int],
                 num_actions: dict[str, int],
                 num_population: int = 2,
                 population_color_dict: dict[str, tuple] = {"blue": (0, 0, 255)},
                 n_obstacles_dict: dict[str, int] = {"blue": 2, },
                 obstacle_locations_dict: dict[str, Union[List[tuple], None]] = {"blue": None},
                 n_targets_dict: dict[str, int] = {"blue": 2},
                 target_locations_dict: dict[str, Union[List[tuple], None]] = {"blue": None},
                 render_mode=None, **kwargs):

        """
        Initializes the mean-field battlefield environment.
        """

        self.window_size = 512

        self._num_population = num_population
        num_states = grid_size[0] * grid_size[1] * 2  # Each position has 2 slots (alive/dead)
        self._num_states = {"blue": num_states, "red": num_states}
        self._population_color_dict = population_color_dict
        self._num_actions = num_actions
        self._num_pos = grid_size[0] * grid_size[1]
        self._grid_size = grid_size

        # Optional parameters alpha and beta for transition dynamics
        self.alpha = kwargs.get("alpha", {"blue": 100, 'red': 5})
        self.beta = kwargs.get("beta", {"blue": 0, 'red': 0})

        # Store obstacle/target details
        self.n_obstacles_dict, self._obstacle_locations_dict = n_obstacles_dict, obstacle_locations_dict
        self.n_targets_dict, self._target_locations_dict = n_targets_dict, target_locations_dict

        self.target_indices_dict = {"blue": [], "red": []}
        self.obstacle_indices_dict = {"blue": [], "red": []}

        # Create initial map and cache obstacle/target indices
        self.map = self._init_map()

        for pop_id in self._num_states.keys():
            for target in self._target_locations_dict[pop_id]:
                self.target_indices_dict[pop_id].append(self._pos2index(target))
            for obstacle in self._obstacle_locations_dict[pop_id]:
                self.obstacle_indices_dict[pop_id].append(self._pos2index(obstacle))

        # render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Call parent class constructor
        super().__init__(render_mode, num_population, population_color_dict)

    def demo(self, **kwargs):
        """Run the environment’s built-in demonstration."""
        pass

    def _init_map(self):
        """
        Create map with 1s for obstacles and targets, and 0s elsewhere.
        """

        # Initialize grid with zeros
        grid = np.zeros(self._grid_size, dtype=np.int32)

        # Function to set positions to 1
        def set_positions(location_dict):
            """Set the initial positions for the configured populations."""
            for locations in location_dict.values():
                if locations:  # Ensure the list is not empty
                    rows, cols = zip(*locations)  # Unzip into separate lists
                    grid[np.array(rows), np.array(cols)] = 1

        # Set obstacles and targets
        set_positions(self._obstacle_locations_dict)
        set_positions(self._target_locations_dict)

        return grid

    def _pos2index(self, pos):
        """
        Convert 2D grid position to 1D index.
        """
        return np.ravel_multi_index(pos, self._grid_size)

    def _get_obs(self):
        """
        Returns global observation: mean-field distributions across populations.
        """
        obs = {}
        obs["global-obs"] = np.concatenate(list(self.mf_info.values()))
        return obs

    def _get_info(self):
        """
        Returns metadata including termination flags and current timestep.
        """

        if self.mf_info["blue"] is None:
            terminal_obs = None
            done_dict = None
        else:
            terminal_obs = self._get_obs()["global-obs"]
            done_dict = self._check_mf_termination()
        info = {
            "time_step": self._time_step,
            "terminal_state": terminal_obs,
            "done_dict": done_dict
        }
        return info

    def _check_mf_termination(self) -> dict[str, dict[str, bool]]:
        """
        Check termination conditions for each population via oracle.
        """
        return self.mf_oracle.check_mf_termination(self.mf_info)

    def _render_frame(self):
        """Render the current environment frame."""
        pass

    def _reset_specific_env_variables(self):
        """Reset environment-specific state for a new episode."""
        pass

    def _random_init_mf(self):
        """
        Randomly initialize the mean-field distributions for each population using various strategies.
        """

        self.mf_oracle.reset_reward()

        # Define initialization types and probabilities
        init_options = ["random", "split", "concentrated"]
        prob_init_options = [0.8, 0.1, 0.1]
        chosen_init = np.random.choice(init_options, p=prob_init_options)

        # Get valid locations (where map == 0)
        valid_locations = np.array(list(zip(*np.where(self.map == 0))))  # Shape (N, 2)
        valid_indices = np.ravel_multi_index(valid_locations.T, self._grid_size)
        valid_indices_alive = valid_indices + self._num_states["blue"] // 2
        valid_indices_DoA = np.concatenate((valid_indices, valid_indices + self._num_states["blue"] // 2))

        # Determine if using a shared blob for "concentrated" initialization
        shared_blob = self.np_random.random() < 0.05
        if shared_blob:
            blob_location = self._get_random_valid_location()

        for population_id in self._num_states.keys():
            self.mf_info[population_id] = np.zeros(self._num_states[population_id])

            if chosen_init == "concentrated":
                if not shared_blob:
                    blob_location = self._get_random_valid_location()
                blob_index = np.ravel_multi_index(blob_location, self._grid_size)
                self.mf_info[population_id][blob_index + self._num_states[population_id] // 2] = 1

            elif chosen_init == "split":
                num_splits = np.random.choice([1, 2], p=[0.3, 0.7])
                selected_pos = self._get_multiple_valid_locations(num_splits)
                selected_pos_index = np.ravel_multi_index(selected_pos.T, self._grid_size)
                init_dist = np.random.dirichlet(np.ones(num_splits))
                self.mf_info[population_id][selected_pos_index + self._num_states[population_id] // 2] = init_dist

            elif chosen_init == "sparse":
                num_zero = np.random.randint(6, 12)
                non_zero_indices = np.random.choice(len(valid_locations), len(valid_locations) - num_zero,
                                                    replace=False)
                active_loc = valid_locations[non_zero_indices]
                probability_value = 1 / (len(valid_locations) - num_zero)

                num_deactivated_loc = np.random.randint(1, len(non_zero_indices) + 1)
                selected_deactivated_indices = np.random.choice(non_zero_indices, num_deactivated_loc, replace=False)
                deactivated_loc = valid_locations[selected_deactivated_indices]

                # Remove deactivated locations from active ones
                active_loc = np.setdiff1d(active_loc.view([('', active_loc.dtype)] * 2),
                                          deactivated_loc.view([('', deactivated_loc.dtype)] * 2)
                                          ).view(active_loc.dtype).reshape(-1, 2)

                active_pos_index = np.ravel_multi_index(active_loc.T, self._grid_size)
                deactive_pos_index = np.ravel_multi_index(deactivated_loc.T, self._grid_size)
                self.mf_info[population_id][active_pos_index + self._num_states[population_id] // 2] = probability_value
                self.mf_info[population_id][deactive_pos_index] = probability_value

            elif chosen_init == "random":
                if np.random.rand() > 0.05:
                    self.mf_info[population_id][valid_indices_alive] = np.random.dirichlet(
                        np.ones(len(valid_locations)))
                else:
                    self.mf_info[population_id][valid_indices_DoA] = np.random.dirichlet(
                        np.ones(2 * len(valid_locations)))

    def _get_random_valid_location(self):
        """Returns a random valid location (row, col) from the map where value is 0."""
        while True:
            blob_location = self.np_random.integers(0, self._grid_size[0], size=2, dtype=int)
            if self.map[blob_location[0], blob_location[1]] == 0:
                return tuple(blob_location)

    def _get_multiple_valid_locations(self, num_locs):
        """Returns multiple valid locations (num_locs, 2) from the map where value is 0."""
        while True:
            selected_pos = self.np_random.integers(0, self._grid_size[0], size=(num_locs, 2), dtype=int)
            if np.all(self.map[selected_pos[:, 0], selected_pos[:, 1]] == 0):
                return selected_pos

    def _reset_mf_oracles(self, seed=None):
        """Reset mean-field oracle."""
        self.mf_oracle.reset(seed=seed)

    def _init_mf_info(self):
        """Return initial empty mean-field information dict."""
        return {key: self._init_population_mf_info() for key in self.population_id_list}

    def _init_population_mf_info(self):
        """Return default (None) info for a population."""
        return None

    def _init_env_info(self):
        """Create initial environment-level information."""
        pass

    def _init_observations(self):
        """Return initial global observation composed of all mean-field distributions."""
        obs = {}
        obs["global-obs"] = np.concatenate(list(self.mf_info.values())) if self.mf_info["blue"] is not None else None
        return obs

    def _setup_mf_oracle(self, **kwargs):
        """Initialize the oracle used for rewars and transitions."""
        return GridBattleMFOracleInfinite(num_populations=self._num_population,
                                          grid_size=self._grid_size,
                                          num_actions=self._num_actions,
                                          population_color_dict=self._population_color_dict,
                                          n_obstacles_dict=self.n_obstacles_dict,
                                          obstacle_locations_dict=self._obstacle_locations_dict,
                                          n_targets_dict=self.n_targets_dict,
                                          target_locations_dict=self._target_locations_dict,
                                          kwargs={"alpha": self.alpha, "beta": self.beta})

    def sample_actions(self):
        """
        Sample random action distributions for each population.
        """
        array_1 = np.random.dirichlet(np.ones(self._num_actions["blue"]), size=self._num_states["blue"])
        array_2 = np.random.dirichlet(np.ones(self._num_actions["red"]), size=self._num_states["red"])
        flattened_array = {"blue": array_1.flatten(), "red": array_2.flatten()}
        return flattened_array

    def _setup_observation_space(self):
        """Define observation space for global observation: a single vector 
        representing the distribution over all states for a given population
        based on the setup for DDPG-MFTG."""
        self.observation_space["global-obs"] = spaces.Box(low=0, high=1,
                                                          shape=(1, self._num_states["blue"] * self._num_population))

    def _setup_action_space(self):
        """Define action space for each population (in this case, local policy)."""
        for population_id in self._num_actions.keys():
            self.action_space[population_id] = spaces.Box(low=0, high=1,
                                                          shape=(1, self._num_states[population_id] * self._num_actions[
                                                              population_id]))


class MFVirusInfiniteEnv(BaseMFInfiniteEnv):
    """
    Mean-field virus infection environment with infinite population.
    Blue team represents the humans with position and status as states. (0: infected, 1: healthy)
    Red team represents the virus agents with only position as states.
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10,
    }

    def __init__(self, grid_size: tuple[int],
                 num_actions: dict[str, int],
                 num_population: int = 2,
                 population_color_dict: dict[str, tuple] = {"blue": (0, 0, 255)},
                 n_obstacles_dict: dict[str, int] = {"blue": 2, },
                 obstacle_locations_dict: dict[str, Union[List[tuple], None]] = {"blue": None},
                 n_hospitals_dict: dict[str, int] = {"blue": 0},
                 hospital_locations_dict: dict[str, Union[List[tuple], None]] = {"blue": None},
                 render_mode=None, **kwargs):

        """
        Initializes the mean-field battlefield environment.
        """

        self.window_size = 512

        self._num_population = num_population
        # Blue team (human) position has 2 status (alive/dead); Red team (virus) position has 1 status (active)
        self._num_states = {"blue": grid_size[0] * grid_size[1] * 2, "red": grid_size[0] * grid_size[1]}
        self._population_color_dict = population_color_dict
        self._num_actions = num_actions
        self._num_pos = grid_size[0] * grid_size[1]
        self._grid_size = grid_size

        # Optional parameters alpha and beta for transition dynamics
        self.alpha = kwargs.get("alpha", {"blue": 100, 'red': 5})
        self.beta = kwargs.get("beta", {"blue": 0, 'red': 0})

        # Store obstacle/hospital details
        self.n_obstacles_dict, self._obstacle_locations_dict = n_obstacles_dict, obstacle_locations_dict
        self.n_hospitals_dict, self._hospital_locations_dict = n_hospitals_dict, hospital_locations_dict

        self.hospital_indices_dict = {"blue": [], "red": []}
        self.obstacle_indices_dict = {"blue": [], "red": []}

        # Create initial map and cache obstacle/hospital indices
        self.red_freespace = self._init_map(
            [self._obstacle_locations_dict["red"], self._hospital_locations_dict["blue"]])
        self.blue_freespace = self._init_map(
            [self._obstacle_locations_dict["blue"], self._hospital_locations_dict["blue"]])

        for pop_id in self._num_states.keys():
            for hospital in self.hospital_indices_dict[pop_id]:
                self.hospital_indices_dict[pop_id].append(self._pos2index(hospital))
            for obstacle in self._obstacle_locations_dict[pop_id]:
                self.obstacle_indices_dict[pop_id].append(self._pos2index(obstacle))

        # render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Call parent class constructor
        super().__init__(render_mode, num_population, population_color_dict)

    def demo(self, **kwargs):
        """Run the environment’s built-in demonstration."""
        pass

    def _init_map(self, location_info_list):
        """
        Create map with 1s for obstacles and targets, and 0s elsewhere.
        """

        # Initialize grid with zeros
        grid = np.zeros(self._grid_size, dtype=np.int32)

        # Function to set positions to 1
        def set_positions(location_info):
            """Set the initial positions for the configured populations."""
            if len(location_info) > 0:  # Ensure the list is not empty
                rows, cols = zip(*location_info)  # Unzip into separate lists
                grid[np.array(rows), np.array(cols)] = 1

        for location_info in location_info_list:
            set_positions(location_info)

        return grid

    def _pos2index(self, pos):
        """
        Convert 2D grid position to 1D index.
        """
        return np.ravel_multi_index(pos, self._grid_size)

    def _get_obs(self):
        """
        Returns global observation: mean-field distributions across populations.
        """
        obs = {}
        obs["global-obs"] = np.concatenate(list(self.mf_info.values()))
        return obs

    def _get_info(self):
        """
        Returns metadata including termination flags and current timestep.
        """

        if self.mf_info["blue"] is None:
            terminal_obs = None
            done_dict = None
        else:
            terminal_obs = self._get_obs()["global-obs"]
            done_dict = self._check_mf_termination()
        info = {
            "time_step": self._time_step,
            "terminal_state": terminal_obs,
            "done_dict": done_dict
        }
        return info

    def _check_mf_termination(self) -> dict[str, dict[str, bool]]:
        """
        Check termination conditions for each population via oracle.
        """
        return self.mf_oracle.check_mf_termination(self.mf_info)

    def _render_frame(self):
        """Render the current environment frame."""
        pass

    def _reset_specific_env_variables(self):
        """Reset environment-specific state for a new episode."""
        pass

    def _random_init_mf(self):
        """
        Randomly initialize the mean-field distributions for each population using various strategies.
        """

        self.mf_oracle.reset_reward()

        # Define initialization types and probabilities
        init_options = ["random", "split", "concentrated"]
        prob_init_options = [0.8, 0.1, 0.1]
        chosen_init = np.random.choice(init_options, p=prob_init_options)

        # Get valid locations (where map == 0)
        blue_valid_locations = np.array(list(zip(*np.where(self.blue_freespace == 0))))  # Shape (N, 2)
        valid_blue_indices = np.ravel_multi_index(blue_valid_locations.T, self._grid_size)
        valid_blue_indices_alive = valid_blue_indices + self._num_states["blue"] // 2
        valid_blue_indices_DoA = np.concatenate(
            (valid_blue_indices, valid_blue_indices + self._num_states["blue"] // 2))

        red_valid_locations = np.array(list(zip(*np.where(self.red_freespace == 0))))
        valid_red_indices = np.ravel_multi_index(red_valid_locations.T, self._grid_size)

        for population_id in self._num_states.keys():
            self.mf_info[population_id] = np.zeros(self._num_states[population_id])

            if chosen_init == "concentrated":
                blob_location = self._get_random_valid_location(population_id)
                blob_index = np.ravel_multi_index(blob_location, self._grid_size)
                if population_id == "blue":
                    self.mf_info[population_id][
                        blob_index + self._num_states[population_id] // 2] = 1  # blue all healthy
                else:
                    self.mf_info[population_id][blob_index] = 1

            elif chosen_init == "split":
                num_splits = np.random.choice([1, 2], p=[0.3, 0.7])
                selected_pos = self._get_multiple_valid_locations(num_splits, population_id)
                selected_pos_index = np.ravel_multi_index(selected_pos.T, self._grid_size)
                init_dist = np.random.dirichlet(np.ones(num_splits))
                if population_id == "blue":
                    self.mf_info[population_id][selected_pos_index + self._num_states[population_id] // 2] = init_dist
                else:
                    self.mf_info[population_id][selected_pos_index] = init_dist

            elif chosen_init == "random":
                if population_id == "blue":
                    if np.random.rand() > 0.05:
                        self.mf_info[population_id][valid_blue_indices_alive] = np.random.dirichlet(
                            np.ones(len(blue_valid_locations)))

                    self.mf_info[population_id][valid_blue_indices_DoA] = np.random.dirichlet(
                        np.ones(2 * len(blue_valid_locations)))
                else:
                    self.mf_info[population_id][valid_red_indices] = np.random.dirichlet(
                        np.ones(len(red_valid_locations)))

    def _get_random_valid_location(self, population_id):
        """Returns a random valid location (row, col) from the map where value is 0."""
        while True:
            blob_location = self.np_random.integers(0, self._grid_size[0], size=2, dtype=int)

            map_tmp = self.blue_freespace if population_id == "blue" else self.red_freespace
            if map_tmp[blob_location[0], blob_location[1]] == 0:
                return tuple(blob_location)

    def _get_multiple_valid_locations(self, num_locs, population_id):
        """Returns multiple valid locations (num_locs, 2) from the map where value is 0."""
        while True:
            selected_pos = self.np_random.integers(0, self._grid_size[0], size=(num_locs, 2), dtype=int)
            map_tmp = self.blue_freespace if population_id == "blue" else self.red_freespace
            if np.all(map_tmp[selected_pos[:, 0], selected_pos[:, 1]] == 0):
                return selected_pos

    def _reset_mf_oracles(self, seed=None):
        """Reset mean-field oracle."""
        self.mf_oracle.reset(seed=seed)

    def _init_mf_info(self):
        """Return initial empty mean-field information dict."""
        return {key: self._init_population_mf_info() for key in self.population_id_list}

    def _init_population_mf_info(self):
        """Return default (None) info for a population."""
        return None

    def _init_env_info(self):
        """Create initial environment-level information."""
        pass

    def _init_observations(self):
        """Return initial global observation composed of all mean-field distributions."""
        obs = {}
        obs["global-obs"] = np.concatenate(list(self.mf_info.values())) if self.mf_info["blue"] is not None else None
        return obs

    def _setup_mf_oracle(self, **kwargs):
        """Initialize the oracle used for rewars and transitions."""
        return VirusMFOracleInfinite(num_populations=self._num_population,
                                     grid_size=self._grid_size,
                                     num_actions=self._num_actions,
                                     population_color_dict=self._population_color_dict,
                                     n_obstacles_dict=self.n_obstacles_dict,
                                     obstacle_locations_dict=self._obstacle_locations_dict,
                                     n_hospitals_dict=self.n_hospitals_dict,
                                     hospital_locations_dict=self._hospital_locations_dict,
                                     kwargs={"alpha": self.alpha, "beta": self.beta})

    def sample_actions(self):
        """
        Sample random action distributions for each population.
        """
        array_1 = np.random.dirichlet(np.ones(self._num_actions["blue"]), size=self._num_states["blue"])
        array_2 = np.random.dirichlet(np.ones(self._num_actions["red"]), size=self._num_states["red"])
        flattened_array = {"blue": array_1.flatten(), "red": array_2.flatten()}
        return flattened_array

    def _setup_observation_space(self):
        """Define observation space for global observation: a single vector
        representing the distribution over all states for a given population
        based on the setup for DDPG-MFTG."""
        self.observation_space["global-obs"] = spaces.Box(low=0, high=1,
                                                          shape=(1, self._num_states["blue"] + self._num_states["red"]))

    def _setup_action_space(self):
        """Define action space for each population (in this case, local policy)."""
        for population_id in self._num_actions.keys():
            self.action_space[population_id] = spaces.Box(low=0, high=1,
                                                          shape=(1, self._num_states[population_id] * self._num_actions[
                                                              population_id]))
