import numpy as np
import pygame
from gymnasium import spaces
from typing import List, Union
from gym_mf_envs.envs.grid_envs.envs.base_grid_env import BaseMFGridEnv
from gym_mf_envs.envs.grid_envs.oracles import GridBattleMFOracle, VirusMFOracle
from gym_mf_envs.wrappers.utils import flatten_dict_box_obs as flatten_obs
from copy import copy
import os
import re


class MFBattleFieldEnv(BaseMFGridEnv):
    """
    Mean-field battlefield environment with finite population.
    Agents in each population navigate a grid world toward their respective targets
    while avoiding obstacles and being deactivated by their opponent teams.
    The environment supports mean-field observations and a mixed cooperative-competitive
    learning structure between the two populations.
    """

    def __init__(self, render_mode=None,
                 grid_size: tuple[int] = (8, 8),
                 identical_grid_world=True,
                 num_population: int = 2,
                 num_agent_dict: dict[str, int] = {"blue": 2, "red": 2},
                 action_map_dict={"blue": [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)],
                                  "red": [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]},
                 population_color_dict: dict[str, tuple] = {"blue": (0, 0, 255), "red": (255, 0, 0)},
                 n_obstacles_dict={"blue": 0, "red": 0},
                 obstacle_locations_dict={"blue": [], "red": []},
                 random_init_obstacles_dict={"blue": False, "red": False},
                 penetrable_obstacles_dict={"blue": False, "red": False},
                 n_targets_dict={"blue": 0, "red": 2},
                 target_locations_dict={"blue": [], "red": [(0, 0), (0, 8)]},
                 random_init_targets_dict={"blue": False, "red": False},
                 oracle_kwargs={}, **kwargs):

        """
        Initializes the finite-agent grid environment.
        """

        kwargs = kwargs | {"identical_grid_world": identical_grid_world,
                           "penetrable_obstacles": penetrable_obstacles_dict}
        self._reset_file_counter = 0
        super().__init__(render_mode=render_mode,
                         grid_size=grid_size,
                         num_population=num_population,
                         num_agent_dict=num_agent_dict,
                         action_map_dict=action_map_dict,
                         population_color_dict=population_color_dict,
                         n_obstacles_dict=n_obstacles_dict,
                         obstacle_locations_dict=obstacle_locations_dict,
                         random_init_obstacles_dict=random_init_obstacles_dict,
                         penetrable_obstacles_dict=penetrable_obstacles_dict,
                         n_targets_dict=n_targets_dict,
                         target_locations_dict=target_locations_dict,
                         random_init_targets_dict=random_init_targets_dict,
                         oracle_kwargs=oracle_kwargs, **kwargs)

    def _update_agent_info(self, actions):
        """
        Executes a step of agent transitions based on the sampled actions and oracle-defined population transitions.
        Updates location, state, alive status, and target reach info.
        """
        population_trans_prob = self.mf_oracle.get_population_transition_probability(population_id_list=None,
                                                                                     mf_info=self.mf_info)
        for population_id in self._num_agents_dict.keys():
            for agent_index in range(self._num_agents_dict[population_id]):
                agent_id = "agent_{}".format(agent_index)
                action = actions[population_id][agent_id]
                state = self.agent_info[population_id][agent_id]["current-state"]
                to_states, probs = population_trans_prob[population_id][action][state]

                new_state = np.random.choice(to_states, p=probs)
                new_pos, alive = self.mf_oracle.state2status(state_list=[new_state], population_id=population_id)[0]

                reached_target = new_pos in self._target_locations_dict[population_id]
                self.agent_info[population_id][agent_id]["old-location"] = copy(
                    self.agent_info[population_id][agent_id]["current-location"])
                self.agent_info[population_id][agent_id]["old-alive"] = copy(
                    self.agent_info[population_id][agent_id]["current-alive"])
                self.agent_info[population_id][agent_id]["old-state"] = copy(
                    self.agent_info[population_id][agent_id]["current-state"])
                self.agent_info[population_id][agent_id]["old-target_reached"] = copy(
                    self.agent_info[population_id][agent_id]["current-target_reached"])
                self.agent_info[population_id][agent_id]["current-location"] = new_pos
                self.agent_info[population_id][agent_id]["current-alive"] = alive
                self.agent_info[population_id][agent_id]["current-state"] = new_state
                self.agent_info[population_id][agent_id]["current-target_reached"] = reached_target

    def _get_obs(self):
        """
        Constructs observation dictionary with:
        - Local location map per agent
        - Shared obstacle/target map (optional)
        - Global mean-field information (per population)
        """
        obs = {}
        mf_obs = {"mean-field-{}".format(pop_id):
                      self.mf_info[pop_id]["mf_3d"] for pop_id in self._num_agents_dict.keys()}
        for population_id in self._num_agents_dict.keys():
            obs[population_id] = {"shared-team-obs": {}}
            for agent_index in range(self._num_agents_dict[population_id]):
                agent_location = self.agent_info[population_id]["agent_{}".format(agent_index)]["current-location"]
                agent_alive = self.agent_info[population_id]["agent_{}".format(agent_index)]["current-alive"]
                location_map = np.zeros(self.grid_size + (1,), dtype=np.float32)
                if agent_location is not None:
                    location_map[
                        agent_location[0], agent_location[1], 0] = agent_alive * 2 - 1  # 1 for alive, -1 for dead
                local_obs = {"location-map": location_map}
                obs[population_id]["agent_{}-local-obs".format(agent_index)] = local_obs
            obs[population_id]["shared-team-obs"]["obstacle-target-map"] = np.zeros(self.grid_size + (1,),
                                                                                    dtype=np.float32)  # self._map_dict[population_id]
        obs["global-obs"] = mf_obs
        return obs

    def flatten_full_observation(self, obs):
        """
        Flatten the observation dictionary into vector-based observations.
        """
        obs_ = {
                   population_id:

                       {
                           "agent_{}-local-obs".format(idx): flatten_obs(
                               obs[population_id]["agent_{}-local-obs".format(idx)])
                           for idx in range(self._num_agents_dict[population_id])
                       } |
                       {
                           "shared-team-obs": flatten_obs(obs[population_id]["shared-team-obs"])
                       }

                   for population_id in self._num_agents_dict.keys()
               } | {
                   "global-obs": flatten_obs(obs["global-obs"])
               }
        return obs_

    def _get_info(self):
        """
        Returns metadata including termination flags and current timestep.
        """
        terminal_obs = self._get_obs()
        terminal_global_obs = terminal_obs["global-obs"]
        if all(value is not None for value in terminal_global_obs.values()):
            terminal_global_obs = flatten_obs(terminal_global_obs)
            terminal_obs = self.flatten_full_observation(terminal_obs)

        info = {
            "agent_locations":
                {
                    population_id:
                        {
                            agent_id: self.agent_info[population_id][agent_id]["current-location"]
                            for agent_id in self.agent_info[population_id]
                        }
                    for population_id in self.population_id_list
                },
            "target_locations": self._target_locations_dict,
            "terminal_mean_field": terminal_global_obs,
            "terminal_full_state": terminal_obs,
            "time_step": self._time_step,
            "done_dict": self._check_agent_termination()
        }
        return info

    def _setup_observation_space(self):
        '''
        local-obs: position of the agent represented as a 2D array,
                   where +1 represents the agent alive and -1 represents the agent dead
        global-obs: mean-field information of all populations
                    the obstacles locations and target locations, where obstacles are +1 and targets are -1
                    (the map of obstacles and targets is optional)
        '''
        for population_id in self._num_agents_dict.keys():
            for agent_index in range(self._num_agents_dict[population_id]):
                self.observation_space[population_id]["agent_{}-local-obs".format(agent_index)] = spaces.Dict(
                    {
                        "location-map":
                            spaces.Box(
                                low=-1, high=1, shape=(self.grid_size[0], self.grid_size[1], 1), dtype=np.float32
                            )
                    })
            self.observation_space[population_id]["shared-team-obs"] = spaces.Dict(
                {
                    "obstacle-target-map": spaces.Box(low=-1, high=1, shape=(self.grid_size[0], self.grid_size[1], 1)),
                }
            )
        self.observation_space["global-obs"] = spaces.Dict(
            {
                "mean-field-{}".format(pop_id): spaces.Box(low=-1, high=1,
                                                           shape=(self.grid_size[0], self.grid_size[1], 2))
                for pop_id in self._num_agents_dict.keys()
            }
        )

    def _init_observations(self):
        """Return initial global observation composed of local + common information."""
        for population_id in self._num_agents_dict.keys():
            for agent_index in range(self._num_agents_dict[population_id]):
                agent_id = "agent_{}".format(agent_index)
                self._observation_dict[population_id][agent_id]["local-obs"] = np.zeros(self.size + (1,))
                for pop_id in self._num_agents_dict.keys():
                    mf_index = "mean-fields-{}".format(pop_id)
                    self._observation_dict[population_id][agent_id]["global-obs"][mf_index] = self.mf_info[pop_id][
                        "mf_3d"]
                self._observation_dict[population_id][agent_id]["global-obs"]["obstacle-target-map"] = self._map_dict[
                    population_id]

    def _setup_action_space(self):
        '''
        Each agent's action is listed in self._action_map
        '''
        for population_id in self.population_id_list:
            self.action_space[population_id] = spaces.Discrete(len(self._action_map_dict[population_id]))

    def _init_agent_info(self, population_id, agent_id):
        """
        Returns default dictionary for an agent's tracked properties.
        """
        return {
            "old-location": None,
            "old-alive": True,
            "old-state": None,
            "old-target_reached": False,
            "current-location": None,
            "current-alive": True,
            "current-state": None,
            "current-target_reached": False,
            "rendered-position": None,
            "rendered-color": None,
        }

    def _init_mf_info(self):
        """Return initial mean-field information dict."""
        return {key: self._init_population_mf_info() for key in self.population_id_list}

    def _init_population_mf_info(self):
        """Return initial empty mean-field information dict."""
        return {
            "mf_3d": None,
            "mf_flat": None,
        }

    def _init_env_info(self):
        """Return initial environment information."""
        return {population_id: self._init_population_env_info() for population_id in self.population_id_list}

    def _init_population_env_info(self):
        """Return initial popualtion information."""
        return {
            "obstacle_mf_3d": None,
            "obstacle_mf_flat": None,
            "target_mf_3d": None,
            "target_mf_flat": None
        }

    def _setup_mf_oracle(self, **kwargs):
        """Initialize the oracle used for rewards and transitions."""
        return GridBattleMFOracle(num_populations=self._num_population, num_agents_dict=self._num_agents_dict,
                                  team_structure_dict={"blue": ["blue"], "red": ["red"]},
                                  grid_world_size=self.grid_size, action_vectors_dict=self._action_map_dict, **kwargs)

    def _random_init_agents(self):
        """
        Randomly initialize the agents for each population using various strategies.
        """
        self.mf_oracle.reset_reward()
        
        self.grid_indices_flat = np.arange(self.grid_size[0] * self.grid_size[1])
        init_options = ["random", "split", "concentrated", "sparse"]
        prob_init_options = [0.1, 0.2, 0.2, 0.5]
        chosen_init = np.random.choice(init_options, p=prob_init_options)

        shared_blob = self.np_random.random() < 0.05
        if shared_blob:
            blob_location = self.np_random.integers(0, self.grid_size[0], size=2, dtype=int)
            while self._map_dict[list(self._num_agents_dict.keys())[0]][blob_location[0], blob_location[1]] != 0:
                blob_location = self.np_random.integers(0, self.grid_size[0], size=2, dtype=int)

        for population_id in self._num_agents_dict.keys():
            init_dist_grid = self._sample_init_mean_fields()
            valid_locations = np.argwhere(self._map_dict[population_id] == 0)[:, :2]
            num_agents = self._num_agents_dict[population_id]

            if chosen_init == "concentrated":
                if not shared_blob:
                    blob_location = self.np_random.integers(0, self.grid_size[0], size=2, dtype=int)
                    while self._map_dict[population_id][blob_location[0], blob_location[1]] != 0:
                        blob_location = self.np_random.integers(0, self.grid_size[0], size=2, dtype=int)
                agent_location = blob_location
                agent_status = 1

            if chosen_init == "split":
                num_splits = np.random.choice([1, 2], p=[0.3, 0.7])
                selected_indices = self.np_random.choice(len(valid_locations), size=num_splits, replace=False)
                init_split_grid = valid_locations[selected_indices].tolist()
                agent_status = 1

            if chosen_init == "sparse":
                num_zero = np.random.randint(5, 12)  # TODO: fix this
                init_split_prob = np.zeros(len(valid_locations))
                non_zero_indices = np.random.choice(len(valid_locations), len(valid_locations) - num_zero,
                                                    replace=False)
                probability_value = 1 / (len(valid_locations) - num_zero)
                init_split_prob[non_zero_indices] = probability_value

                num_deactivated_loc = np.random.randint(1, len(non_zero_indices) + 1)  # At least one index
                selected_deactivated_indices = np.random.choice(non_zero_indices, num_deactivated_loc, replace=False)
                deactivated_loc = valid_locations[selected_deactivated_indices]

            for agent_index in range(num_agents):
                agent_id = "agent_{}".format(agent_index)

                if chosen_init == "random":
                    agent_location = self._sample_init_agent_locations(init_dist_grid)
                    agent_status = np.random.binomial(1, 0.9)
                    while self._map_dict[population_id][agent_location[0], agent_location[1]] != 0:
                        agent_location = self._sample_init_agent_locations(init_dist_grid)

                if chosen_init == "split":
                    probs = np.random.dirichlet(np.ones(len(init_split_grid)))
                    random_index = np.random.choice(len(init_split_grid), p=probs)
                    agent_location = init_split_grid[random_index]

                if chosen_init == "sparse":
                    random_index = np.random.choice(len(valid_locations), p=init_split_prob)
                    agent_location = valid_locations[random_index]
                    agent_status = 1
                    if np.random.rand() < 0.05:
                        if agent_location in deactivated_loc:
                            agent_status = 0

                self.agent_info[population_id][agent_id]["current-location"] = agent_location
                self.agent_info[population_id][agent_id]["current-alive"] = bool(agent_status)
                self.agent_info[population_id][agent_id]["current-state"] = \
                    self.mf_oracle.status2state([(agent_location, agent_status)], population_id=population_id)[0]
                self.agent_info[population_id][agent_id]["current-target_reached"] = False

    def _sample_init_mean_fields(self):
        """Sample a mean-field distribution for initialization purposes."""
        probability_grid = np.random.rand(self.grid_size[0], self.grid_size[1])
        probability_grid /= np.sum(probability_grid)
        return probability_grid

    def _sample_init_agent_locations(self, probability_grid):
        """Sample an agent location for initialization purposes."""
        _flat_prob_grid = probability_grid.flatten()
        agent_loc = np.random.choice(self.grid_indices_flat, p=_flat_prob_grid)
        return np.array(divmod(agent_loc, self.grid_size[0]))

    def render_extras(self, canvas):
        """Draw scenario-specific additions to the rendered grid."""
        pass

    def render_env_extras(self, canvas):
        """Draw scenario-specific environment details."""
        pass

    def _check_agent_termination(self) -> dict[str, dict[str, bool]]:
        """
        Returns a dictionary marking agents as done based on `alive` or if target reached.
        """
        done_dict = {}
        for population_id in self._num_agents_dict.keys():
            done_dict[population_id] = {}
            for agent_index in range(self._num_agents_dict[population_id]):
                agent_info = self.agent_info[population_id]["agent_{}".format(agent_index)]
                if not agent_info["current-alive"] or agent_info["current-target_reached"]:
                    done_dict[population_id]["agent_{}".format(agent_index)] = True
                else:
                    done_dict[population_id]["agent_{}".format(agent_index)] = False
        return done_dict


class MFVirusEnv(BaseMFGridEnv):
    """
    Mean-field virus environment with finite population.
    """

    def __init__(self, render_mode=None,
                 grid_size: tuple[int] = (8, 8),
                 identical_grid_world=True,
                 num_population: int = 2,
                 num_agent_dict: dict[str, int] = {"blue": 2, "red": 2},
                 action_map_dict={"blue": [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)],
                                  "red": [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]},
                 population_color_dict: dict[str, tuple] = {"blue": (0, 0, 255), "red": (255, 0, 0)},
                 n_obstacles_dict={"blue": 0, "red": 0},
                 obstacle_locations_dict={"blue": [], "red": []},
                 random_init_obstacles_dict={"blue": False, "red": False},
                 penetrable_obstacles_dict={"blue": False, "red": False},
                 n_targets_dict={"blue": 0, "red": 0},
                 target_locations_dict={"blue": [], "red": []},
                 random_init_targets_dict={"blue": False, "red": False},
                 n_hospitals_dict: dict[str, int] = {"blue": 0},
                 hospital_locations_dict: dict[str, Union[List[tuple], None]] = {"blue": None},
                 oracle_kwargs={}, **kwargs):

        """
        Initializes the finite-agent grid environment.
        """

        kwargs = kwargs | {"identical_grid_world": identical_grid_world,
                           "penetrable_obstacles": penetrable_obstacles_dict}
        self._reset_file_counter = 0

        oracle_kwargs = {
            "n_hospitals_dict": n_hospitals_dict,
            "hospital_locations_dict": hospital_locations_dict,
            "n_obstacles_dict": n_obstacles_dict,
            "obstacle_locations_dict": obstacle_locations_dict,
        }

        super().__init__(render_mode=render_mode,
                         grid_size=grid_size,
                         num_population=num_population,
                         num_agent_dict=num_agent_dict,
                         action_map_dict=action_map_dict,
                         population_color_dict=population_color_dict,
                         n_obstacles_dict=n_obstacles_dict,
                         obstacle_locations_dict=obstacle_locations_dict,
                         random_init_obstacles_dict=random_init_obstacles_dict,
                         penetrable_obstacles_dict=penetrable_obstacles_dict,
                         n_targets_dict=n_targets_dict,
                         target_locations_dict=target_locations_dict,
                         random_init_targets_dict=random_init_targets_dict,
                         oracle_kwargs=oracle_kwargs, **kwargs)

    def _update_agent_info(self, actions):
        """
        Executes a step of agent transitions based on the sampled actions and oracle-defined population transitions.
        Updates location, state, alive status, and target reach info.
        """
        population_trans_prob = self.mf_oracle.get_population_transition_probability(population_id_list=None,
                                                                                     mf_info=self.mf_info)
        for population_id in self._num_agents_dict.keys():
            for agent_index in range(self._num_agents_dict[population_id]):
                agent_id = "agent_{}".format(agent_index)
                action = actions[population_id][agent_id]
                state = self.agent_info[population_id][agent_id]["current-state"]
                probs = population_trans_prob[population_id][:, state, action]

                new_state = np.random.choice(list(range(len(probs))), p=probs)
                new_pos, alive = self.mf_oracle.state2status(state_list=[new_state], population_id=population_id)[0]

                reached_target = new_pos in self._target_locations_dict[population_id]
                self.agent_info[population_id][agent_id]["old-location"] = copy(
                    self.agent_info[population_id][agent_id]["current-location"])
                self.agent_info[population_id][agent_id]["old-alive"] = copy(
                    self.agent_info[population_id][agent_id]["current-alive"])
                self.agent_info[population_id][agent_id]["old-state"] = copy(
                    self.agent_info[population_id][agent_id]["current-state"])
                self.agent_info[population_id][agent_id]["old-target_reached"] = copy(
                    self.agent_info[population_id][agent_id]["current-target_reached"])
                self.agent_info[population_id][agent_id]["current-location"] = new_pos
                self.agent_info[population_id][agent_id]["current-alive"] = 1 if population_id == "red" else alive
                self.agent_info[population_id][agent_id]["current-state"] = new_state
                self.agent_info[population_id][agent_id]["current-target_reached"] = reached_target

    def _get_obs(self):
        """
        Constructs observation dictionary with:
        - Local location map per agent
        - Shared obstacle/target map (optional)
        - Global mean-field information (per population)
        """
        obs = {}
        mf_obs = {"mean-field-{}".format(pop_id):
                      self.mf_info[pop_id]["mf_3d"] for pop_id in self._num_agents_dict.keys()}
        for population_id in self._num_agents_dict.keys():
            obs[population_id] = {"shared-team-obs": {}}
            for agent_index in range(self._num_agents_dict[population_id]):
                agent_location = self.agent_info[population_id]["agent_{}".format(agent_index)]["current-location"]
                agent_alive = self.agent_info[population_id]["agent_{}".format(agent_index)]["current-alive"]
                location_map = np.zeros(self.grid_size + (1,), dtype=np.float32)
                if agent_location is not None:
                    location_map[
                        agent_location[0], agent_location[1], 0] = agent_alive * 2 - 1  # 1 for alive, -1 for dead
                local_obs = {"location-map": location_map}
                obs[population_id]["agent_{}-local-obs".format(agent_index)] = local_obs
            obs[population_id]["shared-team-obs"]["obstacle-target-map"] = np.zeros(self.grid_size + (1,),
                                                                                    dtype=np.float32)  # self._map_dict[population_id]
        obs["global-obs"] = mf_obs
        return obs

    def flatten_full_observation(self, obs):
        """
        Flatten the observation dictionary into vector-based observations.
        """
        obs_ = {
                   population_id:

                       {
                           "agent_{}-local-obs".format(idx): flatten_obs(
                               obs[population_id]["agent_{}-local-obs".format(idx)])
                           for idx in range(self._num_agents_dict[population_id])
                       } |
                       {
                           "shared-team-obs": flatten_obs(obs[population_id]["shared-team-obs"])
                       }

                   for population_id in self._num_agents_dict.keys()
               } | {
                   "global-obs": flatten_obs(obs["global-obs"])
               }
        return obs_

    def _get_info(self):
        """
        Returns metadata including termination flags and current timestep.
        """
        terminal_obs = self._get_obs()
        terminal_global_obs = terminal_obs["global-obs"]
        if all(value is not None for value in terminal_global_obs.values()):
            terminal_global_obs = flatten_obs(terminal_global_obs)
            terminal_obs = self.flatten_full_observation(terminal_obs)

        info = {
            "agent_locations":
                {
                    population_id:
                        {
                            agent_id: self.agent_info[population_id][agent_id]["current-location"]
                            for agent_id in self.agent_info[population_id]
                        }
                    for population_id in self.population_id_list
                },
            "target_locations": self._target_locations_dict,
            "terminal_mean_field": terminal_global_obs,
            "terminal_full_state": terminal_obs,
            "time_step": self._time_step,
            "done_dict": self._check_agent_termination()
        }
        return info

    def _setup_observation_space(self):
        '''
        local-obs: position of the agent represented as a 2D array,
                   where +1 represents the agent alive and -1 represents the agent dead
        global-obs: mean-field information of all populations
                    the obstacles locations and target locations, where obstacles are +1 and targets are -1
                    (the map of obstacles and targets is optional)
        '''
        for population_id in self._num_agents_dict.keys():
            for agent_index in range(self._num_agents_dict[population_id]):
                self.observation_space[population_id]["agent_{}-local-obs".format(agent_index)] = spaces.Dict(
                    {
                        "location-map":
                            spaces.Box(
                                low=-1, high=1, shape=(self.grid_size[0], self.grid_size[1], 1), dtype=np.float32
                            )
                    })
            self.observation_space[population_id]["shared-team-obs"] = spaces.Dict(
                {
                    "obstacle-target-map": spaces.Box(low=-1, high=1, shape=(self.grid_size[0], self.grid_size[1], 1)),
                }
            )
        self.observation_space["global-obs"] = spaces.Dict(
            {
                "mean-field-blue": spaces.Box(low=-1, high=1,
                                                           shape=(self.grid_size[0], self.grid_size[1], 2)),
                "mean-field-red": spaces.Box(low=-1, high=1,
                                                           shape=(self.grid_size[0], self.grid_size[1], 1))
            }
        )

    def _init_observations(self):
        """Return initial global observation composed of local + common information."""
        for population_id in self._num_agents_dict.keys():
            for agent_index in range(self._num_agents_dict[population_id]):
                agent_id = "agent_{}".format(agent_index)
                self._observation_dict[population_id][agent_id]["local-obs"] = np.zeros(self.size + (1,))
                for pop_id in self._num_agents_dict.keys():
                    mf_index = "mean-fields-{}".format(pop_id)
                    self._observation_dict[population_id][agent_id]["global-obs"][mf_index] = self.mf_info[pop_id][
                        "mf_3d"]
                self._observation_dict[population_id][agent_id]["global-obs"]["obstacle-target-map"] = self._map_dict[
                    population_id]

    def _setup_action_space(self):
        '''
        Each agent's action is listed in self._action_map
        '''
        for population_id in self.population_id_list:
            self.action_space[population_id] = spaces.Discrete(len(self._action_map_dict[population_id]))

    def _init_agent_info(self, population_id, agent_id):
        """
        Returns default dictionary for an agent's tracked properties.
        """
        return {
            "old-location": None,
            "old-alive": True,
            "old-state": None,
            "old-target_reached": False,
            "current-location": None,
            "current-alive": True,
            "current-state": None,
            "current-target_reached": False,
            "rendered-position": None,
            "rendered-color": None,
        }

    def _init_mf_info(self):
        """Return initial mean-field information dict."""
        return {key: self._init_population_mf_info() for key in self.population_id_list}

    def _init_population_mf_info(self):
        """Return initial empty mean-field information dict."""
        return {
            "mf_3d": None,
            "mf_flat": None,
        }

    def _init_env_info(self):
        """Return initial environment information."""
        return {population_id: self._init_population_env_info() for population_id in self.population_id_list}

    def _init_population_env_info(self):
        """Return initial popualtion information."""
        return {
            "obstacle_mf_3d": None,
            "obstacle_mf_flat": None,
            "target_mf_3d": None,
            "target_mf_flat": None
        }

    def _setup_mf_oracle(self, **kwargs):
        """Initialize the oracle used for rewards and transitions."""
        return VirusMFOracle(num_populations=self._num_population, num_agents_dict=self._num_agents_dict,
                             team_structure_dict={"blue": ["blue"], "red": ["red"]},
                             grid_world_size=self.grid_size, action_vectors_dict=self._action_map_dict, **kwargs)

    def _random_init_agents(self):
        """
        Randomly initialize the agents for each population using various strategies.
        """

        self.mf_oracle.reset_reward()
        
        self.grid_indices_flat = np.arange(self.grid_size[0] * self.grid_size[1])
        init_options = ["random", "split", "concentrated", "sparse"]
        prob_init_options = [0.1, 0.2, 0.2, 0.5]
        chosen_init = np.random.choice(init_options, p=prob_init_options)

        for population_id in self._num_agents_dict.keys():
            init_dist_grid = self._sample_init_mean_fields()
            valid_locations = np.argwhere(self._map_dict[population_id] == 0)[:, :2]
            num_agents = self._num_agents_dict[population_id]

            if chosen_init == "concentrated":
                blob_location = self.np_random.integers(0, self.grid_size[0], size=2, dtype=int)
                while self._map_dict[population_id][blob_location[0], blob_location[1]] != 0:
                    blob_location = self.np_random.integers(0, self.grid_size[0], size=2, dtype=int)
                agent_location = blob_location
                agent_status = 1

            if chosen_init == "split":
                num_splits = np.random.choice([1, 2], p=[0.3, 0.7])
                selected_indices = self.np_random.choice(len(valid_locations), size=num_splits, replace=False)
                init_split_grid = valid_locations[selected_indices].tolist()
                agent_status = 1

            if chosen_init == "sparse":
                num_zero = np.random.randint(5, 12)  # TODO: fix this
                init_split_prob = np.zeros(len(valid_locations))
                non_zero_indices = np.random.choice(len(valid_locations), len(valid_locations) - num_zero,
                                                    replace=False)
                probability_value = 1 / (len(valid_locations) - num_zero)
                init_split_prob[non_zero_indices] = probability_value

                num_deactivated_loc = np.random.randint(1, len(non_zero_indices) + 1)  # At least one index
                selected_deactivated_indices = np.random.choice(non_zero_indices, num_deactivated_loc, replace=False)
                deactivated_loc = valid_locations[selected_deactivated_indices]

            for agent_index in range(num_agents):
                agent_id = "agent_{}".format(agent_index)

                if chosen_init == "random":
                    agent_location = self._sample_init_agent_locations(init_dist_grid)
                    if population_id == "red":
                        agent_status = 1
                    else:
                        agent_status = np.random.binomial(1, 0.9)

                    while self._map_dict[population_id][agent_location[0], agent_location[1]] != 0:
                        agent_location = self._sample_init_agent_locations(init_dist_grid)

                if chosen_init == "split":
                    probs = np.random.dirichlet(np.ones(len(init_split_grid)))
                    random_index = np.random.choice(len(init_split_grid), p=probs)
                    agent_location = init_split_grid[random_index]

                if chosen_init == "sparse":
                    random_index = np.random.choice(len(valid_locations), p=init_split_prob)
                    agent_location = valid_locations[random_index]
                    agent_status = 1
                    if np.random.rand() < 0.05:
                        if agent_location in deactivated_loc and population_id == "blue":
                            agent_status = 0

                self.agent_info[population_id][agent_id]["current-location"] = agent_location
                self.agent_info[population_id][agent_id]["current-alive"] = bool(agent_status)
                self.agent_info[population_id][agent_id]["current-state"] = \
                    self.mf_oracle.status2state([(agent_location, agent_status)], population_id=population_id)[0]
                self.agent_info[population_id][agent_id]["current-target_reached"] = False
   
    def _sample_init_mean_fields(self):
        """Sample a mean-field distribution for initialization purposes."""
        probability_grid = np.random.rand(self.grid_size[0], self.grid_size[1])
        probability_grid /= np.sum(probability_grid)
        return probability_grid

    def _sample_init_agent_locations(self, probability_grid):
        """Sample an agent location for initialization purposes."""
        _flat_prob_grid = probability_grid.flatten()
        agent_loc = np.random.choice(self.grid_indices_flat, p=_flat_prob_grid)
        return np.array(divmod(agent_loc, self.grid_size[0]))

    def render_extras(self, canvas):
        """Draw scenario-specific additions to the rendered grid."""
        pass

    def render_env_extras(self, canvas):
        """Draw hospital locations with a clear medical cross marker."""
        hospital_locations = getattr(self.mf_oracle, "_hospital_locations_dict", {}) or {}
        cell_width, cell_height = self._pix_square_size
        for locations in hospital_locations.values():
            for row, column in locations or []:
                rect = pygame.Rect(
                    int(row * cell_width) + 7,
                    int(column * cell_height) + 7,
                    max(1, int(cell_width) - 14),
                    max(1, int(cell_height) - 14),
                )
                pygame.draw.rect(canvas, (255, 237, 232), rect, border_radius=12)
                center_x = int((row + 0.5) * cell_width)
                center_y = int((column + 0.5) * cell_height)
                arm = max(8, int(min(cell_width, cell_height) * 0.19))
                thickness = max(4, int(min(cell_width, cell_height) * 0.08))
                pygame.draw.rect(
                    canvas,
                    (220, 72, 72),
                    pygame.Rect(center_x - thickness // 2, center_y - arm, thickness, 2 * arm),
                    border_radius=3,
                )
                pygame.draw.rect(
                    canvas,
                    (220, 72, 72),
                    pygame.Rect(center_x - arm, center_y - thickness // 2, 2 * arm, thickness),
                    border_radius=3,
                )

    def _check_agent_termination(self) -> dict[str, dict[str, bool]]:
        """
        Returns a dictionary marking agents as done based on `alive` or if target reached.
        """
        done_dict = {}
        for population_id in self._num_agents_dict.keys():
            done_dict[population_id] = {}
            for agent_index in range(self._num_agents_dict[population_id]):
                agent_info = self.agent_info[population_id]["agent_{}".format(agent_index)]
                if not agent_info["current-alive"] or agent_info["current-target_reached"]:
                    done_dict[population_id]["agent_{}".format(agent_index)] = True
                else:
                    done_dict[population_id]["agent_{}".format(agent_index)] = False
        return done_dict
