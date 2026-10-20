import numpy as np
from gymnasium import spaces
from typing import List, Union
from gym_mf_envs.envs.grid_envs.envs.base_grid_env import BaseMFGridEnv
from gym_mf_envs.envs.grid_envs.oracles.cooperative_team_oracles import GridNavMFOracle
from gym_mf_envs.utils import flat_dict_values
from gym_mf_envs.utils import opacity
from copy import copy
import os
import re

class MFGridNavEnv(BaseMFGridEnv):
    """
    Cooperative mean-field grid navigation environment with multiple populations.

    Agents in each population navigate a grid world toward their respective targets
    while avoiding obstacles. The environment supports mean-field observations and
    cooperative learning structures.
    """
       
    def __init__(self, render_mode=None,
                 grid_size: tuple[int] = (8, 8),
                 identical_grid_world=False,
                 num_population: int = 2,
                 num_agent_dict: dict[str, int] = {"blue": 2, "red": 2},
                 population_color_dict: dict[str, tuple] = {"blue": (0, 0, 255), "red": (255, 0, 0)},
                 n_obstacles_dict={"blue": 10, "red": 10},
                 obstacle_locations_dict: dict[str, Union[List[tuple], None]] = {"blue": None},
                 random_init_obstacles_dict={"blue": True, "red": True},
                 penetrable_obstacles_dict={"blue": False, "red": False},
                 n_targets_dict={"blue": 2, "red": 2},
                 target_locations_dict={"blue": None, "red": None},
                 random_init_targets_dict={"blue": True, "red": True},
                 action_map_dict={"blue": [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)],
                                  "red": [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]}):
        
        """
        Initializes the finite-agent grid environment.
        """

        self._reset_file_counter = 0
        super().__init__(render_mode=render_mode,
                         grid_size=grid_size,
                         identical_grid_world=identical_grid_world,
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
                         oracle_kwargs={})

    def _update_agent_info(self, actions):
        """
        Executes a step of agent transitions based on the sampled actions and oracle-defined population transitions.
        Updates location, state, alive status, and target reach info.
        """

        for population_id in self._num_agents_dict.keys():
            for agent_index in range(self._num_agents_dict[population_id]):
                agent_id = "agent_{}".format(agent_index)
                to_states, probs = self.mf_oracle.get_agent_transition_probability_action(
                    population_id, agent_id, actions[population_id][agent_id],
                    self.agent_info, self.mf_info
                )
                new_state = np.random.choice(to_states, p=probs)
                new_pos = self.mf_oracle.get_population_grid_world_simulator(population_id).get_pos(new_state)

                alive = True
                reached_target = new_pos in self._target_locations_dict[population_id]
                self.agent_info[population_id][agent_id]["old-location"] = copy(self.agent_info[population_id][agent_id]["current-location"])
                self.agent_info[population_id][agent_id]["old-alive"] = copy(self.agent_info[population_id][agent_id]["current-alive"])
                self.agent_info[population_id][agent_id]["old-state"] = copy(self.agent_info[population_id][agent_id]["current-state"])
                self.agent_info[population_id][agent_id]["old-target_reached"] = copy(self.agent_info[population_id][agent_id]["current-target_reached"])
                self.agent_info[population_id][agent_id]["current-location"] = new_pos
                self.agent_info[population_id][agent_id]["current-alive"] = alive
                self.agent_info[population_id][agent_id]["current-state"] = new_state
                self.agent_info[population_id][agent_id]["current-target_reached"] = reached_target

    def _get_obs(self):
        """Build and return the current observation structure."""
        obs = {}
        mf_obs = {"mean-field-{}".format(pop_id):
                      self.mf_info[pop_id]["mf_3d"] for pop_id in self._num_agents_dict.keys()}
        for population_id in self._num_agents_dict.keys():
            obs[population_id] = {"shared-team-obs": {}}
            for agent_index in range(self._num_agents_dict[population_id]):
                agent_location = self.agent_info[population_id]["agent_{}".format(agent_index)]["current-location"]
                location_map = np.zeros(self.grid_size + (1,), dtype=np.float32)
                if agent_location is not None:
                    location_map[agent_location[0], agent_location[1], 0] = 1.0
                local_obs = {"location-map": location_map}
                obs[population_id]["agent_{}-local-obs".format(agent_index)] = local_obs
            obs[population_id]["shared-team-obs"]["obstacle-target-map"] = self._map_dict[population_id]
        obs["global-obs"] = mf_obs
        return obs

    def _get_info(self):
        """
        Constructs observation dictionary with:
        - Local location map per agent
        - Shared obstacle/target map (optional)
        - Global mean-field information (per population)
        """
        info = {
            "agent_locations":
                {
                    population_id:
                        {
                            agent_id: self.agent_info[population_id][agent_id]["current-location"]
                            for agent_id in self.agent_info[population_id]
                        }
                    for population_id in self._num_agents_dict.keys()
                },
            "target_locations": self._target_locations_dict,
            "terminal_mean_field": self._get_obs()["global-obs"]["mean-field-blue"], 
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
                                                           shape=(self.grid_size[0], self.grid_size[1], 1))
                for pop_id in self._num_agents_dict.keys()
            }
        )

    def _setup_action_space(self):
        '''
        Each agent's action is listed in self._action_map
        '''
        for population_id in self._num_agents_dict.keys():
            self.action_space[population_id] = spaces.Discrete(len(self._action_map_dict[population_id]))

    def _init_agent_info(self, population_id, agent_id):
        """
        Returns default dictionary for an agent’s tracked properties.
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
        return {key: self._init_population_mf_info() for key in self._num_agents_dict.keys()}

    def _init_population_mf_info(self):
        """Return initial empty mean-field information dict."""
        return {
            "mf_3d": None,
            "mf_flat": None,
        }

    def _init_env_info(self):
        """Return initial environment information."""
        return {population_id: self._init_population_env_info() for population_id in self._num_agents_dict.keys()}

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
        return GridNavMFOracle(num_populations=self._num_population, num_agents=self._num_agents_dict,
                               grid_world_size=self.grid_size, action_vectors=self._action_map_dict, **kwargs)

    def _random_init_agents(self):
        """
        Randomly initialize the agents for each population using various strategies.
        """
        self.mf_oracle.reset_reward()

        self.grid_indices_flat = np.arange(self.grid_size[0]*self.grid_size[1])
        init_options = ["separated", "blob"]
        prob_init_options = [0.8, 0.2]
        chosen_init = np.random.choice(init_options, p=prob_init_options)
        
        for population_id in self._num_agents_dict.keys():
            init_dist_grid = self._sample_init_mean_fields()
            num_agents = self._num_agents_dict[population_id]
            
            if chosen_init=="blob":
                blob_location = self.np_random.integers(0, self.grid_size[0], size=2, dtype=int)
                while self._map_dict[population_id][blob_location[0], blob_location[1]] != 0:
                    blob_location = self.np_random.integers(0, self.grid_size[0], size=2, dtype=int)
                agent_location=blob_location
                
            for agent_index in range(num_agents):
                agent_id = "agent_{}".format(agent_index)
                
                if chosen_init=="separated":
                    agent_location =  self._sample_init_agent_locations(init_dist_grid)
                    while self._map_dict[population_id][agent_location[0], agent_location[1]] != 0:
                        agent_location =  self._sample_init_agent_locations(init_dist_grid)
                
                self.agent_info[population_id][agent_id]["current-location"] = agent_location
                self.agent_info[population_id][agent_id]["current-alive"] = True
                self.agent_info[population_id][agent_id]["current-state"] = self.mf_oracle.get_population_grid_world_simulator(
                    population_id).get_pos_state(agent_location)
                self.agent_info[population_id][agent_id]["current-target_reached"] = False

    def _sample_init_mean_fields(self):
        """Sample a mean-field distribution for initialization purposes."""
        probability_grid = np.random.rand(self.grid_size[0], self.grid_size[1])
        probability_grid /= np.sum(probability_grid)
        return probability_grid
    
    def _sample_init_agent_locations(self, probability_grid):
        """Sample an agent location for initialization purposes."""
        _flat_prob_grid = probability_grid.flatten()
        agent_loc = np.random.choice(self.grid_indices_flat, p = _flat_prob_grid)
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
                agent_id = "agent_{}".format(agent_index)
                done_dict[population_id][agent_id] = self.agent_info[population_id][agent_id]["current-target_reached"]
        return done_dict
