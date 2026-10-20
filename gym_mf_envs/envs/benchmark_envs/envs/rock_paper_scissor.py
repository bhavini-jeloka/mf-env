from typing import Any
import numpy as np
import gymnasium as gym
import pygame
from gymnasium import spaces
import imageio
from pathlib import Path
import random

from gym_mf_envs.envs.base_env.base_env import BaseMFEnv
from gym_mf_envs.envs.benchmark_envs.oracles import RPSBenchmarkOracle
from gym_mf_envs.utils import flat_dict_values
from copy import copy

class RPSMFEnv(BaseMFEnv):

    """
    A finite population constrained Rock-Paper-Scissors Environment with mean-field interactions.
    Builds upon BaseMFEnv to simulate agent interactions over discrete states using 
    Rock-Paper-Scissors derived transitions.
    """

    def __init__(self, render_mode=None, num_population: int = 2, 
                 num_agent_dict: dict[str, int] = {"blue": 10, "red": 10},
                 population_color_dict: dict[str, int] = {"blue": (0, 0, 255), "red": (255, 0, 0)},
                 num_states: dict[str, int] = {"blue": 3, "red": 3}, 
                 num_actions: dict[str, int] = {"blue": 2, "red": 2},
                 agent_radius: dict[str, int] = {"blue": 4, "red": 5}):
        
        """
        Initialize RPS Mean-Field Environment with finite agents.
        """
        
        self.window_size = 512

        self._num_population = num_population
        self._num_agents_dict = num_agent_dict
        self._population_color_dict = population_color_dict
        self._num_states = num_states
        self._num_actions = num_actions
        self._agent_radius_dict = agent_radius
        self.frames = None

        # render mode
        self.render_mode = render_mode

        super().__init__(render_mode, num_population, num_agent_dict)

    def demo(self, **kwargs):
        """Run the environment’s built-in demonstration."""
        pass

    def _get_obs(self):
        """
        Generate structured observations:
        - local one-hot state for each agent
        - global mean-field distribution for all populations (represented as a vector)
        """
        obs = {}
        mf_obs = {"mean-field-{}".format(pop_id):
                      self.mf_info[pop_id]["mf_flat"] for pop_id in self._num_agents_dict.keys()}
        for population_id in self._num_agents_dict.keys():
            obs[population_id] = {}
            for agent_index in range(self._num_agents_dict[population_id]):
                location_map = np.zeros((1, self._num_states[population_id], 1))
                agent_location = self.agent_info[population_id]["agent_{}".format(agent_index)]["current-state"]
                location_map[0][agent_location][0] = 1
                obs[population_id]["agent_{}-local-obs".format(agent_index)] = location_map
        
        obs["global-obs"] = np.stack((np.array([mf_obs["mean-field-blue"]]), np.array([mf_obs["mean-field-red"]])), axis=-1)
        return obs

    def _get_info(self):
        """
        Generate auxiliary info including current state of the agent, time step, global observations and terminal flags.
        """
        info = {
            "agent_locations":
                {
                    population_id:
                        {
                            agent_id: self.agent_info[population_id][agent_id]["current-state"]
                            for agent_id in self.agent_info[population_id]
                        }
                    for population_id in self.population_id_list
                },
            "time_step": self._time_step,
            "terminal_state": self._get_obs()["global-obs"],
            "terminal_full_state": self._get_obs(),
            "done_dict": self._check_agent_termination()
        }
        return info

    def _update_agent_info(self, actions):
        """
        Given actions, update agent states according to transition probabilities
        derived from the mean-field oracle.
        """
        population_trans_prob = self.mf_oracle.get_population_transition_probability(population_id_list=None,
                                                                                     mf_info=self.mf_info)
        
        for population_id in self._num_agents_dict.keys():
            for agent_index in range(self._num_agents_dict[population_id]):
                agent_id = "agent_{}".format(agent_index)
                action = actions[population_id][agent_id]
                state = self.agent_info[population_id][agent_id]["current-state"]
                probs = population_trans_prob[population_id][action][state]
                to_states = np.arange(self._num_states[population_id])

                new_state = np.random.choice(to_states, p=probs)

                self.agent_info[population_id][agent_id]["old-state"] = copy(
                    self.agent_info[population_id][agent_id]["current-state"])
                self.agent_info[population_id][agent_id]["current-state"] = new_state

    def _check_agent_termination(self) -> dict[str, dict[str, bool]]:
        """
        All agent episodes are non-terminating in this setup (always False); truncation due to time still holds.
        """
        done_dict = {}
        for population_id in self._num_agents_dict.keys():
            done_dict[population_id] = {}
            for agent_index in range(self._num_agents_dict[population_id]):
                done_dict[population_id]["agent_{}".format(agent_index)] = False
        return done_dict

    def _reset_specific_env_variables(self):
        """Reset environment-specific state for a new episode."""
        pass

    def _random_init_agents(self):
        """
        Initialize agent states randomly.
        """
        init_options = ["random", "fixed"]
        prob_init_options = [0.4, 0.6]
        chosen_init = np.random.choice(init_options, p=prob_init_options)
        
        for population_id in self._num_agents_dict.keys():
            init_dist = self._sample_init_mean_fields(population_id)
            num_agents = self._num_agents_dict[population_id]
            
            if chosen_init=="fixed":
                if population_id=="blue":
                    agent_location=0
                else:
                    agent_location=1
                
            for agent_index in range(num_agents):
                agent_id = "agent_{}".format(agent_index)
                
                if chosen_init=="random":
                    agent_location =  np.random.choice(self._num_states[population_id], p = init_dist)
                    
                self.agent_info[population_id][agent_id]["current-state"] = agent_location
                
    def _sample_init_mean_fields(self, population_id):
        """
        Sample initial mean-field distribution using Dirichlet distribution.

        Args:
            population_id (str): ID of the population.
        """
        probability_vec = np.random.dirichlet(np.ones(self._num_states[population_id]))
        return probability_vec
    
    def _reset_mf_oracles(self):
        """ Reset oracle when the reset environment function is called. """
        self.mf_oracle.reset()

    def _init_agent_info(self, population_id, agent_id):
        """ Initialize per-agent state. """
        return {
            "old-state": None,
            "current-state": None,
            "old-rendered-position": None,
            "new-rendered-position": None,
        }

    def _init_mf_info(self):
        """ Create mean-field storage for each population. """
        return {key: self._init_population_mf_info() for key in self.population_id_list}

    def _init_population_mf_info(self):
        """ Initialize empty mean-field info structure for all populations. """
        return {
            "mf_flat": None,
        }

    def _init_env_info(self):
        """Create initial environment-level information."""
        pass

    def _init_observations(self):
        """
        Initialize observations for each agent.
        """
        for population_id in self._num_agents_dict.keys():
            for agent_index in range(self._num_agents_dict[population_id]):
                agent_id = "agent_{}".format(agent_index)
                self._observation_dict[population_id][agent_id]["local-obs"] = np.zeros((1, self._num_states[population_id]))
                for pop_id in self._num_agents_dict.keys():
                    self._observation_dict[population_id][agent_id]["global-obs"] = self.mf_info[pop_id][
                        "mf_flat"]

    def _setup_mf_oracle(self, **kwargs):
        """
        Construct oracle that determines population transition behavior.
        """
        return RPSBenchmarkOracle(num_populations=self._num_population, num_agents=self._num_agents_dict,
                                  num_states=self._num_states, num_actions=self._num_actions)

    def _setup_observation_space(self):
        """
        Define observation space for all agents that includes the common information (global state).
        """
        for population_id in self._num_agents_dict.keys():
            for agent_index in range(self._num_agents_dict[population_id]):
                self.observation_space[population_id]["agent_{}-local-obs".format(agent_index)] = spaces.Box(
                                low=-1, high=1, shape=(1, self._num_states[population_id], 1), dtype=np.float32
                            )
        self.observation_space["global-obs"] = spaces.Box(low=-1, high=1,
                                                           shape=(1, 3, self._num_population))

    def _setup_action_space(self):
        """
        Define discrete action space per population.
        """
        for population_id in self._num_agents_dict.keys():
            self.action_space[population_id] = spaces.Discrete(self._num_actions[population_id])


