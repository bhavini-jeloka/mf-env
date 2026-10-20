import numpy as np
from typing import Union, List
from gym_mf_envs.envs.base_env.base_oracle import BaseMFOracle
from gym_mf_envs.envs.grid_envs.utils.grid_world_simulator import GridWorldSimulator


class GridMFOracle(BaseMFOracle):
    """
    Oracle class for mean-field grid environments that handles different populations and
    their dynamics.
    """
    def __init__(self, num_populations: int, num_agents: dict[str, int],
                 action_vectors: dict[str, List[Union[tuple, np.ndarray]]],
                 grid_world_size: tuple,
                 identical_grid_world: bool = False,
                 penetrable_obstacles: Union[dict[str, bool], bool] = False,
                 **kwargs):
        
        """
        Initialize the GridMFOracle.

        Args:
            num_populations (int): Number of populations.
            num_agents (dict): Number of agents per population.
            action_vectors (dict): Action vectors per population.
            grid_world_size (tuple): Size of the grid world.
            identical_grid_world (bool): If True, all populations share the same map.
            penetrable_obstacles (dict or bool): Indicates whether obstacles are penetrable per population.
            kwargs: Additional arguments like time_penalty.
        """

        self._identical_grid_world_simulator = identical_grid_world
        self._grid_world_simulator_dict = {key: GridWorldSimulator(grid_world_size, penetrable_obstacles[key])
                                           for key in penetrable_obstacles.keys()}

        self.action_vectors = action_vectors

        # Compute number of actions and states for each population
        num_actions = {population_id: len(action_vectors[population_id]) for population_id in action_vectors.keys()}
        num_states = {population_id: self.get_population_grid_world_simulator(population_id).get_n_states()
                      for population_id in num_agents.keys()}
        
        # Initialize base oracle class
        super().__init__(num_populations, num_agents, num_states, num_actions, **kwargs)

        # Set time penalty per population
        self._time_penalty = kwargs.get("time_penalty", {pop_id: -1 for pop_id in num_agents.keys()})

        self._obstacle_pos, self._num_obstacles = None, None
        self._target_pos, self._target_pos_states, self._num_targets = None, None, None

    def get_population_grid_world_simulator(self, population_id) -> GridWorldSimulator:
        """
        Get the GridWorldSimulator for a specific population.
        """
        return self._grid_world_simulator_dict[population_id]

    def reset(self, **kwargs):
        """
        Reset the oracle with new grid maps.
        """
        maps = kwargs.get("grid_world_map_list")
        assert maps is not None
        for population_id in self._num_agents.keys():
            self.get_population_grid_world_simulator(population_id).reset(new_map=maps[population_id])
        self._num_states = self.get_num_states_dict()
        self._parse_maps(maps)

    def get_num_states_dict(self):
        """
        Returns the number of states for each population.
        """
        return {population_id: self.get_population_grid_world_simulator(population_id).get_n_states() for
                population_id in self.population_id_list}

    def _parse_maps(self, maps):
        """
        Parse obstacle and target positions from the grid maps.
        """
        self._obstacle_pos, self._num_obstacles = {}, {}
        self._target_pos, self._target_pos_states, self._num_targets = {}, {}, {}
        for population_id in self._num_agents.keys():
            map = maps[population_id]
            obstacle_pos_list = [(i, j) for i in range(len(map)) for j in range(len(map[0])) if map[i][j] == 1]
            self._obstacle_pos[population_id] = obstacle_pos_list
            self._num_obstacles[population_id] = len(obstacle_pos_list)

            target_locations = [(i, j) for i in range(len(map)) for j in range(len(map[0])) if map[i][j] == -1]
            self._target_pos[population_id] = target_locations
            self._target_pos_states[population_id] = self.get_population_grid_world_simulator(
                population_id).get_pos_state_list(target_locations)
            self._num_targets[population_id] = len(target_locations)




