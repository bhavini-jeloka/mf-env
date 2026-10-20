import numpy as np
import pygame
import warnings

from gym_mf_envs.envs.grid_envs.oracles.base_grid_oracle import GridMFOracle
from gym_mf_envs.envs.base_env.base_env import BaseMFEnv
from typing import List, Union
from gym_mf_envs.utils import opacity
from gym_mf_envs.envs.grid_envs.utils.render_utils import place_circle


class BaseMFGridEnv(BaseMFEnv):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30,
    }

    def __init__(self, render_mode=None,
                 grid_size: tuple[int] = (8, 8),
                 identical_grid_world=True,
                 num_population: int = 1,
                 num_agent_dict: dict[str, int] = {"blue": 1, },
                 action_map_dict: dict[str, List[tuple]] = {"blue": [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]},
                 population_color_dict: dict[str, tuple] = {"blue": (0, 0, 255)},
                 n_obstacles_dict: dict[str, int] = {"blue": 2, },
                 obstacle_locations_dict: dict[str, Union[List[tuple], None]] = {"blue": None},
                 random_init_obstacles_dict: dict[str, bool] = {"blue": True},
                 penetrable_obstacles_dict={"blue": False},
                 n_targets_dict: dict[str, int] = {"blue": 2},
                 target_locations_dict: dict[str, Union[List[tuple], None]] = {"blue": None},
                 random_init_targets_dict: dict[str, bool] = {"blue": True},
                 oracle_kwargs: dict = {"rho": 0.5},
                 **kwargs):

        """
        Initializes the base grid environment for mean-field multi-agent systems.

        Each population can have different grid layouts, targets, and obstacles,
        unless specified to be identical.

        :param render_mode:                 render mode, one of ["human", "rgb_array"]
        :param grid_size:                   size of the grid environment
        :param identical_grid_world:        whether the grid world is identical for all populations
        :param num_population:              number of populations
        :param num_agent_dict:              number of agents in each population
        :param action_map_dict:             action map for each population
        :param population_color_dict:       color of each population
        :param n_obstacles_dict:            number of obstacles for each population
        :param obstacle_locations_dict:     locations of obstacles for each population. [] no obstacle; None to be init
        :param random_init_obstacles_dict:  whether to randomly initialize obstacles for each population at each reset
        :param n_targets_dict:              number of targets for each population
        :param target_locations_dict:       locations of targets for each population. [] no target; None to be init
        :param random_init_targets_dict:    whether to randomly initialize targets for each population at each reset
        :param oracle_kwargs:                      additional arguments for the mean field oracle, such as pop_ratio, etc.
        """

        # check inputs
        assert isinstance(grid_size, tuple)
        population_key_list = list(num_agent_dict.keys())
        assert (len(num_agent_dict.keys()) == len(action_map_dict.keys()) == len(population_color_dict.keys())
                == num_population)
        assert all([key in action_map_dict.keys() for key in population_key_list])
        assert all([key in population_color_dict.keys() for key in population_key_list])
        assert isinstance(identical_grid_world, bool)

        # convert non-dictionary inputs to dictionary
        if isinstance(n_obstacles_dict, int):
            n_obstacles_dict = {pop_id: n_obstacles_dict for pop_id in population_key_list}
            warnings.warn("n_obstacles input is a single int value. Assuming the same value for all populations. \n"
                          "If you want different values for different populations, please provide a dictionary.\n"
                          "If you want identical grid world, pleases provide a dictionary with a single key 'identical'.")
        if isinstance(random_init_obstacles_dict, bool):
            random_init_obstacles_dict = {pop_id: random_init_obstacles_dict for pop_id in population_key_list}
            warnings.warn(
                "random_init_obstacles input is a single bool value. Assuming the same value for all populations. \n"
                "If you want different random init settings for different populations, please provide a dictionary.")
        if obstacle_locations_dict is None:
            obstacle_locations_dict = {pop_id: None for pop_id in population_key_list}
            warnings.warn("obstacle_locations input is None. Assuming no obstacles for all populations. \n")
        if isinstance(penetrable_obstacles_dict, bool):
            penetrable_obstacles_dict = {pop_id: penetrable_obstacles_dict for pop_id in population_key_list}
            warnings.warn(
                "penetrable_obstacles input is a single bool value. Assuming the same value for all populations. \n")
        if isinstance(n_targets_dict, int):
            n_targets_dict = {pop_id: n_targets_dict for pop_id in population_key_list}
            warnings.warn("n_targets input is a single int value. Assuming the same value for all populations. \n"
                          "If you want different values for different populations, please provide a dictionary. \n"
                          "If you want identical grid world, pleases provide a dictionary with a single key 'identical'.")
        if isinstance(random_init_targets_dict, bool):
            random_init_targets_dict = {pop_id: random_init_targets_dict for pop_id in population_key_list}
            warnings.warn(
                "random_init_targets input is a single bool value. Assuming the same value for all populations. \n"
                "If you want different random init settings for different populations, please provide a dictionary.")
        if target_locations_dict is None:
            target_locations_dict = {pop_id: None for pop_id in population_key_list}
            warnings.warn("target_locations input is None. Assuming no obstacles for all populations. \n")

        # check identical grid world setting
        # Identical world only applies to obstacles. Targets can still be different.
        # We only keep one copy of the obstacle settings.
        # However, n copies of grid world simulators will be kept to properly simulate different set of targets and terrains.
        if identical_grid_world:
            print("Identical grid world is set to True. All population shares the same grid world (obstacles), "
                  "targets can still be different.")
            identical_obstacle_locations = list(obstacle_locations_dict.values())[0]
            if identical_obstacle_locations is None:
                assert all([obstacle_locations_dict[key] == None for key in population_key_list]), \
                    "Obstacle locations should be the same for all populations under identical grid world setting."
            else:
                identical_obstacle_locations.sort()
                for key in population_key_list:
                    if obstacle_locations_dict[key] is not None:
                        obstacle_locations_dict[key].sort()
                        assert obstacle_locations_dict[key] == identical_obstacle_locations, \
                            "Obstacle locations should be the same for all populations under identical grid world setting."

                    "Obstacle locations should be the same for all populations under identical grid world setting."
            assert all([n == n_obstacles_dict[population_key_list[0]] for n in n_obstacles_dict.values()]), \
                "Number of obstacles should be the same for all populations under identical grid world setting."
            if "identical" in random_init_targets_dict.keys():
                assert (all([random_init_obstacles_dict[key] == random_init_obstacles_dict[population_key_list[0]]
                             for key in population_key_list])), \
                    "Random init obstacles should be the same for all populations under identical grid world setting."

            if "identical" not in n_obstacles_dict.keys():
                n_obstacles_dict = {"identical": n_obstacles_dict[population_key_list[0]]}
            if "identical" not in random_init_obstacles_dict.keys():
                random_init_obstacles_dict = {"identical": random_init_obstacles_dict[population_key_list[0]]}
            if "identical" not in obstacle_locations_dict.keys():
                obstacle_locations_dict = {"identical": obstacle_locations_dict[population_key_list[0]]}

        # check random init settings
        pop_keys = ["identical"] if identical_grid_world else population_key_list
        for pop_id in pop_keys:
            if not random_init_obstacles_dict[pop_id]:
                assert obstacle_locations_dict[
                           pop_id] is not None, "Obstacle locations should be provided if not random init."
        for pop_id in population_key_list:
            if not random_init_targets_dict[pop_id]:
                assert target_locations_dict[
                           pop_id] is not None, "Target locations should be provided if not random init."

        # check input compatability
        assert all(
            [obstacle_locations_dict[key] is None or len(obstacle_locations_dict[key]) == n_obstacles_dict[key] for key
             in pop_keys]), "Number of obstacles should be the same as the length of target obstacles."
        assert all(
            [target_locations_dict[key] is None or len(target_locations_dict[key]) == n_targets_dict[key] for key in
             population_key_list]), "Number of targets should be the same as the length of target locations."

        # check input types
        assert all([isinstance(n, int) for n in n_obstacles_dict.values()]), "Number of obstacles should be integers."
        assert all([isinstance(n, int) for n in n_targets_dict.values()]), "Number of targets should be integers."

        assert all(
            [obstacle_locations_dict[key] is None or isinstance(obstacle_locations_dict[key], list) for key in
             pop_keys]), "obstacle locations of each population should be a list of tuples."
        assert all(
            [target_locations_dict[key] is None or isinstance(target_locations_dict[key], list) for key in
             population_key_list]), "Target locations of each population should be a list of tuples."

        # store inputs
        self.grid_size = grid_size
        self.identical_grid_world = identical_grid_world
        self._num_population, self._num_agents_dict = num_population, num_agent_dict
        self._action_map_dict = action_map_dict
        self.population_color_dict = population_color_dict

        self.n_obstacles_dict, self._obstacle_locations_dict = n_obstacles_dict, obstacle_locations_dict
        self.random_init_obstacles_dict = random_init_obstacles_dict
        self.penetrable_obstacles_dict = penetrable_obstacles_dict

        self.n_targets_dict, self._target_locations_dict = n_targets_dict, target_locations_dict
        self.random_init_targets_dict = random_init_targets_dict

        # setup mean field oracle base type
        self.mf_oracle: GridMFOracle
        oracle_kwargs = self.update_oracle_kwargs(oracle_kwargs)

        # setup maps
        # 1 represents obstacles, -1 represents targets, 0 for free space
        self._map_dict = {}
        if self.identical_grid_world:
            self._map_dict = {"identical_map": np.zeros(self.grid_size + (1,), dtype=np.float32)} | {
                pop_id: np.zeros(self.grid_size + (1,), dtype=np.float32) for pop_id in num_agent_dict.keys()}

        else:
            self._map_dict = {pop_id: np.zeros(self.grid_size + (1,), dtype=np.float32) for pop_id in
                              num_agent_dict.keys()}

        # rendering set up
        self._pix_square_size = None
        self.population_color_list = population_color_dict

        super().__init__(render_mode, num_population, num_agent_dict, oracle_kwargs)

    def _reset_specific_env_variables(self):
        '''
        Reset environment variables like map, obstacles, targets on each episode reset
        '''
        self._reset_maps()
        self._random_init_obstacles()
        self._reset_targets()

    def _reset_maps(self):
        '''
        Reset maps: Since different populations have different targets with identical world, multiple maps are kept.
        If identical world, a single obstacle map is randomly initialized and then copied to individual populations.
        1 represents obstacles, -1 represents targets, 0 for free space
        '''
        self._map_dict = {pop_id: np.zeros(self.grid_size + (1,), dtype=np.float32) for pop_id in
                          self._num_agents_dict.keys()}
        if self.identical_grid_world:
            self._map_dict = self._map_dict | {"identical": np.zeros(self.grid_size + (1,), dtype=np.float32)}

    def _random_init_obstacles(self):
        '''
        Randomly initialize a single obstacle map if identical world, then copy to individual populations
        '''

        if self.identical_grid_world:
            assert "identical" in self._map_dict.keys() and "identical" in self._obstacle_locations_dict.keys() \
                   and "identical" in self.n_obstacles_dict.keys()
            if self.random_init_obstacles_dict["identical"]:
                self._obstacle_locations_dict["identical"] = \
                    self._random_init_obstacle_map(self.n_obstacles_dict["identical"], self._map_dict["identical"])
            for obstacle_loc in self._obstacle_locations_dict["identical"]:
                self._map_dict["identical"][obstacle_loc[0], obstacle_loc[1]] = 1

            for population_id in self._num_agents_dict.keys():
                self._obstacle_locations_dict[population_id] = self._obstacle_locations_dict["identical"]
                self._map_dict[population_id] = self._map_dict["identical"]
        else:
            for population_id in self._num_agents_dict.keys():
                if self.random_init_obstacles_dict[population_id]:
                    self._obstacle_locations_dict[population_id] = self._random_init_obstacle_map(
                        self.n_obstacles_dict[population_id], self._map_dict[population_id])
                for obstacle_loc in self._obstacle_locations_dict[population_id]:
                    self._map_dict[population_id][obstacle_loc[0], obstacle_loc[1]] = 1

    def _random_init_obstacle_map(self, n_obstacles, map):
        '''
        Places n_obstacles at random empty locations in the map
        '''
        obstacle_locations = []
        for _ in range(n_obstacles):
            obstacle_location = None
            while obstacle_location is None or map[obstacle_location[0], obstacle_location[1]] != 0:
                obstacle_location = (self.np_random.integers(0, self.grid_size[0], size=1, dtype=int).item(),
                                     self.np_random.integers(0, self.grid_size[1], size=1, dtype=int).item())
            obstacle_locations.append(tuple(obstacle_location))
            map[obstacle_location[0], obstacle_location[1]] = 1
        return obstacle_locations

    def _reset_targets(self):
        '''
        Place targets randomly or use user-defined locations for each population
        '''
        for population_id in self._num_agents_dict.keys():
            if self.random_init_targets_dict[population_id]:
                self._target_locations_dict[population_id] = []
                for _ in range(self.n_targets_dict[population_id]):
                    target_location = None
                    while target_location is None or self._map_dict[population_id][
                        target_location[0], target_location[1]] != 0:
                        target_location = (self.np_random.integers(0, self.grid_size[0], size=1, dtype=int).item(),
                                           self.np_random.integers(0, self.grid_size[1], size=1, dtype=int).item())
                    target_location = (target_location[0], target_location[1])
                    self._target_locations_dict[population_id].append(target_location)
            for target_location in self._target_locations_dict[population_id]:
                self._map_dict[population_id][target_location[0], target_location[1]] = -1

    def update_oracle_kwargs(self, oracle_kwargs):
        '''
        Add environment-specific parameters to oracle
        '''
        oracle_kwargs = oracle_kwargs | {"penetrable_obstacles": self.penetrable_obstacles_dict,
                                         "identical_grid_world": self.identical_grid_world}
        return oracle_kwargs

    def _reset_mf_oracles(self):
        '''
        Reset oracle with updated map
        '''
        self.mf_oracle.reset(grid_world_map_list=self._map_dict)

    def _render_frame(self):
        '''
        High-level render logic that generates smooth transitions over frames
        '''

        n_frames_between = 10
        old_render_list = [(self.agent_info[pop_id]["agent_{}".format(agent_indx)]["rendered-position"],
                            self.agent_info[pop_id]["agent_{}".format(agent_indx)]["rendered-color"])
                           for pop_id in self.population_id_list
                           for agent_indx in range(self._num_agents_dict[pop_id])]
        current_alive_list = [self.agent_info[pop_id]["agent_{}".format(agent_indx)]["current-alive"]
                              for pop_id in self.population_id_list
                              for agent_indx in range(self._num_agents_dict[pop_id])]

        new_render_list = self.generate_rendered_agent_positions_color()

        # If first frame, directly set positions
        if old_render_list[0][0] is None:
            self.update_agent_render_info(new_render_list)
            self._render_sub_frame()
        else:
            # Interpolate for smoother rendering
            for t in range(n_frames_between):
                intermediate_render_list = []
                for old_render_tuple, new_render_tuple, current_alive in zip(old_render_list, new_render_list, current_alive_list):
                    if current_alive:
                        intermediate_pos = (
                            old_render_tuple[0][0] + (
                                        new_render_tuple[0][0] - old_render_tuple[0][0]) * t / n_frames_between,
                            old_render_tuple[0][1] + (
                                        new_render_tuple[0][1] - old_render_tuple[0][1]) * t / n_frames_between)
                    else:
                        intermediate_pos = old_render_tuple[0]
                    intermediate_color = new_render_tuple[1]
                    intermediate_render_list.append((intermediate_pos, intermediate_color))

                self.update_agent_render_info(intermediate_render_list)

                self._render_sub_frame()

    def update_agent_render_info(self, rendered_list):
        '''
        Assigns new rendering position and color to each agent
        '''
        for pop_id in self.population_id_list:
            for agent_indx in range(self._num_agents_dict[pop_id]):
                agent_info = self.agent_info[pop_id]["agent_{}".format(agent_indx)]
                render_tuple = rendered_list.pop(0)
                agent_info["rendered-position"] = render_tuple[0]
                agent_info["rendered-color"] = render_tuple[1]

    def generate_rendered_agent_positions_color(self):
        '''
        Calculate pixel-level rendering coordinates (with multiple agents in a cell spaced out)
        '''
        agent_positions = [self.agent_info[pop_id]["agent_{}".format(agent_indx)]["current-location"]
                           for pop_id in self.population_id_list
                           for agent_indx in range(self._num_agents_dict[pop_id])]
        agent_color = [self.population_color_list[pop_id]
                       if self.agent_info[pop_id]["agent_{}".format(agent_indx)]["current-alive"]
                       else opacity(self.population_color_list[pop_id], alpha=0.3)
                       for pop_id in self.population_id_list
                       for agent_indx in range(self._num_agents_dict[pop_id])]

        map_count = np.zeros(self.grid_size, dtype=np.int32)
        for agent_pos in agent_positions:
            map_count[agent_pos[0], agent_pos[1]] += 1

        circle_location_map = [[[] for j in range(self.grid_size[1])] for i in range(self.grid_size[0])]
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                n_agents = map_count[i, j]
                circle_locations = place_circle(n=n_agents, r=0.2, x0=i -0.5, y0=j -0.5, L=1)
                circle_location_map[i][j] = circle_locations

        render_list = []
        for pos, color in zip(agent_positions, agent_color):
            rendered_pos = circle_location_map[pos[0]][pos[1]].pop(0)
            render_list.append((rendered_pos, color))

        return render_list

    def _render_sub_frame(self):
        '''
        Draws all environment components: grid lines, obstacles, targets, agents
        '''
    
        if self.window is None and self.render_mode == "human":
            pygame.init()
            #pygame.display.init()
            #self.window = pygame.display.set_mode(
            #    (self.window_size, self.window_size)
            #)
    
        
        if self.clock is None and self.render_mode in ["rgb_array", "human"]:
            self.clock = pygame.time.Clock()

        self._pix_square_size = (
            self.window_size / self.grid_size[0], self.window_size / self.grid_size[1]
        )
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((248, 250, 253))

        # render grid
        for x in range(self.grid_size[0] + 1):
            pygame.draw.line(
                canvas,
                (218, 226, 238),
                (0, self._pix_square_size[0] * x),
                (self.window_size, self._pix_square_size[0] * x),
                width=1,
            )
        for y in range(self.grid_size[1] + 1):
            pygame.draw.line(
                canvas,
                (218, 226, 238),
                (self._pix_square_size[1] * y, 0),
                (self._pix_square_size[1] * y, self.window_size),
                width=1,
            )

        self.render_population_density(canvas)

        # render obstacles
        self.render_obstacles(canvas)

        # render targets
        self.render_targets(canvas)

        # render additional environment features
        self.render_env_extras(canvas)

        # render agents
        self.render_agents(canvas)

        # render extras
        self.render_extras(canvas)

        frame = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        self.frames.append(frame)

        if self.render_mode == "human":
            #self.window.blit(canvas, canvas.get_rect())
            #pygame.event.pump()
            #pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def render_population_density(self, canvas):
        """Shade cells by population density to reveal clusters and movement patterns."""
        density_maps = {}
        maximum_density = 0.0
        for population_id in self._num_agents_dict:
            population_mf = self.mf_info[population_id].get("mf_3d")
            if population_mf is None:
                continue
            density = np.asarray(population_mf).sum(axis=-1)
            density_maps[population_id] = density
            maximum_density = max(maximum_density, float(density.max()))

        if maximum_density <= 0:
            return

        cell_width, cell_height = self._pix_square_size
        for population_id, density in density_maps.items():
            color = tuple(int(channel) for channel in self.population_color_list[population_id])
            for row, column in np.argwhere(density > 0):
                alpha = int(18 + 92 * float(density[row, column]) / maximum_density)
                tile = pygame.Surface(
                    (int(np.ceil(cell_width)), int(np.ceil(cell_height))), pygame.SRCALPHA
                )
                tile.fill((*color, alpha))
                canvas.blit(tile, (int(row * cell_width), int(column * cell_height)))

    def render_obstacles(self, canvas):
        '''
        Renders obstacles
        '''
        if self._num_population > 2 and not self.identical_grid_world:
            warnings.warn("Rendering obstacles for more than 2 populations is not properly supported yet.")
        if self.identical_grid_world:
            obstacle_location_list = self._obstacle_locations_dict["identical"]
            color_list = [(0, 0, 0) for _ in self._obstacle_locations_dict["identical"]]
        else:
            obstacle_location_list, color_list = [], []
            for population_id in self._num_agents_dict.keys():
                population_obstacles = self._obstacle_locations_dict[population_id]
                for obstacle_location in population_obstacles:
                    if obstacle_location not in obstacle_location_list:
                        obstacle_location_list.append(obstacle_location)
                        color_list.append(tuple(np.array(self.population_color_list[population_id]) / 6))
                    else:
                        color_list[obstacle_location_list.index(obstacle_location)] = (0, 0, 0)
        cell_width, cell_height = self._pix_square_size
        for row, column in obstacle_location_list:
            rect = pygame.Rect(
                int(row * cell_width) + 5,
                int(column * cell_height) + 5,
                max(1, int(cell_width) - 10),
                max(1, int(cell_height) - 10),
            )
            pygame.draw.rect(canvas, (51, 65, 85), rect, border_radius=8)
            for offset in range(-rect.height, rect.width, 12):
                start = (rect.left + offset, rect.bottom)
                end = (rect.left + offset + rect.height, rect.top)
                clipped = rect.clipline(start, end)
                if clipped:
                    pygame.draw.line(canvas, (71, 85, 105), clipped[0], clipped[1], width=2)

    def render_agents(self, canvas):
        '''
        Renders each agent as a small circle with color based on its population
        '''
        for population_id in self._num_agents_dict.keys():
            for agent_index in range(self._num_agents_dict[population_id]):
                agent_info = self.agent_info[population_id]["agent_{}".format(agent_index)]
                agent_location = agent_info["rendered-position"]
                agent_color = agent_info["rendered-color"]
                if agent_location is not None:
                    center = (np.asarray(agent_location) + 0.5) * self._pix_square_size
                    radius = max(4, int(min(self._pix_square_size) / 24))
                    pixel_center = (int(center[0]), int(center[1]))
                    pygame.draw.circle(canvas, (255, 255, 255), pixel_center, radius + 2)
                    pygame.draw.circle(canvas, agent_color, pixel_center, radius)
                    pygame.draw.circle(
                        canvas,
                        (255, 255, 255),
                        (pixel_center[0] - max(1, radius // 3), pixel_center[1] - max(1, radius // 3)),
                        max(1, radius // 4),
                    )

    def render_targets(self, canvas):
        '''
        Renders targets
        '''
        cell_width, cell_height = self._pix_square_size
        pygame.font.init()
        badge_font = pygame.font.Font(None, max(16, int(min(cell_width, cell_height) * 0.2)))
        for population_id in self._num_agents_dict:
            color = tuple(int(channel) for channel in self.population_color_list[population_id])
            tile_color = tuple(int(255 - 0.88 * (255 - channel)) for channel in color)
            for row, column in self._target_locations_dict[population_id]:
                rect = pygame.Rect(
                    int(row * cell_width) + 7,
                    int(column * cell_height) + 7,
                    max(1, int(cell_width) - 14),
                    max(1, int(cell_height) - 14),
                )
                pygame.draw.rect(canvas, tile_color, rect, border_radius=12)
                center = (int((row + 0.5) * cell_width), int((column + 0.5) * cell_height))
                radius = max(7, int(min(cell_width, cell_height) * 0.18))
                pygame.draw.circle(canvas, color, center, radius)
                pygame.draw.circle(canvas, (255, 255, 255), center, max(3, radius // 2))
                pygame.draw.circle(canvas, color, center, max(2, radius // 4))

                badge_center = (
                    int((row + 1) * cell_width - 17),
                    int(column * cell_height + 17),
                )
                pygame.draw.circle(canvas, (255, 255, 255), badge_center, 12)
                pygame.draw.circle(canvas, color, badge_center, 10)
                label = badge_font.render(population_id[:1].upper(), True, (255, 255, 255))
                canvas.blit(label, label.get_rect(center=badge_center))

    def render_extras(self, canvas):
        '''
        Render additional agent features, such as capture radius, etc
        '''
        raise NotImplemented

    def render_env_extras(self, canvas):
        '''
        Render additional environment features, such as targets, etc
        '''
        raise NotImplemented

    def color_circle(self, canvas, center_list, color_list, radius_list):
        '''
        Utility to render circles at specified positions
        '''

        if len(color_list) == 0:
            color_list = [color_list[0] for _ in range(len(center_list))]
        if len(radius_list) == 0:
            radius_list = [radius_list[0] for _ in range(len(center_list))]
        for center, color, radius in zip(center_list, color_list, radius_list):
            pygame.draw.circle(
                canvas,
                color,
                (np.array(center) + 0.5) * self._pix_square_size,
                radius,
            )

    def color_square(self, canvas, cell_location_list, color_list):
        '''
        Utility to render colored grid cells
        '''

        if len(color_list) == 0:
            color_list = [color_list[0] for _ in range(len(cell_location_list))]
        for cell_location, color in zip(cell_location_list, color_list):
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    (self._pix_square_size[0] * cell_location[0],
                     self._pix_square_size[1] * cell_location[1]),
                    (self._pix_square_size[0], self._pix_square_size[1]),
                ),
            )

    def get_action_direction(self, population_id, action_index):
        '''
        Returns the direction vector for a population-specific action index
        '''
        return self._action_map_dict[population_id][action_index]
