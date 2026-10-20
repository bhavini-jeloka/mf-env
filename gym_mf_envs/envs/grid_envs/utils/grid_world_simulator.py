import numpy as np


class GridWorldSimulator:
    """
    Simulates a 2D grid world with support for obstacles, targets, and agent movement.

    Attributes:
        size (tuple): Dimensions of the grid (rows, columns).
        penetrable_obstacles (bool): If True, agents can move into obstacle cells with some probability.
        absorbing_targets (bool): If True, agents stay in target cells once reached.
        _map (np.ndarray): Internal representation of the environment, where:
            - 0 indicates free space
            - 1 indicates obstacle
            - -1 indicates target
    """
    def __init__(self, size: tuple, penetrable_obstacles: bool = False, absorbing_targets=True,
                 map: np.ndarray = None):
        """Initialize this object and its environment state."""
        self.size = size
        self.penetrable_obstacles = penetrable_obstacles
        self.absorbing_targets = absorbing_targets
        self._map = map
        self._pos2pos_state_map, self._pos_state2pos_map, self.n_pos_state = None, None, None
        self._target_states, self._target_pos, self._obstacle_pos = None, None, None
        if map is not None:
            assert map.shape == self.size
            self._init_pos2pos_state_map()

    def get_n_states(self):
        """Return number of valid (non-obstacle) states."""
        return self.n_pos_state

    def reset(self, new_map):
        """ 
        Reset the environment with a new map.
        """
        assert new_map.shape == self.size + (1,)
        self._map = new_map
        self._init_pos2pos_state_map()
        obstacle_pos_ = np.where(self._map == 1)
        self._obstacle_pos = [(obstacle_pos_[0][i], obstacle_pos_[1][i]) for i in range(len(obstacle_pos_[0]))]
        target_pos_ = np.where(self._map == -1)
        self._target_pos = [(target_pos_[0][i], target_pos_[1][i]) for i in range(len(target_pos_[0]))]

    def get_random_pos(self, num_positions: int):
        """Generate a list of random valid (non-obstacle) positions."""
        pos_list = []
        for i in range(num_positions):
            pos = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))
            while not self._check_valid_pos(pos):
                pos = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))
            pos_list.append(pos)
        return pos_list

    def get_intended_new_pos(self, current_pos: tuple, action_vector: tuple):
        """
        Compute the intended next position given a current position and an action vector.\
        """
        if self.absorbing_targets and self.is_pos_in_target(current_pos):
            is_in_obstacle = False
            return current_pos, is_in_obstacle
        else:
            new_pos = tuple(np.clip(np.array(current_pos) + np.array(action_vector), 0, np.array(self.size) - 1))
            is_in_obstacle = self.is_pos_in_obstacle(new_pos)

            if self.penetrable_obstacles:
                return new_pos, is_in_obstacle
            elif not is_in_obstacle:
                return new_pos, is_in_obstacle
            else:
                return current_pos, False

    def get_state_action_transition_probability(self, state: int, action_vector: tuple):
        """
        Return the next state given the current state and action vector.
        """
        pos = self.get_pos(state)
        new_pos, is_in_obstacle = self.get_intended_new_pos(pos, action_vector)
        new_state = self.get_pos_state(new_pos)
        return [new_state], [1.0]

    def get_pos_state(self, pos: tuple):
        """Return the state index associated with a position."""
        state = self._pos2pos_state_map[pos[0], pos[1]]
        assert state != -1
        return state

    def get_pos_state_list(self, pos_list: list):
        """Return list of state indices corresponding to given positions."""
        return [self.get_pos_state(pos) for pos in pos_list]

    def get_pos(self, pos_state: int):
        """Return the position tuple for a given state index."""
        return self._pos_state2pos_map[pos_state]

    def _check_valid_pos(self, pos: tuple):
        """
        Check whether the given position is within bounds and not an obstacle (unless penetrable).
        """
        if pos[0] < 0 or pos[0] >= self.size[0] or pos[1] < 0 or pos[1] >= self.size[1]:
            return False
        elif self._map[pos[0], pos[1]] == 1 and not self.penetrable_obstacles:
            return False
        else:
            return True

    def _init_pos2pos_state_map(self):
        """
        Initialize the mapping between position and state
        self._pos2pos_state_map: numpy array with shape self.size, such that self._pos2pos_state_map[i, j] = state,
        where -1 marks the invalid states in obstacle
        self._pos_state2pos_map: list of valid positions, such that self._pos_state2pos_map[state] = (i, j)
        :return: None
        """
        self._pos2pos_state_map = np.zeros(self.size, dtype=int) - 1
        self._pos_state2pos_map = []
        self.n_pos_state = 0

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self._map[i, j] < 1 or self.penetrable_obstacles:
                    self._pos2pos_state_map[i, j] = self.n_pos_state
                    self._pos_state2pos_map.append((i, j))
                    self.n_pos_state += 1

    def is_pos_in_obstacle(self, pos):
        """Check if a position is an obstacle."""
        return self._map[pos[0], pos[1]] == 1

    def is_state_in_obstacle(self, state):
        """Check if a state corresponds to an obstacle (used when obstacles are penetrable)."""
        assert self.penetrable_obstacles
        pos = self.get_pos(state)
        return self.is_pos_in_obstacle(pos)

    def is_pos_in_target(self, pos):
        """Check if a position is a target cell."""
        return self._map[pos[0], pos[1]] == -1
