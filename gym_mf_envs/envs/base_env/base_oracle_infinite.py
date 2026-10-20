from typing import Union, List
from numpy import ndarray

class BaseMFInfiniteOracle:
    def __init__(self, num_populations: int,  num_states: dict[str, int],
                 num_actions: dict[str, int],
                 **kwargs):
        """Initialize this object and its environment state."""
        self._num_populations = num_populations
        self._num_states, self._num_actions = num_states, num_actions

        # setup population ratio
        if kwargs.get("rho") is not None:
            self._population_ratio = kwargs.get("rho")
        else:
            self.rho = 0.5

    @property
    def population_id_list(self):
        """Return the configured population identifiers."""
        return list(self._num_states.keys())

    def reset(self, **kwargs):
        """
        Reset the mf oracle with new environment information, e.g. obstacle locations, etc.
        """
        raise NotImplemented

    def update_mf_info(self, mf_info: dict[str, dict[str, Union[tuple, ndarray]]]):
        """
        Update/compute the mf information 

        Args:
            mf_info[population_id][info_id]: mean-field information with info_id of population_id

        Returns:
            None
        """
        raise NotImplemented

    def get_reward(self, mf_info: dict[str, dict[str, Union[tuple, ndarray]]], policies: dict[str, dict[str, Union[tuple, ndarray]]]) -> dict:
        """
        Get the reward for each population

        Args:
            mf_info[population_id][info_id]: mean-field information with info_id of population_id

        Returns:
            rewards[population_id]: reward for agent_id in population
        """
        raise NotImplemented

    def get_population_transition_probability(self, population_id_list: Union[List[str], None],
                                              mf_info: dict[str, dict[str, Union[tuple, ndarray]]], policy: dict[str, Union[tuple, ndarray]]) -> List[ndarray]:
        """
        Get the full transition probability of population_id under the current mean-field for all states and actions

        Args:
            population_id: (str) population id
            mf_info [population_id][info_id]: mean-field information with info_id of population_id

        Returns:
            transition_prob[a][i, j]: transition probability from state i to state j under action a
        """
        raise NotImplemented
