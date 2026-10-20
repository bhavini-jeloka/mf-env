import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Dict, Box
from typing import List


def flatten_dict_box_obs_space(observation_dict_space: Dict) -> Box:
    """
    Flatten a Dict observation space into a Box observation space.
    Args:
        observation_dict_space: A Dict observation space.
    Returns:
        A Box observation space.
    """
    if not isinstance(observation_dict_space, Dict):
        raise ValueError("The input observation space must be a Dict space.")
    low, high = [], []
    grid_shape, depth = None, (0,)
    for key in observation_dict_space.spaces.keys():
        if isinstance(observation_dict_space.spaces[key], Dict):
            flattened_subspace = flatten_dict_box_obs_space(observation_dict_space.spaces[key])
        else:
            flattened_subspace = observation_dict_space.spaces[key]
            assert isinstance(flattened_subspace, Box)

        low += [flattened_subspace.low]
        high += [flattened_subspace.high]

        if grid_shape is None:
            grid_shape = flattened_subspace.shape[0:2]
        else:
            assert grid_shape == flattened_subspace.shape[0:2]
        depth = (depth[0] + flattened_subspace.shape[2],)
    low = np.concatenate(low, axis=2)
    high = np.concatenate(high, axis=2)
    return Box(low=low, high=high, dtype=np.float32, shape=(grid_shape + depth))


def flatten_dict_box_obs(dict_box_obs: dict) -> npt.NDArray[np.float32]:
    """
    Flatten a dictionary of Box observations into a single Box observation.
    Args:
        dict_box_obs: A dictionary of Box observations.
    Returns:
        A single Box observation.
    """
    flatten_obs = None
    if not isinstance(dict_box_obs, dict):
        raise ValueError("The input observation to be flattened must be a dictionary.")
    for key in dict_box_obs.keys():
        if isinstance(dict_box_obs[key], dict):
            flattened_subobs = flatten_dict_box_obs(dict_box_obs[key])
        else:
            flattened_subobs = dict_box_obs[key]
            assert isinstance(flattened_subobs, np.ndarray)
        if flatten_obs is None:
            flatten_obs = flattened_subobs
        else:
            flatten_obs = np.concatenate((flatten_obs, flattened_subobs), dtype=np.float32, axis=2)
    return flatten_obs

    return np.array(flattened_obs, dtype=np.float32)


def merge_dict(dict_list: List[dict]):
    """
    Merges a list of dictionaries into a single dictionary (same keys and lists of values).
    Args:
        dict_box_obs: A list of dictionaries.
    Returns:
        A single merged dictionary.
    """
    merged_dict = {}
    for key, value in dict_list[0].items():
        if isinstance(value, dict):
            merged_dict[key] = merge_dict([d[key] for d in dict_list])
        elif isinstance(value, np.ndarray):
            merged_dict[key] = np.concatenate([d[key] for d in dict_list], axis=0)
        elif isinstance(value, (np.float32, np.float64, float, int)):
            merged_dict[key] = np.reshape(np.array([d[key] for d in dict_list]), (-1,1))
        else:
            merged_dict[key] = [d[key] for d in dict_list]
    return merged_dict


if __name__ == "__main__":
    dict1 = {"num": {"a": 1, "b": 2}, "fruit": "banana", "array": np.zeros((2, 8, 8, 1))}
    dict2 = {"num": {"a": 3, "b": 4}, "fruit": "apple", "array": np.ones((2, 8, 8, 1))}

    merged_dict = merge_dict([dict1, dict2])

    print("done!")
