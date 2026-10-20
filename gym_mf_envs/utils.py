def flat_dict_values(population_info_dict: dict[str, dict]):
    """
    Flattens the nested dictionary values into a single list.

    Args:
        population_info_dict (dict[str, dict]): A dictionary where each value is itself a dictionary.

    Returns:
        list: A list containing all the values from the nested dictionaries.
    """
    # Iterate over the inner dictionaries and collect all their values in a single list
    return [val for sub_dict in population_info_dict.values() for val in sub_dict.values()]


def opacity(color: tuple, alpha=0.5):
    """
    Applies an opacity transformation to a given RGB color.

    Args:
        color (tuple): A 3-tuple representing an RGB color (R, G, B) with values in [0, 255].
        alpha (float, optional): Opacity level between 0 (fully transparent) and 1 (fully opaque).
                                 Defaults to 0.5.

    Returns:
        tuple: A new RGB color tuple with the opacity applied.
    """
    # Linearly interpolate each RGB component toward 255 based on alpha (simulating blend with white)
    return (
        255 - alpha * (255 - color[0]),
        255 - alpha * (255 - color[1]),
        255 - alpha * (255 - color[2])
    )
