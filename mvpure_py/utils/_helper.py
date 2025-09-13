import inspect


def split_kwargs(kwargs: dict, func_map: dict):
    """
    Split kwargs based on the valid parameters of each function in func_map.

    Parameters
    ----------
    kwargs : dict
        All keyword parameters
    func_map : dict
        Mapping of names to functions

    Returns
    -------
    dict : Dictionary mapping each name to a dict of its valid kwargs.
    """
    valid_params = {
        name: set(inspect.signature(func).parameters)
        for name, func in func_map.items()
    }

    split = {name: {} for name in func_map}

    for key, value in kwargs.items():
        matched = False
        for name, params in valid_params.items():
            if key in params:
                split[name][key] = value
                matched = True
                break
        if not matched:
            raise TypeError(f"Unknown parameter: {key}")

    return split
