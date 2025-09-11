import os
from typing import Any

import copy
import yaml

from simplipy.utils import apply_on_nested


def load_config(config: dict[str, Any] | str, resolve_paths: bool = True) -> dict[str, Any]:
    '''
    Load a configuration file.

    Parameters
    ----------
    config : dict or str
        The configuration dictionary or path to the configuration file.
    resolve_paths : bool, optional
        Whether to resolve relative paths in the configuration file, by default True.

    Returns
    -------
    dict
        The configuration dictionary.
    '''

    if isinstance(config, str):
        config_path = config
        config_base_path = os.path.dirname(config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file {config_path} not found.')
        if os.path.isfile(config_path):
            with open(config_path, 'r') as config_file:
                config_ = yaml.safe_load(config_file)
        else:
            raise ValueError(f'Config file {config_path} is not a valid file.')

        def resolve_path(value: Any) -> str:
            if isinstance(value, str) and (value.endswith('.yaml') or value.endswith('.json')) and value.startswith('.'):  # HACK: Find a way to check if a string is a path
                return os.path.join(config_base_path, value)
            return value

        if resolve_paths:
            config_ = apply_on_nested(config_, resolve_path)

    else:
        config_ = config

    return config_


def save_config(config: dict[str, Any], directory: str, filename: str, reference: str = 'relative', recursive: bool = True, resolve_paths: bool = False) -> None:
    '''
    Save a configuration dictionary to a YAML file.

    Parameters
    ----------
    config : dict
        The configuration dictionary to save.
    directory : str
        The directory to save the configuration file to.
    filename : str
        The name of the configuration file.
    reference : str, optional
        Determines the reference base path. One of
        - 'project': relative to the project root
        - 'absolute': absolute paths
    recursive : bool, optional
        Save any referenced configs too
    '''
    config_ = copy.deepcopy(config)

    def save_config_relative_func(value: Any) -> Any:
        if isinstance(value, str) and value.endswith('.yaml'):
            relative_path = value
            if not value.startswith('.'):
                relative_path = os.path.join('.', os.path.basename(value))
            save_config(load_config(value, resolve_paths=resolve_paths), directory, os.path.basename(relative_path), reference=reference, recursive=recursive, resolve_paths=resolve_paths)
        return value

    def save_config_absolute_func(value: Any) -> Any:
        if isinstance(value, str) and value.endswith('.yaml'):
            relative_path = value
            if not value.startswith('.'):
                relative_path = os.path.abspath(value)
            save_config(load_config(value, resolve_paths=resolve_paths), directory, os.path.basename(relative_path), reference=reference, recursive=recursive, resolve_paths=resolve_paths)
        return value

    if recursive:
        match reference:
            case 'relative':
                apply_on_nested(config_, save_config_relative_func)
            case 'absolute':
                apply_on_nested(config_, save_config_absolute_func)
            case _:
                raise ValueError(f'Invalid reference type: {reference}')

    save_path = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as config_file:
        yaml.dump(config_, config_file, sort_keys=False)
