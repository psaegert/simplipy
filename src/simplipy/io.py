"""Reading of SimpliPy YAML configuration files.

Provides :func:`load_config`, which loads a configuration from a dictionary or YAML
file. (Its former sibling ``save_config`` was removed in 0.12.0: zero callers here or
downstream, and its ``reference=`` switch was a no-op -- both branches wrote identical
output.)

Path resolution is deliberately NOT this module's job: which values are paths is
SCHEMA knowledge, so each path-valued key is resolved by its consumer, key-aware
(``rules`` against the config directory in ``SimpliPyEngine.from_config``;
``snapshot_at`` against the output directory and ``proposals`` against the config
directory in the ``find-rules`` CLI). The former ``resolve_paths`` value-sniffing pass
(any string starting with ``.`` and ending ``.yaml``/``.json``) was removed in 0.12.0:
it silently skipped equally-relative spellings like ``proposals.json`` while rewriting
``./proposals.json``, so resolution depended on the SPELLING of a value instead of the
meaning of its key.
"""
import os
from typing import Any

import yaml

__all__ = ['load_config']  # D11 column R22 (owner-ratified 2026-08-16)


def load_config(config: dict[str, Any] | str) -> dict[str, Any]:
    '''
    Load a configuration file.

    Parameters
    ----------
    config : dict or str
        The configuration dictionary or path to the configuration file.

    Returns
    -------
    dict
        The configuration dictionary. Path-valued entries are returned VERBATIM;
        resolution is the consumer's job (see the module docstring).
    '''

    if isinstance(config, str):
        config_path = config

        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file {config_path} not found.')
        if os.path.isfile(config_path):
            with open(config_path, 'r') as config_file:
                config_ = yaml.safe_load(config_file)
        else:
            raise ValueError(f'Config file {config_path} is not a valid file.')
    else:
        config_ = config

    return config_
