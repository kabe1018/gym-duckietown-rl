"""
Utilities for managing settings using the config.yml file and other sources,
such as updates to it in training scripts.
Adapted for Stable Baselines 3.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import os
import yaml
import glob
import logging
import numpy as np
from pprint import pformat
from duckietown_utils.utils import recursive_dict_update

logger = logging.getLogger(__name__)

def load_config(path="./config/config.yml", update_algo_hparams_from_algo_conf_file=True, config_updates={}):
    """Loads configuration from config.yml."""
    with open(path) as f:
        config = yaml.load(f, yaml.FullLoader)

    if update_algo_hparams_from_algo_conf_file:
        if config_updates.get('algo') is not None:
            config["algo"] = config_updates['algo']
        config = load_algo_config(config)

    if config_updates:
        update_config(config, config_updates)
    return config

def load_algo_config(config):
    """Loads algorithm-specific config from algo config files, based on config['algo']."""
    algo = config['algo']
    if isinstance(algo, dict):  # most likely algo={'grid_search':[...]}
        algo = 'general'
    if algo not in config["algo_config_files"]:
        algo = 'general'
    # Hier wird der Pfad zum Algo-Config-File zusammengesetzt.
    algo_config_file = os.path.join('/workspace/Duckietown-RL/config/algo', f"{algo.lower()}.yml")
    with open(algo_config_file) as f:
        algo_config = yaml.load(f, yaml.FullLoader)
    config.update(algo_config)
    return config

def dump_config(config, path):
    file_path = os.path.join(path, "config_dump_{:04d}.yml".format(config["seed"]))
    with open(file_path, "w") as config_dump:
        yaml.dump(config, config_dump, yaml.Dumper)

def print_config(config: dict):
    logger.info("=== Config ===================================")
    logger.info(pformat(config))

def update_config(config: dict, config_updates: dict):
    logger.warning("Updating default config values by:\n{}".format(pformat(config_updates)))
    recursive_dict_update(config, config_updates)

    # Aktualisiere env_config anhand der globalen Parameter seed und experiment_name.
    config['env_config'].update({
        'seed': config['seed'],
        'experiment_name': config['experiment_name']
    })

    # Für SB3-spezifische Anpassungen (z. B. Debug- oder Inference-Modus) kannst du hier weitere Updates vornehmen.
    mode = config['env_config'].get('mode')
    if mode not in ['train', 'inference', 'debug']:
        raise ValueError("env_config.mode must be one of ['train', 'inference', 'debug']")
    # Bisher wurden RLlib-spezifische Updates vorgenommen – diese entfallen im SB3-Setup.

def find_and_load_config_by_seed(seed, artifact_root="./artifacts",
                                 preselected_experiment_idx=None, preselected_checkpoint_idx=None):
    logger.warning("Found paths with seed {}:".format(str(seed)))
    config_dump_path = _find_and_select_experiment(artifact_root + '/**/config_dump_{:04d}.yml'.format(seed),
                                                   preselected_experiment_idx)

    logger.warning("Found checkpoints in {}:".format(os.path.dirname(config_dump_path)))
    checkpoint_path = _find_and_select_experiment(
        os.path.join(os.path.dirname(config_dump_path), '**', 'checkpoint-*'),
        preselected_checkpoint_idx)

    loaded_config = load_config(config_dump_path, update_algo_hparams_from_algo_conf_file=False)
    logger.warning("Config loaded from {}".format(config_dump_path))
    logger.warning("Model checkpoint loaded from {}".format(checkpoint_path))
    return loaded_config, checkpoint_path

def _find_and_select_experiment(search_string, preselect_index=None):
    paths = glob.glob(search_string, recursive=True)
    paths.sort()
    for i, path in enumerate(paths):
        logger.warning("{:d}: {}".format(i, path))

    number_of_experiments = len(paths)
    if number_of_experiments <= 0:
        raise ValueError("No artifacts found with pattern {}".format(search_string))

    if number_of_experiments > 1:
        if preselect_index is None:
            logger.warning("Enter experiment number: ")
            experiment_num = int(input())
        else:
            experiment_num = preselect_index
        experiment_num = np.clip(experiment_num, 0, number_of_experiments - 1)
    else:
        experiment_num = 0

    return paths[experiment_num]
