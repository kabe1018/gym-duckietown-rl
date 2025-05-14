#!/usr/bin/env python
"""
Script for training lane-following agents (including collision avoidance)
with hyperparameter optimization using Ray Tune.

Basierend auf train-rllib.py, werden hier zusätzlich die Reward-Parameter
(lambda_d, lambda_psi, lambda_v, phi, epsilon) über Ray Tune optimiert.
"""

__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import sys
sys.path.append('/workspace/Duckietown-RL')

import os
import logging
import ray
from ray import tune
from ray.tune.registry import register_env
from duckietown_utils.env import launch_and_wrap_env

# Ray 2.x:
from ray.rllib.algorithms.ppo import PPO as PPOTrainer

from config.paths import ArtifactPaths
from config.config import load_config, print_config, dump_config, update_config, find_and_load_config_by_seed
# Importiere nun die neue Funktion, die in env.py stehen sollte:
from duckietown_utils.env import create_duckietown_env
from duckietown_utils.utils import seed

# Use our new callback class
from duckietown_utils.rllib_callbacks import DuckietownCallbacks

import gym_duckietown.simulator
print("Verwendete simulator.py:", gym_duckietown.simulator.__file__)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

###########################################################
# Load config
config = load_config(
    '/workspace/Duckietown-RL/config/config.yml',
    config_updates={"env_config": {"mode": "train"}}
)
# Set numpy and random seed
seed(1234)

###########################################################
# Set up experiment parameters
config_updates = {
    "seed": 0000,
    "experiment_name": "Train_0029",
    "env_config": {},
    "rllib_config": {},
}
update_config(config, config_updates)

###########################################################
# Restore training if configured
if config['restore_seed'] >= 0:
    pretrained_config, checkpoint_path = find_and_load_config_by_seed(
        config['restore_seed'],
        preselected_experiment_idx=config['restore_experiment_idx'],
        preselected_checkpoint_idx=config['restore_checkpoint_idx']
    )
    logger.warning("Overwriting config from {}".format(checkpoint_path))
    config = pretrained_config
    update_config(config, config_updates)
else:
    checkpoint_path = None

###########################################################
# Print final config
print_config(config)

###########################################################
# Setup paths
paths = ArtifactPaths(config['experiment_name'], config['seed'], algo_name=config['algo'])

# Code backup
os.system(f'cp -ar /workspace/Duckietown-RL/duckietown_utils {paths.code_backup_path}/')
os.system(f'cp -ar /workspace/Duckietown-RL/experiments {paths.code_backup_path}/')
os.system(f'cp -ar /workspace/Duckietown-RL/config {paths.code_backup_path}/')

###########################################################
# Initialize Ray and register env
ray.init(**config["ray_init_config"])
# Verwende die neue Umgebungserzeugungsfunktion mit Hyperparameter-Support
register_env('Duckietown', launch_and_wrap_env)

# Fix the RLlib config to match Ray 2.x:
cfg = config["rllib_config"]

# Entferne ggf. den alten Schlüssel "evaluation_num_episodes":
if "evaluation_num_episodes" in cfg:
    del cfg["evaluation_num_episodes"]

# Update RLlib-Konfiguration:
cfg.update({
    "env": "Duckietown",
    "env_config": config["env_config"],
    # Übergabe der Callback-Klasse:
    "callbacks": DuckietownCallbacks,
    # Ersetze "evaluation_num_episodes" mit:
    "evaluation_duration": 2,  # z.B. 2 Episoden
    "evaluation_duration_unit": "episodes",  # oder "timesteps"
    # Weitere Parameter können beibehalten werden
})

# Neue Einstellungen:
cfg["ignore_worker_failures"] = True  # Training läuft weiter, auch wenn Worker abstürzen
cfg["recreate_failed_workers"] = False  # Keine Worker neu starten, um Fehler besser zu debuggen

import yaml
# Lade die YAML-Konfiguration (ppo.yml)
with open("config/algo/ppo.yml", "r") as f:
    config = yaml.safe_load(f)

# Hier fügst du deine GridSearch-Parameter hinzu:
#cfg["train_batch_size"] = tune.grid_search([2048, 4096, 6144])
#cfg["rollout_fragment_length"] = tune.grid_search([256, 341, 512])
#cfg["num_sgd_iter"] = tune.grid_search([20, 30, 40])
cfg["lr"] = tune.grid_search([1e-5, 5e-5, 1e-4])
#cfg["kl_target"] = tune.grid_search([0.01, 0.02])
#cfg["vf_clip_param"] = tune.grid_search([0.01, 0.02, 3.00])
#cfg["batch_mode"] = "truncate_episodes"


# Füge hier den Hyperparameter-Suchraum für den Reward-Wrapper hinzu:
# Diese Werte überschreiben ggf. die in der config.yml gesetzten Standardwerte.
cfg["env_config"]["reward_wrapper_params"] = {
    "lambda_d": tune.grid_search([2.0]),
    "lambda_psi": tune.grid_search([2.0]),
    "lambda_v": tune.grid_search([1.5]),
    #"phi": tune.grid_search([30.0, 50.0, 70.0]),
    #"epsilon": tune.grid_search([0.01, 0.05, 0.1]),
}


###########################################################
# Set up a Reporter for Ray Tune (zeigt während des Trainings wichtige Metriken)
from ray.tune import CLIReporter
reporter = CLIReporter(
    metric_columns={
        "training_iteration": "iter",
        "time_total_s": "time (s)",
        "timesteps_total": "ts",
        "episode_reward_mean": "reward",
        "episode_reward_max": "reward_max",
        "episode_reward_min": "reward_min",
        "episode_len_mean": "len_mean",
        "custom_metrics/distance_travelled_mean": "dist_mean",
    },
    parameter_columns=["-"]
)

###########################################################
# Start training via Ray Tune
tune.run(
    PPOTrainer,
    stop={"timesteps_total": config["timesteps_total"]},
    config=cfg,
    progress_reporter=reporter,
    # Ray 2.x nutzt "storage_path" anstelle von "local_dir"
    local_dir="./artifacts",
    #log_to_file=("stdout.log", "stderr.log"),
    checkpoint_at_end=True,
    trial_name_creator=lambda trial: trial.trainable_name,
    name=paths.experiment_folder,
    keep_checkpoints_num=1,
    checkpoint_score_attr="episode_reward_mean",
    checkpoint_freq=1,
    restore=checkpoint_path,
)

