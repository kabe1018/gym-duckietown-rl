#!/usr/bin/env python3
"""
Script for training lane-following agents (including collision avoidance)
using Stable Baselines 3 (SB3).

Die Konfiguration wird ähnlich wie bisher geladen; 
nur die Trainingslogik wurde auf SB3 umgestellt.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import sys
sys.path.append('/workspace/Duckietown-RL')

from duckietown_utils.sb3_callbacks import DuckietownSB3Callback
import os
import logging
import time

from config.paths import ArtifactPaths
from config.config import load_config, print_config, dump_config, update_config, find_and_load_config_by_seed
from duckietown_utils.env import launch_and_wrap_env
from duckietown_utils.utils import seed

# SB3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

logger = logging.getLogger()
logger.setLevel(logging.INFO)

###########################################################
# Load config
config = load_config(
    '/workspace/Duckietown-RL/config/config.yml',
    config_updates={"env_config": {"mode": "train"}}
)
seed(1234)

###########################################################
# Set up experiment parameters – hier wird zusätzlich ein "sb3_config" Block erzeugt.
config_updates = {
    "seed": 0000,
    "experiment_name": "Train_0030",
    "env_config": {},
    "sb3_config": {}  # Hier können SB3-spezifische Hyperparameter (z.B. n_steps, batch_size, n_epochs, etc.) stehen.
}
update_config(config, config_updates)

###########################################################
# Optional: Restore training if configured (SB3 verwendet in der Regel .zip-Dateien)
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
# Setup paths and code backup
paths = ArtifactPaths(config['experiment_name'], config['seed'], algo_name=config['algo'])
os.system(f'cp -ar /workspace/Duckietown-RL/duckietown_utils {paths.code_backup_path}/')
os.system(f'cp -ar /workspace/Duckietown-RL/experiments {paths.code_backup_path}/')
os.system(f'cp -ar /workspace/Duckietown-RL/config {paths.code_backup_path}/')

###########################################################
# Create environment
env = launch_and_wrap_env(config["env_config"])
print("Observation space shape:", env.observation_space.shape)

###########################################################
# Instantiate SB3 PPO model
# SB3-Konfiguration aus der config laden und n_envs entfernen (falls vorhanden)
sb3_config = config.get("sb3_config", {}).copy()
sb3_config.pop("n_envs", None)  # Entferne n_envs, da SB3 diesen nicht erwartet
# Stelle sicher, dass learning_rate ein Float ist:
if isinstance(sb3_config.get("learning_rate"), str):
    sb3_config["learning_rate"] = float(sb3_config["learning_rate"])

model = PPO("CnnPolicy", env, verbose=1, **sb3_config)

# Falls ein Checkpoint vorliegt, versuche diesen zu laden (Checkpoint-Datei muss im SB3-Format vorliegen)
if checkpoint_path is not None:
    logger.info("Restoring model from checkpoint: {}".format(checkpoint_path))
    model = PPO.load(checkpoint_path, env=env)

###########################################################
logger.setLevel(logging.WARNING)
# Set up callbacks
# Speichere zwischendurch Modelle und evaluiere auf einer separaten Eval-Umgebung.
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=paths.experiment_base_path, name_prefix='ppo_model')
eval_env = launch_and_wrap_env(config["env_config"])  # Separate Umgebung für die Evaluation
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=paths.experiment_base_path,
    log_path=paths.experiment_base_path,
    eval_freq=50000,
    deterministic=True,
    render=False
)

# Importiere und initialisiere den Metriken-Callback
from duckietown_utils.sb3_callbacks import DuckietownSB3Callback
result_callback = DuckietownSB3Callback(
    save_path=os.path.join(os.path.abspath(paths.experiment_base_path), "result.json")
)

###########################################################
# Dump final config for record
dump_config(config, paths.experiment_base_path)

###########################################################
# Training starten mit Zeitmessung
total_timesteps = config["timesteps_total"]
start_time = time.time()
print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

model.learn(total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, result_callback], 
            log_interval=10)

end_time = time.time()
elapsed = end_time - start_time
print(f"Training finished after {elapsed:.2f} seconds.")
print("Artifacts and final model saved in:", os.path.abspath(paths.experiment_base_path))

# Speichere das finale Modell
model.save(os.path.join(paths.experiment_base_path, "final_model.zip"))