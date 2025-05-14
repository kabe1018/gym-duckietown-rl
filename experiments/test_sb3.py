#!/usr/bin/env python3
"""
Script to evaluate our agents in many ways using Stable Baselines 3 (SB3).
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import gymnasium as gym
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys
sys.path.append('/workspace/Duckietown-RL')

# SB3 Import
from stable_baselines3 import PPO

# GPU-Check (benötigt torch)
#import torch


import time
import logging
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import cv2
import json
from tqdm import tqdm

# Konfigurations- und Hilfsfunktionen
from config.config import find_and_load_config_by_seed, update_config, print_config
from config.paths import ArtifactPaths
from duckietown_utils.env import launch_and_wrap_env, get_wrappers
from duckietown_utils.utils import seed
from duckietown_utils.duckietown_world_evaluator import DuckietownWorldEvaluator, DEFAULT_EVALUATION_MAP, myTestMapA
from duckietown_utils.trajectory_plot import correct_gym_duckietown_coordinates, plot_trajectories
# Falls du bisher etwas aus dem Modul "duckietown_utils.salient_object_visualization" genutzt hast, entferne den Import:
# from duckietown_utils.salient_object_visualization import display_salient_map2



class GymToGymnasiumWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Erstelle ein leeres Info-Dictionary, falls nicht vorhanden
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Hier kannst du entscheiden, ob "done" als "terminated" gilt und "truncated" immer False ist
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode)
    
    def close(self):
        return self.env.close()



logger = logging.getLogger()
logger.setLevel(logging.INFO)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

###########################################################
# Kommandozeilen-Argumente einlesen
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed-model-id', default=0000, type=int,
                    help='Unique experiment identifier, referred to as seed (4 digit).')
parser.add_argument('--analyse-trajectories', action='store_true',
                    help='Calculate metrics and create trajectory plots.')
parser.add_argument('--visualize-salient-obj', action='store_true',
                    help='Visualize salient object while running the policy in closed loop simulation (not implemented for SB3).')
parser.add_argument('--visualize-dot-trajectories', action='store_true',
                    help='Visualize trajectories of many episodes as dotted lines.')
parser.add_argument('--reward-plots', action='store_true',
                    help='Simulate closed loop behavior and show time-plots of the reward, distance between vehicles, etc.')
parser.add_argument('--map-name', default=DEFAULT_EVALUATION_MAP, help="Specify the map")
parser.add_argument('--domain-rand', action='store_true', help='Enable domain randomization')
parser.add_argument('--top-view', action='store_true',
                    help="View the simulation from a fixed bird's eye view, instead of the robot's view")
parser.add_argument('--results-path', default='default', type=str,
                    help='Analysis results are saved to this folder. If "default" is given, results are saved to the path of the loaded model.')
args = parser.parse_args()

render_mode = 'top_down' if args.top_view else 'human'
test_map = args.map_name
seed(1234)

###########################################################
# Experiment laden
SEED = args.seed_model_id
# Hier laden wir den Checkpoint und die Config aus Dateien.
# Passe diese Pfade nach Bedarf an.
#checkpoint_path = "/workspace/Duckietown-RL/artifacts/Train_0030_0000/Mar01_19-56-32/best_model.zip"
#config_file = "/workspace/Duckietown-RL/artifacts/Train_0030_0000/Mar01_19-56-32/config_dump_0000.yml"
#checkpoint_path = "/workspace/Duckietown-RL/artifacts/Train_0030_0000/Mar06_01-56-23/best_model.zip"
#config_file = "/workspace/Duckietown-RL/artifacts/Train_0030_0000/Mar06_01-56-23/config_dump_0000.yml"


#Testen
#checkpoint_path = "/workspace/Duckietown-RL/artifacts/Train_0030_0000/Mar06_19-14-57/best_model.zip"
#config_file = "/workspace/Duckietown-RL/artifacts/Train_0030_0000/Mar06_19-14-57/config_dump_0000.yml"

checkpoint_path = "/workspace/Duckietown-RL/artifacts/Train_0030_0000/Mar07_01-33-02/best_model.zip"
config_file = "/workspace/Duckietown-RL/artifacts/Train_0030_0000/Mar07_01-33-02/config_dump_0000.yml"




import yaml
with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)
print("Aktuelle Konfiguration:")
print(config)

# Update der Konfiguration für Inference und den gewünschten Map
update_config(config, {
    'env_config': {
        'mode': 'inference',
      #  'training_map': test_map,
      #  'sim_host': 'localhost',
    },
})

###########################################################
# GPU-Verfügbarkeit prüfen
#if torch.cuda.is_available():
#    print("CUDA is available. GPU will be used.")
#else:
#    print("CUDA is not available. Using CPU.")

###########################################################
# Erstelle die Umgebung
env = launch_and_wrap_env(config["env_config"])
env = GymToGymnasiumWrapper(env)
print("Observation space shape:", env.observation_space.shape)

###########################################################
# Modell ohne Environment laden
model = PPO.load(checkpoint_path)
# Anschließend das Environment zuweisen
model.observation_space = env.observation_space
model.action_space = env.action_space
model.set_env(env)
print("Model loaded from:", os.path.abspath(checkpoint_path))

print_config(config)


# Hilfsfunktion, um model.predict zu imitieren
def compute_action(obs, explore=False):
    action, _ = model.predict(obs, deterministic=not explore)
    return action

###########################################################
# Option 1: Einfache Demonstration im Closed-Loop
if not (args.analyse_trajectories or args.visualize_salient_obj or args.reward_plots or args.visualize_dot_trajectories):
    demo_env = launch_and_wrap_env(config["env_config"])
    for i in range(5):
        obs = demo_env.reset()
        demo_env.render(render_mode)
        done = False
        while not done:
            action = compute_action(obs, explore=False)
            obs, reward, done, info = demo_env.step(action)
            time.sleep(0.15)
            # Alternativer Render: Verzerrung kurz deaktivieren
            orig_distortion = demo_env.unwrapped.distortion
            demo_env.unwrapped.distortion = False
            demo_env.render(render_mode)
            demo_env.unwrapped.distortion = orig_distortion
            if demo_env.unwrapped.frame_skip > 1:
                time.sleep(demo_env.unwrapped.delta_time * demo_env.unwrapped.frame_skip)

###########################################################
# Option 2: Trajektorienanalyse
if args.analyse_trajectories:
    config['env_config']['spawn_forward_obstacle'] = False
    evaluator = DuckietownWorldEvaluator(config['env_config'], eval_lenght_sec=15, eval_map=test_map)
    results_path = os.path.abspath(os.path.split(checkpoint_path)[0]) if args.results_path == 'default' else args.results_path
    evaluator.evaluate(lambda obs: compute_action(obs, explore=False), results_path)

###########################################################
# Option 3: Visualize Salient Object
if args.visualize_salient_obj:
    logger.warning("Visualize salient object is not implemented for SB3 (TensorFlow-based code has been removed).")

###########################################################
# Option 4: Dotted Trajektorien Plots
if args.visualize_dot_trajectories:
    config['env_config']['training_map'] = 'custom_huge_loop'
    trajectories = []
    for i in tqdm(range(10)):
        traj_env = launch_and_wrap_env(config["env_config"], i)
        ego_robot_pos = []
        obs = traj_env.reset()
        done = False
        while not done:
            action = compute_action(obs, explore=False)
            obs, reward, done, info = traj_env.step(action)
            time.sleep(0.15)
            ego_robot_pos.append(correct_gym_duckietown_coordinates(traj_env.unwrapped, traj_env.unwrapped.cur_pos))
        trajectories.append(ego_robot_pos)
        traj_env.close()
    plot_trajectories(trajectories, show_plot=True)
    plot_trajectories(trajectories, show_plot=True, unify_start_tile=False)

###########################################################
# Option 5: Detaillierte Demonstration (Reward-Plots)
if args.reward_plots:
    from duckietown_utils.duckietown_world_evaluator import DuckiebotObj
    for i in range(1):
        plot_env = launch_and_wrap_env(config["env_config"], i)
        prox_pen = []
        coll_reward = []
        vel_reward = []
        angl_reward = []
        ego_robot_pos = []
        npc_robot_pos = []
        timestamps = []
        obs = plot_env.reset()
        plot_env.render('top_down')
        done = False
        step = 0
        while not done:
            t0 = time.time()
            action = compute_action(obs, explore=False)
            t1 = time.time()
            obs, reward, done, info = plot_env.step(action)
            time.sleep(0.15)
            prox_pen.append(info['Simulator'].get('proximity_penalty', 0))
            coll_reward.append(info.get('custom_rewards', {}).get('collision_avoidance', 0))
            vel_reward.append(info.get('custom_rewards', {}).get('velocity', 0))
            angl_reward.append(info.get('custom_rewards', {}).get('orientation', 0))
            timestamps.append(info['Simulator'].get('timestamp', 0))
            ego_robot_pos.append(correct_gym_duckietown_coordinates(plot_env.unwrapped, plot_env.unwrapped.cur_pos))
            for npc in plot_env.unwrapped.objects:
                if isinstance(npc, DuckiebotObj):
                    npc_robot_pos.append(correct_gym_duckietown_coordinates(plot_env.unwrapped, npc.pos))
            t2 = time.time()
            plot_env.unwrapped.distortion = False
            plot_env.render(render_mode)
            step += 1
            plot_env.unwrapped.distortion = True
            t3 = time.time()
            print("Inference time {:.3f}ms | Env step time {:.3f}ms | Render time {:.3f}ms".format(
                  (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000))
        matplotlib.use('TkAgg')
        plt.subplot(211)
        plt.plot(np.array(prox_pen))
        plt.plot(np.array(coll_reward))
        plt.legend(["Proximity penalty", "Collision Avoidance Reward"])
        plt.grid('on')
        plt.subplot(212)
        plt.plot(np.array(vel_reward))
        plt.plot(np.array(angl_reward))
        plt.plot(np.array(coll_reward) + np.array(vel_reward) + np.array(angl_reward))
        plt.legend(["Velocity", "Orientation", "Sum"])
        plt.ylabel("Reward components")
        plt.grid('on')
        plt.show()
        if len(npc_robot_pos) > 0:
            ROBOT_LENGTH = 0.18
            plt.plot(np.array(timestamps), np.linalg.norm(np.array(npc_robot_pos)-np.array(ego_robot_pos), axis=1)-ROBOT_LENGTH)
            plt.ylabel("Distance between robot centers [m]")
            plt.grid('on')
            plt.xlabel("Time [s]")
            plt.show()
        plt.plot(np.array(timestamps[1:]), plot_env.unwrapped.frame_rate *
                 np.linalg.norm(np.array(ego_robot_pos)[1:] - np.array(ego_robot_pos)[:-1], axis=1))
        if len(npc_robot_pos) > 0:
            plt.plot(np.array(timestamps[1:]), plot_env.unwrapped.frame_rate *
                     np.linalg.norm(np.array(npc_robot_pos)[1:] - np.array(npc_robot_pos)[:-1], axis=1))
            plt.legend(["Controlled robot", "Obstacle robot"])
        plt.ylabel("Robot speed [m/s]")
        plt.grid('on')
        plt.xlabel("Time [s]")
        plt.show()