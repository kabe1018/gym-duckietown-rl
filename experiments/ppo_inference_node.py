#!/usr/bin/env python3
import sys
sys.path.append('/workspace/Duckietown-RL')

import rospy
import cv2
import numpy as np
import time
import yaml

import gym
from gym.envs.registration import register

from ray.tune.registry import register_env
from duckietown_utils.env import launch_and_wrap_env, get_wrappers

# Ray/RLlib:
import ray
from ray.rllib.algorithms.ppo import PPO as PPOTrainer

# ROS-Nachrichten
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Pandas requires version")

class PPOInferenceNode:
    def __init__(self):
        # === Laden des RLlib-PPO-Modells ===
        config_file = "/workspace/Duckietown-RL/artifacts/Train_0029_0000/Feb19_18-38-58/config_dump_0000.yml"
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        if "memory" in config["ray_init_config"]:
            del config["ray_init_config"]["memory"]
        if "redis_max_memory" in config["ray_init_config"]:
            del config["ray_init_config"]["redis_max_memory"]

        ray.init(**config["ray_init_config"])
        register_env('Duckietown', launch_and_wrap_env)

        if "env_config" not in config["rllib_config"]:
            config["rllib_config"]["env_config"] = {}
        for k, v in config["env_config"].items():
            config["rllib_config"]["env_config"][k] = v
        config["rllib_config"]["env"] = "Duckietown"
        config["rllib_config"]["num_gpus"] = 0
        config["rllib_config"]["num_gpus_per_worker"] = 0

        self.trainer = PPOTrainer(config=config["rllib_config"])
        checkpoint_path = "/workspace/Duckietown-RL/artifacts/Train_0029_0000/Feb19_18-38-58/PPO_0_2025-02-19_18-39-02/checkpoint_000244/checkpoint-244"
        self.trainer.restore(checkpoint_path)
        rospy.loginfo("Modell wurde erfolgreich geladen von %s", checkpoint_path)

        # Dummy-Test: Erstelle eine Beobachtung der Form (84,84,3)
        dummy_obs = np.zeros((84, 84, 3), dtype=np.float32)
        test_action = self.trainer.compute_single_action(dummy_obs, explore=False)
        rospy.loginfo("Testaktion für Dummy-Eingabe: %s", str(test_action))

        # === ROS-Publisher & Subscriber ===
        duckiebot_name = "dagobert"  # anpassen, z. B. "mybot"
        camera_topic = f"/dagobert/camera_node/image/compressed"
        wheels_topic = f"/dagobert/wheels_driver_node/wheels_cmd"

        self.pub_wheels = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        self.sub_cam = rospy.Subscriber(camera_topic, CompressedImage, self.callback_image)

        # === Frame-Stacking Variablen ===
        self.frame_stack_depth = 3
        self.frame_buffer = []
        self.obs_shape = (84, 84)  # Zielgröße für die Bildverarbeitung

        # Parameter für die Steuerung
        self.base_speed = 0.15
        self.steering_scale = 0.5

    def callback_image(self, msg):
        rospy.loginfo("Callback triggered")
        # 1) Bild decodieren
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            rospy.logwarn("Empfangenes Bild konnte nicht decodiert werden.")
            return

        # 2) Top-Crop (oberes Drittel wegschneiden)
        H, W, _ = image.shape
        crop_h = H // 3
        image = image[crop_h:, :, :]

        # 3) In Graustufen umwandeln
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 4) Resize auf 84x84
        image = cv2.resize(image, self.obs_shape, interpolation=cv2.INTER_AREA)

        # 5) Normieren und in float32 wandeln
        image = image.astype(np.float32) / 255.0
        rospy.loginfo("Bild empfangen: Originalgröße: (%d, %d), Nach Crop & Resize: (%d, %d)", H, W, self.obs_shape[0], self.obs_shape[1])

        # 6) Frame in den Buffer einfügen
        self.frame_buffer.append(image)
        rospy.loginfo("Frame buffer Länge: %d", len(self.frame_buffer))
        if len(self.frame_buffer) < self.frame_stack_depth:
            rospy.loginfo("Warte auf weitere Frames...")
            return
        if len(self.frame_buffer) > self.frame_stack_depth:
            self.frame_buffer.pop(0)

        # 7) Frames stacken: aktueller Buffer hat Form (3,84,84)
        obs_stack = np.stack(self.frame_buffer, axis=0)
        rospy.loginfo("Gestapeltes Beobachtungsarray Form (vor Transposition): %s", str(obs_stack.shape))

        # 8) Transponieren, um die Kanaldimension ans Ende zu bringen: (84,84,3)
        obs = np.transpose(obs_stack, (1, 2, 0))
        rospy.loginfo("Beobachtungsarray für Modell (Form): %s", str(obs.shape))

        # 9) Aktion berechnen ohne Exploration
        action = self.trainer.compute_single_action(obs, explore=False)
        rospy.loginfo("Berechnete Aktion: %s", str(action))

        # 10) Aktion interpretieren: Annahme, dass 'action' ein Skalar (heading) ist
        heading = action[0] if isinstance(action, (list, np.ndarray)) else action
        vel_left = self.base_speed - self.steering_scale * heading
        vel_right = self.base_speed + self.steering_scale * heading

        # 11) WheelsCmd publizieren
        wheels_cmd = WheelsCmdStamped()
        wheels_cmd.vel_left = vel_left
        wheels_cmd.vel_right = vel_right
        self.pub_wheels.publish(wheels_cmd)
        rospy.loginfo("Action: %.3f -> L=%.3f, R=%.3f", heading, vel_left, vel_right)

def main():
    rospy.init_node('ppo_inference_node', anonymous=False)
    node = PPOInferenceNode()
    rospy.spin()

if __name__ == '__main__':
    main()
