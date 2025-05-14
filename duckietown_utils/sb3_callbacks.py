# duckietown_utils/sb3_callbacks.py
import logging
import numpy as np
import json
from stable_baselines3.common.callbacks import BaseCallback
from .trajectory_plot import correct_gym_duckietown_coordinates

# Globaler Speicher für Trainingsmetriken (wird von der Trainingsroutine befüllt)
LAST_TRAIN_INFO = {}

# Definiere Limits für Actions (wie in RLlib)
ACTION_HISTOGRAM_LIMITS = [-1., 1.]

logger = logging.getLogger(__name__)

class DuckietownSB3Callback(BaseCallback):
    def __init__(self, save_path="result.json", verbose=0):
        super(DuckietownSB3Callback, self).__init__(verbose)
        # Speichert Umgebungsmetriken pro Episode
        self.episode_data = dict()
        self.episode_summaries = []
        self.save_path = save_path
        # Rollout-Metriken (z. B. ep_len_mean, ep_rew_mean) werden hier abgelegt, falls vorhanden
        self.rollout_metrics = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", None)
        
        for env_idx, info in enumerate(infos):
            if info.get("episode_start", False):
                self.episode_data[env_idx] = {
                    "robot_speed": [],
                    "deviation_centerline": [],
                    "deviation_heading": [],
                    "distance_travelled": [],
                    "distance_travelled_any": [],
                    "proximity_penalty": [],
                    "collision_risk_step_cnt": 0,
                    "reward_orientation": [],
                    "reward_velocity": [],
                    "reward_collision_avoidance": [],
                    "sampled_actions": []
                }
            if "Simulator" in info:
                sim_info = info["Simulator"]
                data = self.episode_data.setdefault(env_idx, dict())
                if "robot_speed" in sim_info:
                    data.setdefault("robot_speed", []).append(sim_info["robot_speed"])
                if "proximity_penalty" in sim_info:
                    penalty = sim_info["proximity_penalty"]
                    data.setdefault("proximity_penalty", []).append(penalty)
                    if penalty < 0:
                        data["collision_risk_step_cnt"] = data.get("collision_risk_step_cnt", 0) + 1
                custom = info.get("custom_rewards", {})
                data.setdefault("reward_orientation", []).append(custom.get("orientation", 0))
                data.setdefault("reward_velocity", []).append(custom.get("velocity", 0))
                data.setdefault("reward_collision_avoidance", []).append(custom.get("collision_avoidance", 0))
                if "lane_position" in sim_info:
                    lp = sim_info["lane_position"]
                    data.setdefault("deviation_centerline", []).append(abs(lp.get("dist", 0)))
                    data.setdefault("deviation_heading", []).append(abs(lp.get("angle_deg", 0)))
                if actions is not None:
                    act = actions[env_idx]
                    clipped = np.clip(act, ACTION_HISTOGRAM_LIMITS[0], ACTION_HISTOGRAM_LIMITS[1])
                    data.setdefault("sampled_actions", []).append(clipped)
            if info.get("episode", False):
                data = self.episode_data.pop(env_idx, None)
                if data is not None:
                    mean_robot_speed = np.mean(data.get("robot_speed", [])) if data.get("robot_speed", []) else 0
                    deviation_centerline = np.mean(data.get("deviation_centerline", [])) if data.get("deviation_centerline", []) else 0
                    deviation_heading = np.mean(data.get("deviation_heading", [])) if data.get("deviation_heading", []) else 0
                    episode_summary = {
                        "mean_robot_speed": float(mean_robot_speed),
                        "deviation_centerline": float(deviation_centerline),
                        "deviation_heading": float(deviation_heading)
                    }
                    logger.info(f"Episode abgeschlossen: {episode_summary}")
                    self.episode_summaries.append(episode_summary)
        return True

    def _on_rollout_end(self):
        # Hier könntest du zusätzlich Rollout-Metriken erfassen, z.B. aus dem Monitor des Environments.
        try:
            monitor = self.training_env.envs[0]
            if hasattr(monitor, "ep_info_buffer") and monitor.ep_info_buffer:
                ep_info = monitor.ep_info_buffer
                ep_len_mean = np.mean([info.get("l", 0) for info in ep_info])
                ep_rew_mean = np.mean([info.get("r", 0) for info in ep_info])
                self.rollout_metrics = {
                    "episode_len_mean": float(ep_len_mean),
                    "episode_reward_mean": float(ep_rew_mean)
                }
            else:
                self.rollout_metrics = {}
        except Exception as e:
            logger.warning(f"Could not retrieve rollout metrics: {e}")
            self.rollout_metrics = {}

    def _on_training_end(self) -> None:
        # Nutze den globalen LAST_TRAIN_INFO, der von der Trainingsroutine aktualisiert wurde.
        from .sb3_callbacks import LAST_TRAIN_INFO  # importiert den globalen Speicher
        training_metrics = LAST_TRAIN_INFO or {}
        # Füge Rollout-Metriken hinzu (falls vorhanden)
        training_metrics.update(self.rollout_metrics)
        desired_train_keys = [
            "episode_len_mean", "episode_reward_mean", 
            "approx_kl", "clip_fraction", "explained_variance", 
            "loss", "policy_gradient_loss", "value_loss"
        ]
        filtered_train_metrics = {k: training_metrics.get(k, None) for k in desired_train_keys}
        result = {
            "episode_summaries": self.episode_summaries,
            "training_metrics": filtered_train_metrics
        }
        try:
            with open(self.save_path, "w") as f:
                json.dump(result, f, indent=4)
            logger.info(f"Episode summaries and training metrics saved to {self.save_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
