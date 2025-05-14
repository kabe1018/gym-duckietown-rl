"""
Utilities to instantiate and configure a Duckietown environment, including the addition of wrappers.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import logging
from gym_duckietown.simulator import Simulator, DEFAULT_ROBOT_SPEED, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT
from gym import RewardWrapper
import math

from duckietown_utils.wrappers.observation_wrappers import *
from duckietown_utils.wrappers.action_wrappers import *
from duckietown_utils.wrappers.reward_wrappers import *
from duckietown_utils.wrappers.simulator_mod_wrappers import *
from duckietown_utils.wrappers.aido_wrapper import AIDOWrapper
from config.config import load_config

logger = logging.getLogger(__name__)

MAPSETS = {'multimap1': ['_custom_technical_floor', '_huge_C_floor', '_huge_V_floor', '_plus_floor',
                         'small_loop', 'small_loop_cw', 'loop_empty'],
           'multimap2': ['_custom_technical_floor', '_custom_technical_grass', 'udem1', 'zigzag_dists',
                         'loop_dyn_duckiebots'],
           'multimap_lfv': ['_custom_technical_floor_lfv', 'loop_dyn_duckiebots', 'loop_obstacles', 'loop_pedestrians'],
           'multimap_lfv_dyn_duckiebots': ['_loop_dyn_duckiebots_inner', '_loop_dyn_duckiebots_outer'],
           'multimap_lfv_duckiebots': ['_loop_duckiebots', '_loop_dyn_duckiebots_inner', '_loop_dyn_duckiebots_outer'],
           'multimap_aido5': ['LF-norm-loop', 'LF-norm-small_loop', 'LF-norm-zigzag', 'LF-norm-techtrack',
                              '_custom_technical_floor', '_huge_C_floor', '_huge_V_floor', '_plus_floor',
                              'small_loop', 'small_loop_cw', 'loop_empty'
                              ],
           }

CAMERA_WIDTH = DEFAULT_CAMERA_WIDTH
CAMERA_HEIGHT = DEFAULT_CAMERA_HEIGHT


def launch_and_wrap_env(env_config, default_env_id=0):
    env_id = env_config.get("worker_index", default_env_id)

    robot_speed = env_config.get('robot_speed', DEFAULT_ROBOT_SPEED)
    # If random robot speed is specified, the robot speed key holds a dictionary
    if type(robot_speed) is dict or robot_speed == 'default':
        robot_speed = DEFAULT_ROBOT_SPEED  # The initial robot speed won't be random

    # The while loop and try block are necessary to prevent instant training crash from the
    # "Exception: Could not find a valid starting pose after 5000 attempts" in duckietown-gym-daffy 5.0.13
    spawn_successful = False
    seed = 1234 + env_id
    while not spawn_successful:
        try:
            env = Simulator(
                seed=seed,  # random seed
                map_name=resolve_multimap_name(env_config["training_map"], env_id),
                max_steps=env_config.get("episode_max_steps", 500),
                domain_rand=env_config["domain_rand"],
                dynamics_rand=env_config["dynamics_rand"],
                camera_rand=env_config["camera_rand"],
                camera_width=CAMERA_WIDTH,
                camera_height=CAMERA_HEIGHT,
                accept_start_angle_deg=env_config["accepted_start_angle_deg"],
                full_transparency=True,
                distortion=env_config["distortion"],
                frame_rate=env_config["simulation_framerate"],
                frame_skip=env_config["frame_skip"],
                robot_speed=robot_speed
            )
            spawn_successful = True
        except Exception as e:
            seed += 1  # Otherwise it selects the same tile in the next attempt
            logger.error("{}; Retrying with new seed: {}".format(e, seed))
    logger.debug("Env init successful")
    env = wrap_env(env_config, env)
    return env


def resolve_multimap_name(training_map_conf, env_id):
    if 'multimap' in training_map_conf:
        mapset = MAPSETS[training_map_conf]
        map_name_single_env = mapset[env_id % len(mapset)]
    else:
        map_name_single_env = training_map_conf
    return map_name_single_env


def wrap_env(env_config: dict, env=None):
    if env is None:
        # Create a dummy Duckietown-like env if None was passed. This is mainly necessary to easily run
        # dts challenges evaluate
        env = DummyDuckietownGymLikeEnv()

    # Simulation mod wrappers
    if env_config["mode"] in ['train', 'debug'] and env_config['aido_wrapper']:
        env = AIDOWrapper(env)
    env = InconvenientSpawnFixingWrapper(env)
    if env_config.get('spawn_obstacles', False):
        env = ObstacleSpawningWrapper(env, env_config)
    if env_config.get('spawn_forward_obstacle', False):
        env = ForwardObstacleSpawnnigWrapper(env, env_config)
    if env_config['mode'] in ['train', 'debug']:
        if type(env_config.get('frame_skip')) is dict or type(env_config.get('robot_speed')) is dict:
            # Randomize frame skip or robot speed
            env = ParamRandWrapper(env, env_config)

        if isinstance(env_config.get('action_delay_ratio', 0.), float):
            if env_config.get('action_delay_ratio', 0.) > 0.:
                env = ActionDelayWrapper(env, env_config)
        if env_config.get('action_delay_ratio', 0.) == 'random':
            env = ActionDelayWrapper(env, env_config)

    # Observation wrappers
    if env_config["crop_image_top"]:
        env = ClipImageWrapper(env, top_margin_divider=env_config["top_crop_divider"])
    if env_config.get("grayscale_image", False):
        env = RGB2GrayscaleWrapper(env)
    env = ResizeWrapper(env, shape=env_config["resized_input_shape"])
    if env_config['mode'] in ['train', 'debug'] and env_config.get('frame_repeating', 0.0) > 0:
        env = RandomFrameRepeatingWrapper(env, env_config)
    if env_config["frame_stacking"]:
        env = ObservationBufferWrapper(env, obs_buffer_depth=env_config["frame_stacking_depth"])
    if env_config["mode"] in ['train', 'debug'] and env_config['motion_blur']:
        env = MotionBlurWrapper(env)
    env = NormalizeWrapper(env)

    # Füge diesen Wrapper hinzu, um von Channel-last auf channel-first zu konvertieren:
    env = ChannelsLast2ChannelsFirstWrapper(env)

    # Action wrappers
    if env_config["action_type"] == 'discrete':
        env = DiscreteWrapper(env)
    elif 'heading' in env_config["action_type"]:
        env = Heading2WheelVelsWrapper(env, env_config["action_type"])
    elif env_config["action_type"] == 'leftright_braking':
        env = LeftRightBraking2WheelVelsWrapper(env)
    elif env_config["action_type"] == 'leftright_clipped':
        env = LeftRightClipped2WheelVelsWrapper(env)
    elif env_config["action_type"] == 'steering_braking':
        env = SteeringBraking2WheelVelsWrapper(env)

    # Reward wrappers
    if env_config['mode'] in ['train', 'debug', 'inference']:
        if env_config["reward_function"] in ['Posangle', 'posangle']:
            env = DtRewardPosAngle(env)
            env = DtRewardVelocity(env)
        elif env_config["reward_function"] == 'target_orientation':
            env = DtRewardTargetOrientation(env)
            env = DtRewardVelocity(env)
        elif env_config["reward_function"] == 'lane_distance':
            env = DtRewardWrapperDistanceTravelled(env)
        elif env_config["reward_function"] == 'default_clipped':
            env = DtRewardClipperWrapper(env, 2, -2)
        elif env_config["reward_function"] == "custom":
            env = CustomRewardWrapper(env)
        else:  # Also env_config['mode'] == 'default'
            logger.warning("Default Gym Duckietown reward used")
        env = DtRewardCollisionAvoidance(env)
        # env = DtRewardProximityPenalty(env)

    # Und JETZT: am Ende den NaNCheckWrapper
    env = NaNCheckWrapper(env)


    return env


class DummyDuckietownGymLikeEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )
        self.road_tile_size = 0.585

    def reset(self):
        logger.warning("Dummy Duckietown Gym reset() called!")
        return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3))

    def step(self, action):
        logger.warning("Dummy Duckietown Gym step() called!")
        obs = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3))
        reward = 0.0
        done = True
        info = {}
        return obs, reward, done, info


def get_wrappers(wrapped_env):
    obs_wrappers = []
    action_wrappers = []
    reward_wrappers = []
    orig_env = wrapped_env
    while not (isinstance(orig_env, gym_duckietown.simulator.Simulator) or
               isinstance(orig_env, DummyDuckietownGymLikeEnv)):
        if isinstance(orig_env, gym.ObservationWrapper):
            obs_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.ActionWrapper):
            action_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.RewardWrapper):
            reward_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.Wrapper):
            None
        else:
            assert False, ("[duckietown_utils.env.get_wrappers] - {} Wrapper type is none of these:"
                           " gym.ObservationWrapper, gym.ActionWrapper, gym.ActionWrapper".format(orig_env))
        orig_env = orig_env.env

    return obs_wrappers[::-1], action_wrappers[::-1], reward_wrappers[::-1]


if __name__ == "__main__":
    # execute only if run as a script to test some functionality
    config = load_config('./config/config.yml')
    dummy_env = wrap_env(config['env_config'])
    obs_wrappers, action_wrappers, reward_wrappers = get_wrappers(dummy_env)
    print("Observation wrappers")
    print(*obs_wrappers, sep="\n")
    print("\nAction wrappers")
    print(*action_wrappers, sep="\n")
    print("\nReward wrappers")
    print(*reward_wrappers, sep="\n")




import gym
import numpy as np
from gym import RewardWrapper




class NaNCheckWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # Aktuelle Position prüfen **BEVOR** der nächste Step ausgeführt wird:
        pos = self.unwrapped.cur_pos
        if pos is not None:
            x, y, z = pos
            if not math.isfinite(x) or not math.isfinite(z):
                print("[WARN] NaN detected in position -> force done before step!")
                return self.reset(), -10.0, True, {"nan_crash": True}

        # Danach normalen Schritt ausführen:
        obs, reward, done, info = self.env.step(action)

        # Nochmal NaN-Prüfung nach dem Step
        pos = self.unwrapped.cur_pos
        if pos is not None:
            x, y, z = pos
            if not math.isfinite(x) or not math.isfinite(z):
                print("[WARN] NaN detected in position -> force done!")
                done = True
                reward -= 10.0
                info["nan_crash"] = True

        return obs, reward, done, info




def create_duckietown_env(config):
    # Erzeuge die Umgebung (z.B. mittels launch_and_wrap_env aus deiner env.py)
    env = launch_and_wrap_env(config)
    # Lies zusätzliche Parameter aus config, falls du den CustomRewardWrapper parametrisierbar machen möchtest.
   # if "reward_wrapper_params" in config:
        # Ersetze den Standard-Rewards-Wrapper durch deine parametrisierte Version:
    #    env = CustomRewardWrapper(env, **config["reward_wrapper_params"])
    return env




import gym
import math

class CustomRewardWrapper(gym.RewardWrapper):
    """
    Belohnungsfunktion für PPO, die das Spurhalten stark belohnt,
    schnelles Fahren weniger incentiviert und abrupte Richtungsänderungen
    bestraft (Reward Shaping für sanftere Bewegungen).

    Die Gesamt-Belohnung setzt sich zusammen aus:
    
      r_total = r_distance + orientation_weight * r_orientation 
                + velocity_weight * r_velocity + smoothness_penalty

    smoothness_penalty = - smoothness_weight * abs(psi_deg - prev_psi)
    """

    def __init__(self, env,
                 phi=50.0,               # Winkelgrenze (in Grad) für die Orientierung
                 eps=0.05,               # Strafwert, falls |psi_err| > phi
                 orientation_weight=2.0, # Verstärkung des Spurhalte-Rewards
                 velocity_weight=0.01,   # Gewichtung des Geschwindigkeits-Rewards
                 smoothness_weight=0.1   # Straffaktor für abrupte Änderungen in der Orientierung
                 ):
        super().__init__(env)
        self.phi = float(phi)
        self.eps = float(eps)
        self.orientation_weight = float(orientation_weight)
        self.velocity_weight = float(velocity_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.prev_psi = None  # Speicher für den vorherigen Orientierungswert

    def orientation_lambda(self, psi_deg):
        """
        Berechnet den Orientierungs-Rewards:
        
          Lambda(x) = 1/2 + 1/2 * cos(pi * x / phi)    , falls |x| <= phi
                    = -eps                              , sonst
        """
        x = abs(psi_deg)
        if x <= self.phi:
            return 0.5 + 0.5 * math.cos(math.pi * x / self.phi)
        else:
            return -self.eps

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        # Standardwerte, falls keine gültigen lane_position-Daten vorhanden sind (z. B. Off-track)
        distance_term = 0.0
        orientation_term = -self.eps  # Bei fehlender Information als Strafwert
        velocity_term = 0.0
        smoothness_penalty = 0.0

        sim_info = info.get('Simulator', {})
        lane_pos = sim_info.get('lane_position', None)

        if lane_pos is not None:
            # 1) DISTANCE-BASED:
            dot_dir = lane_pos.get('dot_dir', 0.0)
            speed   = sim_info.get('robot_speed', 0.0)
            distance_term = speed * dot_dir if dot_dir > 0 else 0.0

            # 2) ORIENTATION-BASED:
            psi_deg = lane_pos.get('angle_deg', 999.0)
            orientation_term = self.orientation_lambda(psi_deg)

            # 3) SMOOTHNESS-SHAPING:
            # Falls bereits ein vorheriger Winkel vorhanden ist, bestraft diese Komponente abrupte Änderungen.
            if self.prev_psi is not None:
                delta = abs(psi_deg - self.prev_psi)
                smoothness_penalty = -self.smoothness_weight * delta
            # Update des vorherigen Orientierungswertes
            self.prev_psi = psi_deg
        else:
            # Kein gültiger lane_pos: Zurücksetzen der vorherigen Orientierung
            self.prev_psi = None

        # 4) VELOCITY-BASED:
        wheel_vels = sim_info.get('wheel_velocities', [0.0, 0.0])
        velocity_term = max(wheel_vels[0], wheel_vels[1])

        # Gesamt-Belohnung: Die einzelnen Terme werden kombiniert.
        total_reward = (distance_term +
                        self.orientation_weight * orientation_term +
                        self.velocity_weight * velocity_term +
                        smoothness_penalty)

        return obs, float(total_reward), done, info




