"""
Gym wrapper classes that transform the observations from Gym-Duckietown to alternative representations.
- Clipping
- Resizing
- Normalization
- Stacking buffer
- Grayscale images
- Simple, convolution based motion blur simulation
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import ast
import gym
import cv2
from gym import spaces
import numpy as np
import logging
from gym_duckietown.simulator import CAMERA_FOV_Y
from gym.spaces import Box


logger = logging.getLogger(__name__)


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=(64, 64)):
        super(ResizeWrapper, self).__init__(env)
        
        # Falls shape ein String ist, parsen:
        if isinstance(shape, str):
            shape = ast.literal_eval(shape)  # z.B. "(64,64)" in ein Tupel konvertieren

        # Falls in shape noch Strings statt Integer stehen:
        shape = [int(x) for x in shape]
        self.shape = tuple(shape)

        # Verwende die aktuelle Anzahl der Kanäle aus dem bestehenden observation_space.
        # Falls diese nicht definiert sein sollte, gehe von 1 Kanal aus.
        if len(self.observation_space.shape) >= 3:
            channels = self.observation_space.shape[2]
        else:
            channels = 1

        self.observation_space = Box(
            low=0, 
            high=255, 
            shape=(self.shape[0], self.shape[1], channels), 
            dtype=np.uint8
        )


    def observation(self, observation):
        # Falls mehr Kanäle vorhanden sind als erwartet, schneide auf die ersten channels zu.
        channels_expected = self.observation_space.shape[2]
        if observation.ndim == 3 and observation.shape[2] > channels_expected:
            observation = observation[..., :channels_expected]

        # Resizing auf die gewünschte (Höhe, Breite)
        resized = cv2.resize(observation, self.shape[::-1], interpolation=cv2.INTER_AREA)
        # Auf gültigen Bereich [0,255] clippen und in uint8 konvertieren
        resized = np.clip(resized, 0, 255).astype(np.uint8)
        return resized



class ClipImageWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, top_margin_divider=3):
        super(ClipImageWrapper, self).__init__(env)
        img_height, img_width, depth = self.observation_space.shape
        top_margin = img_height // top_margin_divider
        img_height = img_height - top_margin
        # Region Of Interest
        # r = [margin_left, margin_top, width, height]
        self.roi = [0, top_margin, img_width, img_height]

        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (img_height, img_width, depth),
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        r = self.roi
        observation = observation[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        return observation

class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)
        assert self.observation_space.contains(np.zeros(obs_shape, dtype=np.float32)), "Invalid observation_space in NormalizeWrapper!"


    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            assert self.observation_space.contains(obs), "Normalized observation out of bounds!"
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ChannelsLast2ChannelsFirstWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ChannelsLast2ChannelsFirstWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class RGB2GrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(RGB2GrayscaleWrapper, self).__init__(env)
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (self.observation_space.shape[0], self.observation_space.shape[1], 1),
            dtype=self.observation_space.dtype)

    def observation(self, obs):
        # cv2.imshow("Camera", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("Camera2", gray)
        # cv2.waitKey(0)

        # Add an extra dimension, because conv lasers need an input as (batch, height, width, channels)
        gray = np.expand_dims(gray, 2)
        return gray




class ObservationBufferWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, obs_buffer_depth=3):
        super(ObservationBufferWrapper, self).__init__(env)
        obs_space_shape_list = list(self.observation_space.shape)
        
        # Falls der Beobachtungsraum 2D ist, füge eine Kanaldimension hinzu.
        if len(obs_space_shape_list) == 2:
            obs_space_shape_list.append(1)
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=tuple(obs_space_shape_list),
                dtype=np.uint8
            )
        
        # Die letzte Dimension ist die Kanaldimension.
        self.buffer_axis = len(obs_space_shape_list) - 1
        
        # Berechne die neue Kanalanzahl für das gestapelte Observation:
        original_channels = obs_space_shape_list[self.buffer_axis]
        new_channels = original_channels * obs_buffer_depth
        obs_space_shape_list[self.buffer_axis] = new_channels
        
        # Erstelle das neue observation_space, indem du low und high entlang der Kanalachse konkatenierst.
        low = np.concatenate([self.observation_space.low] * obs_buffer_depth, axis=self.buffer_axis)
        high = np.concatenate([self.observation_space.high] * obs_buffer_depth, axis=self.buffer_axis)
        self.observation_space = spaces.Box(low=low, high=high, shape=tuple(obs_space_shape_list), dtype=self.observation_space.dtype)
        
        self.obs_buffer_depth = obs_buffer_depth
        self.obs_buffer = None

    def observation(self, obs):
        # Falls das Eingangsobservation 2D ist (also z. B. bei Graustufen), erweitere es auf 3D.
        if obs.ndim == 2:
            obs = np.expand_dims(obs, axis=-1)
        
        # Falls kein Buffer verwendet wird, gebe das einzelne Observation zurück.
        if self.obs_buffer_depth == 1:
            return obs
        
        if self.obs_buffer is None:
            # Beim ersten Mal: Erstelle den Buffer durch Konkatenation mehrerer Kopien des aktuellen Frames.
            self.obs_buffer = np.concatenate([obs for _ in range(self.obs_buffer_depth)], axis=self.buffer_axis)
        else:
            # Beim Update: Entferne den ältesten Frame und füge das neue Frame hinzu.
            # Bestimme die Anzahl der Kanäle pro Frame:
            c = obs.shape[self.buffer_axis]
            # Schneide den Buffer so, dass die ersten c Kanäle entfernt werden,
            # und konkatenieren dann das neue Frame.
            self.obs_buffer = np.concatenate((self.obs_buffer[..., c:], obs), axis=self.buffer_axis)
        
        # Prüfe, ob der resultierende Buffer dem definierten observation_space entspricht.
        assert self.observation_space.contains(self.obs_buffer), (
            f"Buffered observation out of bounds! Buffer shape: {self.obs_buffer.shape} "
            f"vs expected: {self.observation_space.shape}"
        )
        return self.obs_buffer

    def reset(self, **kwargs):
        self.obs_buffer = None
        observation = self.env.reset(**kwargs)
        return self.observation(observation)






class MotionBlurWrapper(gym.ObservationWrapper):
    """
    Simulates motion blur separately for horizontal (left-right) rotational and forward movement.
    Both are simulated using a smoothing convolutional filter (with varying sized "horizontal" or "vertical" kernel).
    Forward motion should produce non-uniform blur (none at the center point of the horizon, and larger towards the
     edges), here it is simulated using the same filtering over the whole image (the size of the kernel depends on the
     robot velocity.

    According to gluPerspective (used by the Simulator) x is width, y is height.
    """
    def __init__(self, env):
        super(MotionBlurWrapper, self).__init__(env)
        self.camera_fov_height_angle = self.unwrapped.cam_fov_y / 180. * np.pi
        self.camera_fov_width_angle =\
            self.unwrapped.camera_width / float(self.unwrapped.camera_height) * self.camera_fov_height_angle
        self.camera_height_px = self.observation_space.shape[0]
        self.camera_width_px = self.observation_space.shape[1]
        self.blur_time = 0.05
        self.prev_angle = self.unwrapped.cur_angle
        self.simulate_rotational_blur = True
        self.simulate_forward_blur = False  # Vertical convolution doesn't work well for this

    def observation(self, observation):
        if self.simulate_rotational_blur:
            cur_angle = self.unwrapped.cur_angle
            angular_vel = self._angle_diff(self.prev_angle, cur_angle) * self.unwrapped.frame_rate * self.unwrapped.frame_skip
            self.prev_angle = cur_angle
            delta_angle = angular_vel * self.blur_time
            if abs(delta_angle) > 0:
                ksize = np.round(np.abs(delta_angle / self.camera_fov_width_angle * self.camera_width_px)).astype(int) + 1
                logger.debug("Rotational motion blur kernel size {}".format(ksize))
                kernel = np.zeros((ksize, ksize))
                kernel[ksize // 2, :] = 1. / ksize
                observation = cv2.filter2D(observation, -1, kernel)
        if self.simulate_forward_blur:
            # Empirical kernel size, proportional to the current speed over the max speed of the robot
            ksize = np.round(np.abs(self.camera_width_px / 30 * self.unwrapped.speed / self.unwrapped.robot_speed)).astype(int) + 1
            logger.debug("Forward motion blur kernel size {}".format(ksize))
            kernel = np.zeros((ksize, ksize))
            kernel[:, ksize // 2] = 1. / ksize
            observation = cv2.filter2D(observation, -1, kernel)
        return observation

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.prev_angle = self.unwrapped.cur_angle
        return self.observation(observation)

    @staticmethod
    def _angle_diff(x, y):
        """
        Angle difference between two angles (orientations, directions). The smallest signed value is returned.
        The difference is measured from x to y (the difference is positive if y is larger)
        :param x, y: angles in radians
        :return: smallest angle difference between the two angles, should be in the range [-pi, pi]
        """
        diff = y-x
        remainder = diff % (2 * np.pi)
        quotient = diff // (2 * np.pi)
        if remainder > np.pi:
            diff -= 2 * np.pi
        diff -= quotient * 2 * np.pi
        return diff


class RandomFrameRepeatingWrapper(gym.ObservationWrapper):
    """
    Implements an evil domain randomisation option, replaces some observations with the previous.
    env_config["frame_repeating"]
    Values: 0...0.99 specify the probability of repeatingg
    """

    def __init__(self, env, env_config):
        super(RandomFrameRepeatingWrapper, self).__init__(env)
        self.repeat_config = np.clip(env_config["frame_repeating"], 0, 0.999)
        self.previous_frame = None

    def observation(self, observation):
        if self.previous_frame is None:
            self.previous_frame = observation
            return observation
        if np.random.random() < self.repeat_config:
            # New observation "not received" keeping the last one
            observation = self.previous_frame
        else:
            # New observation "received", store and don't change it
            self.previous_frame = observation
        return observation

    def reset(self, **kwargs):
        self.previous_frame = None
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

