#!/usr/bin/env python3
# duckiebot_trt_agent.py
"""
ROS-Knoten: Duckietown-PPO-Policy (TensorRT) auf dem realen Duckiebot

* Abo:  /<robot>/camera_node/image/(raw|compressed)
* Pub:  /<robot>/car_cmd_switch_node/cmd      (Twist2DStamped)
* Vorverarbeitung 1-zu-1 aus dem Training
"""

import numpy as np
np.bool = bool

import os
import rospy
import yaml
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit   # initialisiert CUDA-Runtime
import collections

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from duckietown_msgs.msg import Twist2DStamped

import gym
from gym import spaces


class DummyEnv(gym.Env):
    def __init__(self, img):
        self._img = img
        self.observation_space = spaces.Box(0, 255, img.shape, dtype=np.uint8)
    def reset(self, **_):
        return self._img


class DuckiebotTrtAgent:
    def __init__(self):
        rospy.init_node("duckiebot_trt_agent")

        #default_cfg    = "/workspace/Duckietown-RL/config_backups/grayscale/config_dump_0000.yml"
        #default_engine = "/workspace/Duckietown-RL/new_export/grayscale.plan"

        #default_cfg    = "/workspace/Duckietown-RL/config_backups/dr/config_dump_0000.yml"
        #default_engine = "/workspace/Duckietown-RL/new_export/dr.plan"

        default_cfg    = "/workspace/Duckietown-RL/config_backups/new_grayscale/config_dump_0000.yml"
        default_engine = "/workspace/Duckietown-RL/new_export/new_grayscale.plan"

       # default_cfg    = "/workspace/Duckietown-RL/config_backups/no_randomization/config_dump_0000.yml"
       # default_engine = "/workspace/Duckietown-RL/new_export/no_randomization.plan"
       

        #default_cfg    = "/workspace/Duckietown-RL/config_backups/gray_dr/config_dump_0000.yml"
        #default_engine = "/workspace/Duckietown-RL/new_export/gray_dr.plan"

        #default_cfg    = "/workspace/Duckietown-RL/config_backups/real_data_gray_new_rew/config_dump_0000.yml"
        #default_engine = "/workspace/Duckietown-RL/new_export/real_gray_new.plan"

        #default_cfg    = "/workspace/Duckietown-RL/config_backups/real_data_gray_old_rew/config_dump_0000.yml"
        #default_engine = "/workspace/Duckietown-RL/new_export/real_gray_old.plan"



        cfg_path    = rospy.get_param("~config_path", default_cfg)
        engine_path = rospy.get_param("~engine_path", default_engine)
        self.speed  = rospy.get_param("~robot_speed", 0.22)

        self.image_topic = rospy.get_param("~image_topic",
                                          "/daffy/camera_node/image/compressed")
        self.cmd_topic   = rospy.get_param("~cmd_topic",
                                          "/daffy/car_cmd_switch_node/cmd")

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.env_cfg = cfg["env_config"]
        self.env_cfg["mode"] = "inference"

        # TensorRT laden
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime    = trt.Runtime(TRT_LOGGER)
        rospy.loginfo("[TRT] Lade Engine: %s", engine_path)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine  = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Bindings & CUDA-Puffer
        self.input_binding_idx  = next(i for i in range(self.engine.num_bindings)
                                       if self.engine.binding_is_input(i))
        self.output_binding_idx = next(i for i in range(self.engine.num_bindings)
                                       if not self.engine.binding_is_input(i))

        in_shape  = tuple(self.engine.get_binding_shape(self.input_binding_idx))
        out_shape = tuple(self.engine.get_binding_shape(self.output_binding_idx))

        self.h_input   = cuda.pagelocked_empty(trt.volume(in_shape), dtype=np.float32)
        self.h_output  = cuda.pagelocked_empty(trt.volume(out_shape), dtype=np.float32)
        self.d_input   = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output  = cuda.mem_alloc(self.h_output.nbytes)
        self.bindings  = [int(self.d_input), int(self.d_output)]

        # ROS-Kommunikation
        self.bridge   = CvBridge()
        self.cmd_pub  = rospy.Publisher(self.cmd_topic,
                                       Twist2DStamped,
                                       queue_size=1)

        msg_type = CompressedImage if self.image_topic.endswith("/compressed") else Image
        self.image_sub = rospy.Subscriber(self.image_topic,
                                          msg_type,
                                          self.image_cb,
                                          queue_size=1)

        self.depth = (self.env_cfg.get("frame_stacking_depth")
                      if self.env_cfg.get("frame_stacking", False) else 1)
        self.frame_buffer = collections.deque(maxlen=self.depth)

        self.last_obs = None
        rospy.on_shutdown(self.on_shutdown)

        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)
        rospy.loginfo("✓  duckiebot_trt_agent läuft – %s → %s",
                      self.image_topic, self.cmd_topic)

    def image_cb(self, msg):
        try:
            if isinstance(msg, CompressedImage):
                bgr = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            else:
                bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr_throttle(2.0, "CV-Bridge Fehler: %s", e)
            return

        obs = self._preprocess(bgr)
        self.last_obs = obs

    def control_loop(self, _event):
        if self.last_obs is None:
            return

        # Ensure the PyCUDA context is active in this thread
        import pycuda.autoinit as _autoinit
        _autoinit.context.push()
        try:
            # Host-Buffer füllen
            inp = self.last_obs[None].astype(np.float32)
            np.copyto(self.h_input, inp.ravel())

            # Daten auf GPU & Inferenz
            cuda.memcpy_htod(self.d_input, self.h_input)
            self.context.execute_v2(self.bindings)
            cuda.memcpy_dtoh(self.h_output, self.d_output)
        finally:
            _autoinit.context.pop()

        # Ausgabe verarbeiten
        out = self.h_output.reshape(
            self.engine.get_binding_shape(self.output_binding_idx))
        act = float(out[0, 0])

        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v     = self.speed
        cmd.omega = act
        self.cmd_pub.publish(cmd)
        
        # Debug: zeige die Steuerkommandos im ROS-Log an
        rospy.loginfo("→ Steuerkommando: v=%.3f m/s, ω=%.3f rad/s", cmd.v, cmd.omega)
        
   
    def _preprocess(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 1) Croppen
        if self.env_cfg.get("crop_image_top", False):
            h = rgb.shape[0]
            top = h // self.env_cfg["top_crop_divider"]
            rgb = rgb[top:, :, :]

        # 2) Graustufen oder Farbbild
        if self.env_cfg.get("grayscale_image", False):
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb

        # 3) Resize
        target = tuple(self.env_cfg["resized_input_shape"][::-1])
        gray   = cv2.resize(gray, target, interpolation=cv2.INTER_AREA)

        # Sicherstellen, dass gray 3D ist
        if gray.ndim == 2:
            gray = gray[:, :, None]

        # 4) Frame-Stacking
        self.frame_buffer.append(gray)
        while len(self.frame_buffer) < self.depth:
            self.frame_buffer.append(gray.copy())
        stacked = np.concatenate(list(self.frame_buffer), axis=2)

        # 5) Normalisieren + Channel-first
        norm    = stacked.astype(np.float32) / 255.0
        c_first = np.moveaxis(norm, 2, 0)
        return c_first

    def on_shutdown(self):
        rospy.loginfo("Shutdown: Stoppe Duckiebot.")
        stop = Twist2DStamped()
        stop.v, stop.omega = 0.0, 0.0
        self.cmd_pub.publish(stop)


if __name__ == "__main__":
    try:
        DuckiebotTrtAgent()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass



