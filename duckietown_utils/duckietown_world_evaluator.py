"""
DuckietownWorldEvaluator evaluates a SB3 agent using the same evaluator (duckietown-world)
that is used in Duckietown's official evaluator (DTS evaluate).
Duckietown World takes trajectory data recorded on one of its built-in maps.
For adapting to a different agent implementation than RLlib (e.g. SB3),
override _compute_action (and _collect_trajectory if needed).
Custom maps should be copied to the installation folder of Duckietown World.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import os
import numpy as np
import json
import time
import logging
import copy
from pprint import pformat

from duckietown_world import SE2Transform
from duckietown_world.rules import evaluate_rules
from duckietown_world.rules.rule import EvaluatedMetric, make_timeseries
from duckietown_world.seqs.tsequence import SampledSequence
from duckietown_world.svg_drawing import draw_static
from duckietown_world.world_duckietown.duckiebot import DB18
from duckietown_world.world_duckietown.map_loading import load_map

from duckietown_utils.env import launch_and_wrap_env
from duckietown_utils.trajectory_plot import correct_gym_duckietown_coordinates

DEFAULT_EVALUATION_MAP = 'ETHZ_autolab_technical_track'
myTestMapA = '_myTestA'
logger = logging.getLogger(__name__)


class DuckietownWorldEvaluator:
    """
    Evaluates a SB3 agent using the same evaluator as in DTS evaluate.
    To adapt for a different agent implementation than RLlib, override _compute_action.
    """
    # These start poses are exactly the same as used by dts evaluate
    # Pose description: [[x, 0, z], rot] in meters and radians.
    start_poses = {
        'ETHZ_autolab_technical_track': [
            [[0.7019999027252197, 0, 0.41029359288296474], -0.2687807048071267],
            [[0.44714101540138385, 0, 2.2230001401901243], 1.31423292675173],
            [[1.5552862449923595, 0, 1.0529999446868894], 1.4503686084072878],
            [[1.6380000114440918, 0, 3.0929162652880935], -3.0264009229581674],
            [[0.3978251477698904, 0, 2.8080000591278074], 1.2426744274199626],
        ],
        '_myTestA': [
            [[0.585 / 4, 0, 0.585], -np.pi / 2],
            [[0.585 / 4 * 3, 0, 0.585 * 2], +np.pi / 2]
        ]
    }

    def __init__(self, env_config, eval_lenght_sec=15, eval_map=DEFAULT_EVALUATION_MAP):
        _env_config = copy.deepcopy(env_config)
        # An official evaluation episode is 15 seconds long.
        _env_config['episode_max_steps'] = eval_lenght_sec * _env_config['simulation_framerate']
        # Evaluate on the official eval map.
        _env_config['training_map'] = eval_map
        self.map_name = _env_config['training_map']
        # Create testing environment.
        self.env = launch_and_wrap_env(_env_config)

        # Set up evaluator: create a duckiebot object.
        self.ego_name = 'duckiebot'
        self.db = DB18()  # Provides the appearance.
        # Load one of the maps.
        self.dw = load_map(self.map_name)

    def evaluate(self, agent, outdir, episodes=None):
        """
        Evaluates the agent on the map initialized in __init__.
        :param agent: SB3 model to be evaluated.
        :param outdir: Directory for outputs (trajectory plots + numeric data).
        :param episodes: Number of evaluation episodes; if None, use the number of predefined start poses.
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if episodes is None:
            episodes = len(self.start_poses.get(self.map_name, []))
        totals = {}
        for i in range(episodes):
            episode_path, episode_orientations, episode_timestamps = self._collect_trajectory(agent, i)
            logger.info("Episode {}/{} sampling completed".format(i+1, episodes))
            if len(episode_timestamps) <= 1:
                continue
            episode_path = np.stack(episode_path)
            episode_orientations = np.stack(episode_orientations)
            # Convert to a SampledSequence.
            transforms_sequence = []
            for j in range(len(episode_path)):
                transforms_sequence.append(SE2Transform(episode_path[j], episode_orientations[j]))
            transforms_sequence = SampledSequence.from_iterator(enumerate(transforms_sequence))
            transforms_sequence.timestamps = episode_timestamps

            _outdir = outdir
            if outdir is not None and episodes > 1:
                _outdir = os.path.join(outdir, "Trajectory_{}".format(i+1))
            evaluated = self._eval_poses_sequence(transforms_sequence, outdir=_outdir)
            logger.info("Episode {}/{} plotting completed".format(i+1, episodes))
            totals = self._extract_total_episode_eval_metrics(evaluated, totals, display_outputs=True)

        # Calculate aggregated metrics.
        median_totals = {key: np.median(value) for key, value in totals.items()}
        mean_totals   = {key: np.mean(value) for key, value in totals.items()}
        stdev_totals  = {key: np.std(value) for key, value in totals.items()}
        with open(os.path.join(outdir, "total_metrics.json"), "w") as json_file:
            json.dump({'median_totals': median_totals,
                       'mean_totals': mean_totals,
                       'stdev_totals': stdev_totals,
                       'episode_totals': totals}, json_file, indent=2)

        logger.info("\nMedian total metrics: \n {}".format(pformat(median_totals)))
        logger.info("\nMean total metrics: \n {}".format(pformat(mean_totals)))
        logger.info("\nStandard deviation of total metrics: \n {}".format(pformat(stdev_totals)))

    def _collect_trajectory(self, agent, i):
        episode_path = []
        episode_orientations = []
        episode_timestamps = []
        if self.map_name in self.start_poses and i < len(self.start_poses.get(self.map_name, [])):
            self.env.unwrapped.user_tile_start = [0, 0]
            self.env.unwrapped.start_pose = self.start_poses[self.map_name][i]
        else:
            self.env.unwrapped.user_tile_start = None
            self.env.unwrapped.start_pose = None
        obs = self.env.reset()
        done = False
        while not done:
            action = self._compute_action(agent, obs)
            obs, reward, done, info = self.env.step(action)
            cur_pos = correct_gym_duckietown_coordinates(self.env.unwrapped, self.env.unwrapped.cur_pos)
            episode_path.append(cur_pos)
            episode_orientations.append(np.array(self.env.unwrapped.cur_angle))
            episode_timestamps.append(info['Simulator']['timestamp'])
        self.env.unwrapped.start_pose = None
        self.user_tile_start = None
        return episode_path, episode_orientations, episode_timestamps

    def _compute_action(self, agent, obs):
        """
        Computes action for a given observation.
        For SB3, we assume 'agent' is a SB3 model and use its predict() method.
        """
        action, _ = agent.predict(obs, deterministic=True)
        return action

    def _eval_poses_sequence(self, poses_sequence, outdir=None):
        self.dw.set_object(self.ego_name, self.db, ground_truth=poses_sequence)
        interval = SampledSequence.from_iterator(enumerate(poses_sequence.timestamps))
        evaluated = evaluate_rules(poses_sequence=poses_sequence,
                                   interval=interval, world=self.dw, ego_name=self.ego_name)
        if outdir is not None:
            timeseries = make_timeseries(evaluated)
            draw_static(self.dw, outdir, timeseries=timeseries)
        print(self.dw.get_drawing_children())
        self.dw.remove_object(self.ego_name)
        self.dw.remove_object('visualization')
        return evaluated

    @staticmethod
    def _extract_total_episode_eval_metrics(evaluated, totals, display_outputs=False):
        episode_totals = {}
        for k, rer in evaluated.items():
            from duckietown_world.rules import RuleEvaluationResult
            assert isinstance(rer, RuleEvaluationResult)
            for km, evaluated_metric in rer.metrics.items():
                assert isinstance(evaluated_metric, EvaluatedMetric)
                episode_totals[k] = evaluated_metric.total
                if k not in totals:
                    totals[k] = [evaluated_metric.total]
                else:
                    totals[k].append(evaluated_metric.total)
        if display_outputs:
            logger.info("\nEpisode total metrics: \n {}".format(pformat(episode_totals)))
        return totals
