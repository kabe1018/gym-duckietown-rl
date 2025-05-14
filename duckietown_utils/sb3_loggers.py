"""
Custom SB3 Logger classes for logging images to Tensorboard
and logging various metrics and images to Weights & Biases.
Diese Klassen sind für den Einsatz in einem SB3-CustomCallback gedacht.
"""

import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import wandb
import numpy as np

logger = logging.getLogger(__name__)
weights_and_biases_project = 'duckietown-rllib'


def flatten_dict(d, parent_key='', sep='/'):
    """
    Hilfsfunktion, um verschachtelte Dictionaries zu einem flachen Dictionary zu vereinfachen.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class TensorboardImageLoggerSB3:
    """
    Logger, der Trainings-Trajektorien (z. B. als Plot) in Tensorboard schreibt.
    Diese Klasse verwendet tensorboardX.SummaryWriter.
    """
    def __init__(self, logdir):
        self._writer = SummaryWriter(logdir=logdir, filename_suffix="_img")

    def on_result(self, result: dict):
        # Wir erwarten, dass 'result' einen Schlüssel "hist_stats" enthält,
        # unter dem unter anderem _robot_coordinates abgelegt wurden.
        step = result.get("timesteps_total") or result.get("training_iteration", 0)
        try:
            traj_fig = plot_trajectories(result["hist_stats"]["_robot_coordinates"])
            traj_fig.savefig("Trajectory.png")
            self._writer.add_figure("TrainingTrajectories", traj_fig, global_step=step)
            plt.close(traj_fig)
        except Exception as e:
            logger.warning("Error while logging trajectory: %s", e)
        self.flush()

    def flush(self):
        if self._writer is not None:
            self._writer.flush()


class WeightsAndBiasesLoggerSB3:
    """
    Logger, der Metriken und Bilder an Weights & Biases sendet.
    """
    def __init__(self, config: dict, logdir):
        self.config = config
        self.logdir = logdir
        self.wandb_run = wandb.init(
            project=weights_and_biases_project,
            name=config['env_config']['experiment_name'],
            reinit=True
        )
        valid_config = config.copy()
        if "callbacks" in valid_config:
            del valid_config["callbacks"]
        valid_config = flatten_dict(valid_config, sep="/")
        self.wandb_run.config.update(valid_config, allow_val_change=True)

    def on_result(self, result: dict):
        step = result.get("timesteps_total") or result.get("training_iteration", 0)
        
        # Logge ausgewählte Scalar-Werte:
        logged_results = [
            'episode_reward_max', 'episode_reward_mean', 'episode_reward_min',
            'episode_len_mean', 'custom_metrics', 'sampler_perf', 'info', 'perf'
        ]
        result_copy = result.copy()
        for key in list(result_copy.keys()):
            if key not in logged_results:
                del result_copy[key]
        flat_result = flatten_dict(result_copy, sep="/")
        self.wandb_run.log(flat_result, step=step, sync=False)

        # Logge Histogramme aus "hist_stats" (außer _robot_coordinates)
        hist_stats = result.get("hist_stats", {})
        for key, val in hist_stats.items():
            try:
                if key != "_robot_coordinates":
                    self.wandb_run.log({f"Histograms/{key}": wandb.Histogram(val)}, step=step, sync=False)
            except ValueError:
                logger.warning("Unable to log histogram for %s", key)

        # Logge Trajektorien als Bild:
        try:
            traj_fig = plot_trajectories(hist_stats["_robot_coordinates"])
            traj_fig.savefig("Trajectory.png")
            self.wandb_run.log({'Episode Trajectories': wandb.Image(traj_fig)}, step=step, sync=False)
            plt.close(traj_fig)
        except Exception as e:
            logger.warning("Error while logging trajectory to wandb: %s", e)

    def close(self):
        wandb.join()
