import logging
import wandb
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class ConsoleLogger:
    """Simple console logger for when wandb is disabled."""
    def __init__(self, project_name, experiment_name):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.start_time = time.time()
        self.metrics_history = {}

        # Print header using standard logger
        logger.info("=" * 50)
        logger.info(f"Project: {project_name} | Experiment: {experiment_name}")
        logger.info("=" * 50)

    def log(self, metrics, step=None):
        """Log metrics using Python's logger."""
        # Update history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

        # Log each metric on a separate line
        logger.info("-" * 30 + " Metrics " + "-" * 30)
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")

        # Log elapsed time
        elapsed = time.time() - self.start_time
        logger.info(f"Elapsed time: {int(elapsed // 60)}m {int(elapsed % 60)}s")

    def finish(self):
        """Print summary of the run."""
        logger.info("=" * 25 + " Run Summary " + "=" * 25)
        logger.info(f"Project: {self.project_name} | Experiment: {self.experiment_name}")

        # Calculate final metrics (last value for each)
        final_metrics = {k: v[-1] for k, v in self.metrics_history.items()}

        # Log final metrics
        for key, value in final_metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")

        # Total runtime
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")


# Global variable to store the active logger (wandb or console)
active_logger = None

def init_wandb(config):
    """Initialize wandb or fallback to console logger."""
    global active_logger

    if not hasattr(config.wandb, 'use_wandb') or not config.wandb.use_wandb:
        logger.info("WandB disabled. Using console logger.")
        active_logger = ConsoleLogger(
            project_name=config.wandb.get('project', 'Pawpularity'),
            experiment_name=config.exp_name
        )
        return None

    logger.info("Initializing wandb")
    wandb.init(
        project=config.wandb.project,
        name=config.exp_name,
        config=dict(config),
    )
    active_logger = wandb.run
    return wandb.run

def watch_model(model, log_freq=100):
    """Set wandb to watch the model training."""
    if wandb.run is None:
        return

    wandb.watch(model, log="all", log_freq=log_freq)

def log_metrics(metrics, step=None):
    """Log metrics to wandb or console."""
    global active_logger

    if wandb.run is not None:
        wandb.log(metrics, step=step)
    elif active_logger is not None:
        active_logger.log(metrics, step=step)

def log_model_artifact(model_path, artifact_name):
    """Log model as an artifact."""
    if wandb.run is None:
        logger.info(f"Model saved to {model_path} (WandB disabled, no artifact logging)")
        return

    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)

def finish():
    """Finish the wandb run or console logger."""
    global active_logger

    if wandb.run is not None:
        wandb.finish()
    elif active_logger is not None:
        active_logger.finish()
