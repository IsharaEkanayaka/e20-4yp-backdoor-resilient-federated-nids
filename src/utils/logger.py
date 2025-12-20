# src/utils/logger.py
import wandb
from omegaconf import OmegaConf

class Logger:
    def __init__(self, cfg, project_name="fl-nids-optimization"):
        """
        Initializes the W&B run using the Hydra config.
        """
        # Convert Hydra config to standard Python dict for W&B
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        
        # Check for existing runs to avoid conflicts
        if wandb.run is not None:
            wandb.finish()
            
        print(f"📊 Initializing W&B Logger for: {project_name}")
        
        self.run = wandb.init(
            project=project_name,
            config=config_dict,
            reinit=True,
            mode="online"  # Change to "disabled" for debugging without internet
        )

    def log_metrics(self, metrics, step=None):
        """
        Logs a dictionary of metrics.
        usage: logger.log_metrics({'acc': 90}, step=1)
        """
        if self.run is not None:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

    def finish(self):
        """
        Closes the W&B run.
        """
        if self.run is not None:
            print("📊 Finishing W&B Run...")
            wandb.finish()