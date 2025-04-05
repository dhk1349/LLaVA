import wandb
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


class LlavaLogger:
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: Dict[str, Any],
        description: str = "",
        tags: Optional[list] = None,
    ):
        """Initialize WandB logger with structured metadata.

        Args:
            project_name: Name of the WandB project
            experiment_name: Name of this specific experiment
            config: Dictionary containing model and training configuration
            description: Detailed description of the experiment
            tags: List of tags for easy filtering
        """
        self.run = wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            notes=self._format_description(description),
            tags=tags or [],
        )

        # Save experiment metadata
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "config": config,
        }

    def _format_description(self, description: str) -> str:
        """Format experiment description with mandatory sections."""
        template = """
        ## Experiment Description
        {description}
        
        ## Changes from Previous Version
        [Please describe key changes from the previous version]
        
        ## Expected Outcomes
        [Please describe expected outcomes and hypotheses]
        
        ## Notes
        [Additional notes, observations, or concerns]
        """
        return template.format(description=description)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training/validation metrics."""
        wandb.log(metrics, step=step)

    def log_model_checkpoint(self, checkpoint_path: str, metadata: Dict[str, Any]):
        """Log model checkpoint with metadata."""
        artifact = wandb.Artifact(name=f"model-checkpoint-{wandb.run.id}", type="model", metadata=metadata)
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

    def log_dataset_version(self, dataset_path: str, version: str, metadata: Dict[str, Any]):
        """Log dataset version information."""
        artifact = wandb.Artifact(name=f"dataset-{version}", type="dataset", metadata=metadata)
        artifact.add_dir(dataset_path)
        wandb.log_artifact(artifact)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters with their descriptions."""
        for key, value in params.items():
            if isinstance(value, dict) and "value" in value and "description" in value:
                wandb.config.update({key: value["value"]})
                # Store description in run metadata
                self.metadata.setdefault("hyperparameter_descriptions", {})[key] = value["description"]
            else:
                wandb.config.update({key: value})

    def finish(self):
        """Save final metadata and close the run."""
        # Save metadata as JSON artifact
        metadata_path = f"metadata_{wandb.run.id}.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        artifact = wandb.Artifact(name=f"experiment-metadata-{wandb.run.id}", type="metadata")
        artifact.add_file(metadata_path)
        wandb.log_artifact(artifact)

        wandb.finish()

        # Cleanup
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
