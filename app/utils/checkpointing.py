"""Checkpoint management for saving, loading, and resuming training."""

import json
import shutil
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch


class CheckpointManager:
    """
    Manages checkpoint saving, loading, and run directories.

    Features:
    - Timestamped run directories (never overwrite)
    - Configurable save frequency
    - Retention policy (keep last N checkpoints)
    - Best checkpoint tracking
    - Full state restoration (model, optimizer, RNG)
    """

    def __init__(
        self,
        experiment_name: str,
        output_base: str = "./outputs",
        save_every_n_steps: int = 50,
        keep_last_n: int = 5,
        keep_best: bool = True,
    ):
        """
        Initialize checkpoint manager.

        Args:
            experiment_name: Name of the experiment
            output_base: Base directory for outputs
            save_every_n_steps: Save checkpoint every N steps
            keep_last_n: Number of recent checkpoints to keep
            keep_best: Whether to keep best checkpoint separately
        """
        self.experiment_name = experiment_name
        self.output_base = Path(output_base)
        self.save_every_n_steps = save_every_n_steps
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best

        self.run_id: Optional[str] = None
        self.run_dir: Optional[Path] = None
        self.best_metric: float = float("-inf")

    def create_new_run(self, config: Optional[Dict] = None) -> Path:
        """
        Create a new run directory with timestamp.

        Args:
            config: Optional config dict to save

        Returns:
            Path to the new run directory
        """
        self.run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = self.output_base / self.experiment_name / self.run_id

        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        # Save run info
        run_info = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "status": "running",
        }
        with open(self.run_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)

        # Save config if provided
        if config:
            with open(self.run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2, default=str)

        return self.run_dir

    def get_experiment_dir(self) -> Path:
        """Get experiment directory."""
        return self.output_base / self.experiment_name

    def get_latest_run(self) -> Optional[Path]:
        """Find the most recent run directory."""
        exp_dir = self.get_experiment_dir()
        if not exp_dir.exists():
            return None

        runs = sorted(exp_dir.glob("run_*"))
        return runs[-1] if runs else None

    def get_latest_checkpoint(self, run_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Get the latest checkpoint from a run.

        Args:
            run_dir: Run directory (defaults to current run)

        Returns:
            Path to latest checkpoint or None
        """
        run_dir = run_dir or self.run_dir
        if run_dir is None:
            return None

        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            return None

        # Check for 'latest' symlink first
        latest_link = ckpt_dir / "latest"
        if latest_link.exists():
            return latest_link.resolve()

        # Otherwise find highest step
        steps = list(ckpt_dir.glob("step_*"))
        if not steps:
            return None

        steps = sorted(steps, key=lambda p: int(p.name.split("_")[1]))
        return steps[-1]

    def should_save(self, step: int) -> bool:
        """Check if should save at this step."""
        return step > 0 and step % self.save_every_n_steps == 0

    def save_checkpoint(
        self,
        step: int,
        model,
        optimizer=None,
        scheduler=None,
        metrics: Optional[Dict] = None,
        extra_state: Optional[Dict] = None,
    ) -> Path:
        """
        Save a checkpoint with all necessary state.

        Args:
            step: Current training step
            model: Model to save (should be PEFT model)
            optimizer: Optional optimizer
            scheduler: Optional LR scheduler
            metrics: Optional metrics dict
            extra_state: Optional extra state to save

        Returns:
            Path to saved checkpoint
        """
        if self.run_dir is None:
            raise RuntimeError("No run directory. Call create_new_run() first.")

        ckpt_dir = self.run_dir / "checkpoints" / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model (PEFT adapter)
        model.save_pretrained(ckpt_dir)

        # Save trainer state
        trainer_state = {
            "global_step": step,
            "metrics": metrics or {},
            "best_metric": self.best_metric,
            "timestamp": datetime.now().isoformat(),
        }
        if extra_state:
            trainer_state.update(extra_state)

        with open(ckpt_dir / "trainer_state.json", "w") as f:
            json.dump(trainer_state, f, indent=2)

        # Save optimizer and scheduler
        if optimizer is not None:
            opt_state = {"optimizer": optimizer.state_dict()}
            if scheduler is not None:
                opt_state["scheduler"] = scheduler.state_dict()
            torch.save(opt_state, ckpt_dir / "optimizer.pt")

        # Save RNG states for reproducibility
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_states, ckpt_dir / "rng_states.pt")

        # Update 'latest' symlink
        self._update_latest_link(ckpt_dir)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        # Check if this is the best
        if metrics:
            metric_value = metrics.get("accuracy", metrics.get("reward", 0))
            if metric_value > self.best_metric:
                self.best_metric = metric_value
                self._save_best(ckpt_dir)

        return ckpt_dir

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model,
        optimizer=None,
        scheduler=None,
        load_rng: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore state.

        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load into
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
            load_rng: Whether to restore RNG states

        Returns:
            Trainer state dictionary
        """
        checkpoint_path = Path(checkpoint_path)

        # Load adapter weights
        from peft import PeftModel, set_peft_model_state_dict
        adapter_path = checkpoint_path

        if hasattr(model, "load_adapter"):
            model.load_adapter(adapter_path, "default")
        else:
            # Load state dict manually
            adapter_state = torch.load(
                adapter_path / "adapter_model.safetensors",
                map_location="cpu"
            )
            set_peft_model_state_dict(model, adapter_state)

        # Load trainer state
        trainer_state = {}
        state_file = checkpoint_path / "trainer_state.json"
        if state_file.exists():
            with open(state_file) as f:
                trainer_state = json.load(f)
            self.best_metric = trainer_state.get("best_metric", float("-inf"))

        # Load optimizer
        opt_file = checkpoint_path / "optimizer.pt"
        if optimizer is not None and opt_file.exists():
            opt_state = torch.load(opt_file, map_location="cpu")
            optimizer.load_state_dict(opt_state["optimizer"])
            if scheduler is not None and "scheduler" in opt_state:
                scheduler.load_state_dict(opt_state["scheduler"])

        # Restore RNG states
        rng_file = checkpoint_path / "rng_states.pt"
        if load_rng and rng_file.exists():
            rng_states = torch.load(rng_file)
            random.setstate(rng_states["python"])
            np.random.set_state(rng_states["numpy"])
            torch.set_rng_state(rng_states["torch"])
            if "cuda" in rng_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_states["cuda"])

        return trainer_state

    def _update_latest_link(self, ckpt_dir: Path):
        """Update the 'latest' symlink."""
        latest_link = self.run_dir / "checkpoints" / "latest"

        # Remove existing link
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create relative symlink
        try:
            latest_link.symlink_to(ckpt_dir.name)
        except OSError:
            # Windows might not support symlinks, just skip
            pass

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping last N."""
        if self.run_dir is None:
            return

        ckpt_dir = self.run_dir / "checkpoints"
        steps = [p for p in ckpt_dir.glob("step_*") if p.is_dir()]
        steps = sorted(steps, key=lambda p: int(p.name.split("_")[1]))

        # Keep last N
        if len(steps) > self.keep_last_n:
            for old_ckpt in steps[:-self.keep_last_n]:
                shutil.rmtree(old_ckpt)

    def _save_best(self, ckpt_dir: Path):
        """Save best checkpoint."""
        if not self.keep_best:
            return

        best_dir = self.get_experiment_dir() / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(ckpt_dir, best_dir)

    def list_runs(self) -> List[Dict]:
        """List all runs for this experiment."""
        exp_dir = self.get_experiment_dir()
        if not exp_dir.exists():
            return []

        runs = []
        for run_dir in sorted(exp_dir.glob("run_*")):
            info = {"run_id": run_dir.name, "path": str(run_dir)}

            # Load run info if available
            info_file = run_dir / "run_info.json"
            if info_file.exists():
                with open(info_file) as f:
                    info.update(json.load(f))

            # Get latest checkpoint step
            latest = self.get_latest_checkpoint(run_dir)
            if latest:
                info["latest_step"] = int(latest.name.split("_")[1])
            else:
                info["latest_step"] = 0

            runs.append(info)

        return runs

    def mark_run_complete(self):
        """Mark current run as complete."""
        if self.run_dir is None:
            return

        info_file = self.run_dir / "run_info.json"
        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)
            info["status"] = "complete"
            info["completed_at"] = datetime.now().isoformat()
            with open(info_file, "w") as f:
                json.dump(info, f, indent=2)

    def set_run_dir(self, run_dir: Path):
        """Set run directory for resuming."""
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name


def find_checkpoint(path: str) -> Optional[Path]:
    """
    Find checkpoint from various path formats.

    Accepts:
    - Direct checkpoint path: outputs/exp/run_xxx/checkpoints/step_100
    - Run path: outputs/exp/run_xxx (returns latest checkpoint)
    - Experiment path: outputs/exp (returns latest run's latest checkpoint)

    Args:
        path: Path string

    Returns:
        Path to checkpoint directory or None
    """
    path = Path(path)

    if not path.exists():
        return None

    # Direct checkpoint path
    if (path / "adapter_config.json").exists():
        return path

    # Run path
    if (path / "checkpoints").exists():
        latest = path / "checkpoints" / "latest"
        if latest.exists():
            return latest.resolve()
        steps = list((path / "checkpoints").glob("step_*"))
        if steps:
            return sorted(steps, key=lambda p: int(p.name.split("_")[1]))[-1]

    # Experiment path (find latest run)
    runs = list(path.glob("run_*"))
    if runs:
        latest_run = sorted(runs)[-1]
        return find_checkpoint(str(latest_run))

    return None
