import functools
import sys
import argparse
from pathlib import Path
from typing import Any, Callable, Optional

from omegaconf import OmegaConf, DictConfig


def hydra_runner(
    config_path: Optional[str] = ".",
    config_name: Optional[str] = None,
) -> Callable:
    """
    Decorator that loads a YAML config and passes it to the wrapped function.
    CLI overrides use dot notation: key=value or key.subkey=value.
    """

    def decorator(task_function: Callable) -> Callable:
        @functools.wraps(task_function)
        def wrapper(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            # If a config is passed directly (e.g. in tests), skip file loading.
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)

            # --- 1. Load base config from YAML ---
            config_file = Path(config_path) / f"{config_name}.yaml"
            cfg = OmegaConf.load(config_file)

            # --- 2. Parse CLI overrides (key=value pairs) ---
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("overrides", nargs="*")
            args, _ = parser.parse_known_args()

            # Merge overrides: "model.lr=1e-4" → {"model": {"lr": 1e-4}}
            for override in args.overrides:
                key, _, value = override.partition("=")
                override_cfg = OmegaConf.from_dotlist([f"{key}={value}"])
                cfg = OmegaConf.merge(cfg, override_cfg)

            return task_function(cfg)

        return wrapper

    return decorator


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    @hydra_runner(config_path="conf", config_name="train_config")
    def main(cfg: DictConfig):
        print(OmegaConf.to_yaml(cfg))

    main()
