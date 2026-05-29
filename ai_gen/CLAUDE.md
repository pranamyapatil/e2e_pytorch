# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`e2e_pytorch` is a PyTorch-based end-to-end deep learning framework designed to make training any deep neural network model easy and composable — similar in spirit to HuggingFace Transformers. The goal is to provide clean abstractions for models, configs, trainers, and data pipelines that are fully configurable via dataclasses or config files.

## Architecture Philosophy

Every major component should follow this pattern:
- A **Config dataclass** (or `BaseConfig` subclass) that holds all hyperparameters and construction arguments.
- A **Base class** defining the interface (abstract methods, shared logic).
- **Concrete subclasses** registered so they can be instantiated by name from a config.

This mirrors HuggingFace's `PretrainedConfig` → `PreTrainedModel` → specific model pattern. New model types, trainers, optimizers, schedulers, and data collators should all be addable without modifying existing code.

## Key Abstractions (Planned/In Progress)

| Abstraction | Responsibility |
|---|---|
| `BaseConfig` | Serializable dataclass; holds all constructor args; supports `from_dict`, `to_dict`, `from_json` |
| `BaseModel` | Wraps `nn.Module`; tied to a config; exposes `forward`, `save_pretrained`, `from_pretrained` |
| `BaseTrainer` | Training loop, gradient accumulation, logging, checkpointing; configurable via `TrainingArguments` |
| `BaseDataset` / `BaseDataCollator` | Data loading and batching contracts |
| `Registry` | Maps string names → classes; used by all base classes so components can be built from configs |

## Design Rules

- **Config-first**: every class that has tuneable parameters must accept a config object, not raw kwargs scattered across `__init__`.
- **Registry pattern**: concrete implementations register themselves (decorator or `__init_subclass__`) so the framework can instantiate them from a string name in a config.
- **No magic defaults hidden in code**: defaults live in the config dataclass, not in method signatures.
- **Separation of concerns**: model definition, training loop, data pipeline, and evaluation are independent and can be swapped without touching each other.

## Development Setup

```bash
pip install -e ".[dev]"
```

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_trainer.py

# Run a single test by name
pytest tests/test_trainer.py::test_gradient_accumulation

# Lint
ruff check .

# Type check
mypy src/
```

*(Update these once `pyproject.toml` / `setup.py` is added.)*
