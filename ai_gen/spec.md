# spec.md — e2e_pytorch Project Specification

This is a pure PyTorch end-to-end implementation of deep learning projects across domains (speech, text, etc.). All implementation is in pure PyTorch with a modular, reusable class structure.

## Goal

Build a clean, educational deep learning framework where:
- Each domain (audio, text, ...) has its own self-contained directory with data loading, processing, datasets, model classes, and evaluation metrics.
- A universal set of training, evaluation, and inference scripts works across all domains via a shared config and registry system.
- Learners can trace the full forward pass of complex architectures without framework magic hiding the details.

## Target Users

Those who want to learn and understand complex architectures (Transformers, Conformers, etc.) from first principles — without relying on high-level abstractions that hide what's happening.

## Directory Structure

```
e2e_pytorch/
├── common/                      # Shared building blocks
│   ├── layers/                  # Reusable nn.Module primitives
│   │   ├── attention.py         # MultiHeadAttention, SelfAttention, CrossAttention
│   │   ├── feedforward.py       # FFN, GLU variants
│   │   ├── normalization.py     # LayerNorm, RMSNorm, BatchNorm
│   │   ├── positional.py        # Sinusoidal, RoPE, ALiBi embeddings
│   │   └── dropout.py           # DropPath, StochasticDepth
│   ├── config.py                # BaseConfig dataclass
│   ├── model.py                 # BaseModel (wraps nn.Module)
│   └── registry.py              # Registry: maps name → class
│
├── audio/                       # Speech / audio domain
│   ├── data/
│   │   ├── dataset.py           # AudioDataset
│   │   ├── collator.py          # AudioCollator
│   │   └── processor.py         # Feature extraction (MFCC, mel-spectrogram)
│   ├── models/
│   │   ├── conformer.py
│   │   └── wav2vec.py
│   └── metrics.py               # WER, CER
│
├── text/                        # NLP domain
│   ├── data/
│   │   ├── dataset.py           # TextDataset
│   │   ├── collator.py
│   │   └── tokenizer.py
│   ├── models/
│   │   ├── transformer.py
│   │   └── bert.py
│   └── metrics.py               # BLEU, ROUGE, accuracy
│
├── trainer/                     # Universal training infrastructure
│   ├── trainer.py               # BaseTrainer: loop, grad accum, checkpointing
│   ├── optimizer.py             # Optimizer factory
│   └── scheduler.py             # LR scheduler factory
│
├── train.py                     # Entry point: config → model + data → train
├── evaluate.py                  # Entry point: checkpoint → eval
├── configs/                     # YAML/JSON configs per experiment
└── tests/
```

## Core Features

- **`common/layers/`** — shared primitive pool (Attention, FFN, Norm, Positional encodings). Every domain model imports from here; no duplication across domains.
- **`BaseConfig`** — serializable dataclass (supports `from_dict`, `to_dict`, `from_json`). All hyperparameters live here, not scattered in `__init__` signatures.
- **`BaseModel`** — thin `nn.Module` wrapper tied to a config; exposes `save_checkpoint` / `load_checkpoint`.
- **`Registry`** — maps string names to classes so any component can be instantiated from a config file (`Registry.build("model", "conformer", config)`).
- **`BaseTrainer`** — domain-agnostic training loop with gradient accumulation, logging, and checkpointing.
- **Domain dirs** (`audio/`, `text/`) are self-contained and depend only on `common/`.
- **`train.py` / `evaluate.py`** are domain-agnostic entry points — they resolve the right classes via the registry and a config file.

## Key Design Decisions

- **Config-first**: every tuneable component accepts a config object, not raw kwargs.
- **Registry pattern**: concrete subclasses register themselves so the framework can build them by name from a config.
- **No hidden defaults**: all defaults live in the config dataclass, never in method signatures.
- **Pure PyTorch**: no Lightning, no HuggingFace Trainer — the full training loop is visible and readable.
- **Separation of concerns**: model, trainer, data pipeline, and evaluation are independent and swappable.
- Python concepts in use: dataclasses, abstract base classes, decorators (for registry), `__init_subclass__`, type hints throughout.

## Out of Scope

- Multi-modal models (cross-domain fusion)
- Distributed / multi-GPU training (can be added later)
- Deployment / serving infrastructure

## Milestones / Phases

1. `common/` — BaseConfig, BaseModel, Registry, all shared layers
2. `trainer/` — BaseTrainer, optimizer/scheduler factories
3. `text/` domain — Transformer, dataset, metrics
4. `audio/` domain — Conformer, feature extraction, WER metric
5. End-to-end `train.py` / `evaluate.py` wiring
6. Example configs and a working training run per domain

## Open Questions

- YAML vs Python dataclass configs as the primary interface?
- Should `configs/` live inside each domain dir or stay top-level?
- Which audio datasets to support first (LibriSpeech, LJSpeech)?
- Which text tasks to support first (language modeling, classification, seq2seq)?
