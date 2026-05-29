# e2e_pytorch

A pure PyTorch end-to-end deep learning framework for training models across domains (speech, text, and more). Built for learners who want to understand complex architectures from first principles — no framework magic hiding the details.

---

## Installation

```bash
pip install -e ".[dev]"
```

---

## Key Wiring Patterns

### 1. Config — every component starts with a typed dataclass

All hyperparameters live in a `BaseConfig` subclass, never scattered in `__init__` signatures.

```python
from dataclasses import dataclass
from common.config import BaseConfig

@dataclass
class TransformerConfig(BaseConfig):
    vocab_size: int = 32000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
```

Configs are serializable out of the box:

```python
cfg = TransformerConfig(d_model=256)
cfg.to_json("configs/text/transformer_small.json")

cfg2 = TransformerConfig.from_json("configs/text/transformer_small.json")
```

---

### 2. Registry — build any component by name from a config

Register a class once with a decorator. After that it can be instantiated by string name — no `if/elif` chains needed when adding new models or datasets.

```python
from common.registry import Registry
from common.model import BaseModel

@Registry.register("model", "transformer")
class TransformerModel(BaseModel):
    config_class = TransformerConfig   # links the config class to this model

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        # build layers using config fields ...

    def forward(self, batch: dict) -> dict:
        # batch is a plain dict (input_ids, attention_mask, labels, ...)
        # must return a dict with at least {"loss": tensor}
        ...
        return {"loss": loss, "logits": logits}
```

Look up and instantiate from a string:

```python
model_cls = Registry.get("model", "transformer")
model = model_cls(config)

# or in one step:
model = Registry.build("model", "transformer", config)
```

---

### 3. Dataset — same registration pattern

```python
from torch.utils.data import Dataset

@Registry.register("dataset", "text_lm")
class TextLMDataset(Dataset):
    def __init__(self, split: str, data_path: str, max_length: int = 512):
        ...

    def __len__(self): ...
    def __getitem__(self, idx) -> dict: ...  # returns a dict batch
```

---

### 4. Experiment config — one JSON ties everything together

```json
{
    "model_type":    "transformer",
    "dataset_type":  "text_lm",
    "do_eval":       true,
    "model_config": {
        "vocab_size": 32000,
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6
    },
    "dataset_config": {
        "data_path": "data/wikitext",
        "max_length": 512
    },
    "training_args": {
        "output_dir":    "outputs/transformer-base",
        "num_epochs":    20,
        "batch_size":    64,
        "learning_rate": 3e-4,
        "scheduler":     "cosine",
        "warmup_steps":  1000,
        "fp16":          true
    }
}
```

---

### 5. Training and evaluation

```bash
# Train from scratch
python train.py --config configs/text/transformer_base.json

# Resume from a checkpoint
python train.py --config configs/text/transformer_base.json --resume outputs/transformer-base/checkpoint-epoch-0005

# Evaluate a checkpoint on the test split
python evaluate.py --checkpoint outputs/transformer-base/checkpoint-epoch-0020
```

`train.py` automatically saves the experiment config to `output_dir/experiment_config.json`, so `evaluate.py` can find it without extra flags.

---

### 6. Custom eval metrics — override `compute_metrics()`

`BaseTrainer.evaluate()` always returns `eval_loss`. Add domain-specific metrics by subclassing:

```python
from trainer import BaseTrainer

class AudioTrainer(BaseTrainer):
    def compute_metrics(self, outputs: list[dict]) -> dict[str, float]:
        hypotheses = [o["hypothesis"] for o in outputs]
        references = [o["reference"] for o in outputs]
        return {"wer": compute_wer(hypotheses, references)}
```

Then pass `AudioTrainer` instead of `BaseTrainer` in your training script.

---

## Project Structure

```
e2e_pytorch/
├── common/          # BaseConfig, BaseModel, Registry, shared layers
├── audio/           # AudioDataset, feature extraction, Conformer, WER metric
├── text/            # TextDataset, tokenizer, Transformer/BERT, BLEU metric
├── trainer/         # BaseTrainer, TrainingArguments, optimizer/scheduler factories
├── train.py         # CLI: config → model + data → train
├── evaluate.py      # CLI: checkpoint → eval metrics
└── configs/         # Experiment JSON configs per domain/model
```

---

## Design Principles

- **Config-first** — every tuneable parameter lives in a `BaseConfig` subclass.
- **Registry pattern** — new models, datasets, and trainers plug in without modifying existing code.
- **Pure PyTorch** — the full forward pass, training loop, and backward pass are readable with no hidden abstractions.
- **Separation of concerns** — model, data pipeline, trainer, and evaluation are independent and swappable.
