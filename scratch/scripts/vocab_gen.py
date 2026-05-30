import os
import sys
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf
from src.utils import hydra_runner

_CONF_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "vocab", "train")


def _load_class(target: str):
    """Load a class from a dotted path string, e.g. 'src.vocab.bpe.BPETokenizer'."""
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@hydra_runner(config_path=_CONF_DIR, config_name="bpe")
def main(cfg):
    OmegaConf.resolve(cfg)
    vcfg = cfg.vocab_builder

    TokenizerClass = _load_class(vcfg._target_)

    # --- Train ---
    print(f"Training  {vcfg._target_}  vocab_size={vcfg.vocab_size}  file={vcfg.file_path}")
    tokenizer = TokenizerClass.train_vocab(vocab_size=vcfg.vocab_size, file_path=vcfg.file_path)

    print(f"\nLearned {len(tokenizer.merge_rules)} merge rules:")
    for (a, b), idx in tokenizer.merge_rules.items():
        print(f"  [{idx}]  {repr(a)} + {repr(b)}  ->  {repr(a + b)}")

    # --- Encode dry run ---
    print("\nEncode dry run:")
    print(f"  {'input':<14}  {'token ids':<34}  decoded")
    print(f"  {'-'*14}  {'-'*34}  {'-'*24}")
    for s in cfg.samples:
        ids     = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
        print(f"  {repr(s):<14}  {str(ids):<34}  {repr(decoded)}")

    # --- Save ---
    print(f"\nSaving vocab -> {cfg.vocab_dir}")
    tokenizer.save_vocab(cfg.vocab_dir)

    # --- Reload and verify ---
    print("Reloading and re-encoding first two samples:")
    tokenizer2 = TokenizerClass.load_vocab(cfg.vocab_dir)
    for s in list(cfg.samples)[:2]:
        ids     = tokenizer2.encode(s)
        decoded = tokenizer2.decode(ids)
        print(f"  {repr(s):<14}  {str(ids):<34}  {repr(decoded)}")


if __name__ == "__main__":
    main()
