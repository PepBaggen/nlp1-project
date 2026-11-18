"""
Run project 2 experiments on Modal GPUs, one container per configuration.

Each configuration is trained with 3 seeds x 5 folds (the same folds as project 1 and notebooks/project2.ipynb).
Results land in results/project2/modal/<config name>.json (accuracies per seed and fold, best epochs, curves).

Usage (from the repo root):
    modal run project2/modal_run.py                      # all configs in CONFIGS
    modal run project2/modal_run.py --only lstm          # only configs whose name contains "lstm"
    modal run project2/modal_run.py --only datasize --dry  # list what would run
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "project2" / "modal"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .run_commands("python -m nltk.downloader -d /usr/share/nltk_data punkt_tab stopwords")
    .add_local_dir(ROOT / "part1", "/root/part1", ignore=["*.ipynb", "__pycache__"])
    .add_local_dir(ROOT / "part2", "/root/part2", ignore=["*.ipynb", "__pycache__"])
    .add_local_dir(ROOT / "project1", "/root/project1", ignore=["*.pptx", "__pycache__"])
    .add_local_dir(ROOT / "project2", "/root/project2", ignore=["*.zip", "*.npz", "*.vec", "__pycache__"])
)
cache = modal.Volume.from_name("nlp1-embeddings", create_if_missing=True)
app = modal.App("nlp1-project2", image=image)

# ---------------------------------------------------------------------------------------------
# Configurations. Baselines are the best settings in project2/documentation.md; every other config
# changes ONE thing relative to its baseline, so differences can be attributed to that change.
# ---------------------------------------------------------------------------------------------
POOL = dict(enc_type="embed-and-pool", tokenizer="bow", emb="glove50",
            enc_kwargs={"freeze_embeddings": False},
            regressor_kwargs={"hidden_size1": 200, "hidden_size2": 200, "dropout": 0.5},
            opt_kwargs={"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "num_epochs": 35},
            train_fraction=1.0)
LSTM = dict(enc_type="lstm", tokenizer="rnn", emb="glove50",
            enc_kwargs={"freeze_embeddings": True, "hidden_size": 100, "word_dropout": 0.2},
            regressor_kwargs={"hidden_size1": 200, "hidden_size2": 200, "dropout": 0.5},
            opt_kwargs={"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "num_epochs": 50},
            train_fraction=1.0)


def variant(base, **changes):
    """Copy a baseline and overwrite fields; nested dicts (enc_kwargs, ...) are merged, not replaced."""
    cfg = json.loads(json.dumps(base))
    for key, value in changes.items():
        if isinstance(value, dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


CONFIGS = {
    "pool-base": POOL,
    "lstm-base": LSTM,
}
# Data size: does pretraining help more when there is little data? (asked in the project, point 4)
for frac in [0.1, 0.25, 0.5]:
    for frozen in [True, False]:
        tag = "frozen" if frozen else "trainable"
        CONFIGS[f"datasize-pool-{tag}-{frac}"] = variant(POOL, train_fraction=frac, enc_kwargs={"freeze_embeddings": frozen})
        CONFIGS[f"datasize-lstm-{tag}-{frac}"] = variant(LSTM, train_fraction=frac, enc_kwargs={"freeze_embeddings": frozen})
CONFIGS["datasize-pool-frozen-1.0"] = variant(POOL, enc_kwargs={"freeze_embeddings": True})
CONFIGS["datasize-lstm-trainable-1.0"] = variant(LSTM, enc_kwargs={"freeze_embeddings": False})
# Bigger embedding models: is 50d GloVe the bottleneck?
for emb in ["glove300", "fasttext300"]:
    CONFIGS[f"emb-pool-{emb}"] = variant(POOL, emb=emb)
    CONFIGS[f"emb-lstm-{emb}"] = variant(LSTM, emb=emb)
# LSTM convergence: best epochs were late (35-50), so train longer / faster / bigger.
CONFIGS["lstm-epochs100"] = variant(LSTM, opt_kwargs={"num_epochs": 100})
CONFIGS["lstm-lr3e-3"] = variant(LSTM, opt_kwargs={"learning_rate": 3e-3})
CONFIGS["lstm-hidden200"] = variant(LSTM, enc_kwargs={"hidden_size": 200})
# Pooling: mean vs sum
CONFIGS["pool-sum"] = variant(POOL, enc_kwargs={"average_pool": False})

# Round 2: fastText 300d was the clear winner in round 1, so vary around it.
FT_POOL, FT_LSTM = variant(POOL, emb="fasttext300"), variant(LSTM, emb="fasttext300")
CONFIGS["r2-ft-pool-frozen"] = variant(FT_POOL, enc_kwargs={"freeze_embeddings": True})
CONFIGS["r2-ft-pool-lr3e-4"] = variant(FT_POOL, opt_kwargs={"learning_rate": 3e-4})   # trainable overfit by epoch 3
CONFIGS["r2-ft-lstm-trainable"] = variant(FT_LSTM, enc_kwargs={"freeze_embeddings": False})
CONFIGS["r2-ft-lstm-hidden200"] = variant(FT_LSTM, enc_kwargs={"hidden_size": 200})
for frac in [0.1, 0.25, 0.5]:
    CONFIGS[f"r2-ft-datasize-pool-{frac}"] = variant(FT_POOL, train_fraction=frac)
    CONFIGS[f"r2-ft-datasize-lstm-{frac}"] = variant(FT_LSTM, train_fraction=frac)


@app.function(gpu="T4", timeout=3 * 60 * 60, volumes={"/cache": cache})
def run_config(name, cfg, seeds=(0, 1, 2), num_folds=5):
    import random
    import re

    import numpy as np
    import torch

    import part1
    import part2
    from project1.data import load_data
    from project2 import embeddings

    embeddings.EMB_DIR = Path("/cache")   # downloads and the small .npz tables live on the volume

    dataset, outcome_space = load_data(test_size=0.2, seed=0)
    pairs = list(zip(dataset["train"]["text"], dataset["train"]["label"]))

    # same vocabulary filter as the notebook: every token of train + test under both tokenizers, plus hyphen parts
    all_texts = list(dataset["train"]["text"]) + list(dataset["test"]["text"])
    keep = set()
    for tok in [part2.clslib.BasicTokenizer(True, True, True), part2.clslib.BasicTokenizer(True, False, False)]:
        for text in all_texts:
            keep.update(tok.tokens(text))
    keep |= {p for w in list(keep) for p in w.split("-")}
    emb = embeddings.load(cfg["emb"], keep)
    cache.commit()

    class HyphenSplitTokenizer(part2.clslib.BasicTokenizer):
        def tokens(self, text):
            return super().tokens(re.sub(r"(?<=\w)-(?=\w)", " ", text))

    if cfg["tokenizer"] == "bow":
        tokenizer = part2.clslib.BasicTokenizer(lowercase=True, remove_punct=True, remove_stopwords=True,
                                               language="english", vocab=emb.vocab)
    else:
        tokenizer = HyphenSplitTokenizer(lowercase=True, remove_punct=False, remove_stopwords=False,
                                         language="english", vocab=emb.vocab)

    result = {"name": name, "config": cfg, "accuracies": [], "best_epochs": [], "curves": []}
    for seed in seeds:
        folds = part1.experiment.cross_validation_split(pairs=pairs, k=num_folds, seed=seed)
        accs, best_epochs, curves = [], [], []
        for fold, split in folds.items():
            training = split["training"]
            if cfg["train_fraction"] < 1.0:   # subsample the training part only; validation stays the full fold
                training = random.Random(seed * 100 + fold).sample(list(training), int(cfg["train_fraction"] * len(training)))
            torch.manual_seed(seed * 100 + fold)
            np.random.seed(seed * 100 + fold)
            metrics, best = part2.experiment.train(
                training, split["validation"], tokenizer, emb,
                num_classes=len(outcome_space), save_to=f"/tmp/{name}-{seed}-{fold}",
                enc_type=cfg["enc_type"], enc_kwargs=dict(cfg["enc_kwargs"]),
                regressor_kwargs=dict(cfg["regressor_kwargs"]), opt_kwargs=dict(cfg["opt_kwargs"]),
                device="cuda",
            )
            accs.append(metrics[best]["val_metrics"]["accuracy"])
            best_epochs.append(best + 1)
            curves.append([{"train_loss": m["train_metrics"]["loss"], "train_acc": m["train_metrics"]["accuracy"],
                            "val_loss": m["val_metrics"]["loss"], "val_acc": m["val_metrics"]["accuracy"]}
                           for m in metrics])
        result["accuracies"].append(accs)
        result["best_epochs"].append(best_epochs)
        result["curves"].append(curves)
    acc = np.array(result["accuracies"])
    result["mean"], result["std"] = float(acc.mean()), float(acc.std())
    return result


@app.local_entrypoint()
def main(only: str = "", dry: bool = False):
    todo = {n: c for n, c in CONFIGS.items() if only in n}
    print(f"{len(todo)} configs:", ", ".join(todo))
    if dry:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    progress = OUT_DIR / "progress.txt"
    summary = OUT_DIR / "results.txt"

    def log(line):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {line}"
        print(line)
        with open(progress, "a") as f:
            f.write(line + "\n")

    log(f"started {len(todo)} configs: {', '.join(todo)}")
    done = 0
    for result in run_config.starmap(todo.items(), return_exceptions=True, order_outputs=False):
        done += 1
        if isinstance(result, Exception):
            log(f"[{done}/{len(todo)}] FAILED: {result!r}")
            continue
        name = result["name"]
        (OUT_DIR / f"{name}.json").write_text(json.dumps(result))
        epochs = sorted(e for s in result["best_epochs"] for e in s)
        line = f"{name:35s} mean={result['mean']:.4f} std={result['std']:.4f} best epochs={epochs}"
        log(f"[{done}/{len(todo)}] {line}")
        with open(summary, "a") as f:
            f.write(line + "\n")
        # per-config training curves: one line per epoch, averaged over seeds and folds
        curves = [c for seed in result["curves"] for c in seed]
        with open(OUT_DIR / f"{name}.txt", "w") as f:
            f.write(f"{name}\nconfig: {json.dumps(result['config'])}\n")
            f.write(f"accuracies (seed x fold): {result['accuracies']}\n")
            f.write(f"mean={result['mean']:.4f} std={result['std']:.4f}\n\n")
            f.write("epoch  train_loss  train_acc  val_loss  val_acc   (mean over seeds x folds)\n")
            for e in range(len(curves[0])):
                avg = {k: sum(c[e][k] for c in curves) / len(curves) for k in curves[0][e]}
                f.write(f"{e + 1:5d}  {avg['train_loss']:10.4f}  {avg['train_acc']:9.4f}  {avg['val_loss']:8.4f}  {avg['val_acc']:7.4f}\n")
    log("all done")
