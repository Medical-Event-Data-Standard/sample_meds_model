#!/usr/bin/env python

"""Run a simple binary classification model over an ESDS dataset."""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import datasets
import lightning as L
import numpy as np
import polars as pl
import torch
from ESDS_transformations import (
    JoinCohortFntr,
    NormalizeFntr,
    SampleSubsequencesFntr,
    TensorizeFntr,
    TokenizeFntr,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sample_ESDS_model.simple_model import LightningModel

logger = logging.getLogger(__name__)


def get_vocab(ds: datasets.Dataset) -> list[str]:
    """Get the size of the vocabulary.

    TODO(mmd): Leverage the metadata file to get this.
    """
    vocab = set()
    for row in tqdm(ds, desc="Building vocabulary", total=len(ds)):
        vocab.update([m["code"] for m in row["static_measurements"]])
        for e in row["events"]:
            vocab.update([m["code"] for m in e["measurements"]])
    return sorted(list(vocab))


def get_norm_params(ds: datasets.Dataset) -> dict[str, tuple[float, float]]:
    """Get the means and standard deviations of observed code values.

    TODO(mmd): Leverage the metadata file to get this.
    """

    # stats will map code -> (count, sum, sum_sq)
    stats = {}

    def add_measurements(measurements):
        for m in measurements:
            code = m["code"]
            value = m["numeric_value"]
            if value is None or value is np.nan:
                continue
            if code not in stats:
                stats[code] = (0, 0, 0)
            count, sum_vals, sum_sq_vals = stats[code]
            stats[code] = (count + 1, sum_vals + value, sum_sq_vals + value**2)

    for row in tqdm(ds, desc="Measuring normalization parameters", total=len(ds)):
        add_measurements(row["static_measurements"])
        for e in row["events"]:
            add_measurements(e["measurements"])
    return {
        code: (sum_vals / count, np.sqrt(sum_sq_vals / count - (sum_vals / count) ** 2))
        for code, (count, sum_vals, sum_sq_vals) in stats.items()
    }

def get_max_measurements(ds: datasets.Dataset) -> int:
    """Get the max number of measurements observed across all patients in the dataset.

    TODO(mmd): Leverage the metadata file to get this.
    """

    max_measurements = 0

    for row in tqdm(ds, desc="Computing the max # of measurements", total=len(ds)):
        max_measurements = max(max_measurements, len(row["static_measurements"]))
        for e in row["events"]:
            max_measurements = max(max_measurements, len(e["measurements"]))
    return max_measurements



def main():
    parser = ArgumentParser("Train a model over a binary classification task on an ESDS dataset.")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--task_df_path", type=str)
    parser.add_argument("--output_path", type=str)

    # Model params
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=64)

    # Training params
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    args = parser.parse_args()

    logger.info("Loading the core dataset...")
    ds = datasets.load_dataset(
        "parquet",
        data_files={
            sp: str(Path(args.dataset_path) / sp / "*.parquet") for sp in ("train", "tuning", "held_out")
        },
    )

    logger.info("Loading the task dataframe...")
    task_df = pl.read_parquet(args.task_df_path)

    vocab_file = Path(args.output_path) / "vocab.json"
    if vocab_file.exists():
        logger.info("Loading vocab from disk...")
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
    else:
        logger.info("Getting vocab...")
        vocab = get_vocab(ds["train"])
        with open(vocab_file, "w") as f:
            json.dump(vocab, f)

    idxmap = {code: i for i, code in enumerate(vocab)}
    vocab_size = len(vocab)

    norm_file = Path(args.output_path) / "norm_params.json"
    if norm_file.exists():
        logger.info("Loading normalization params from disk...")
        with open(norm_file, "r") as f:
            norm_params = json.load(f)
    else:
        logger.info("Getting normalization params...")
        norm_params = get_norm_params(ds["train"])
        with open(norm_file, mode="w") as f:
            json.dump(norm_params, f)

    max_measurements_file = Path(args.output_path) / "max_measurements.json"
    if max_measurements_file.exists():
        logger.info("Loading max # of measurements from disk...")
        with open(max_measurements_file, "r") as f:
            max_measurements = json.load(f)["max_measurements"]
    else:
        logger.info("Getting the max # of measurements...")
        max_measurements = get_max_measurements(ds["train"])
        with open(max_measurements_file, "w") as f:
            json.dump({"max_measurements": max_measurements}, f)

    logger.info("Applying transformations...")
    transforms = [
        JoinCohortFntr(task_df),
        SampleSubsequencesFntr(
            max_seq_len=args.max_seq_len, n_samples_per_patient=1, sample_strategy="to_end"
        ),
        TokenizeFntr(vocab),
        NormalizeFntr(norm_params),
    ]

    for transform_fn in transforms:
        ds = ds.map(transform_fn, batch_size=256, batched=True)

    ds = ds.map(
        TensorizeFntr(idxmap, pad_sequences_to=args.max_seq_len, pad_measurements_to=max_measurements),
        batch_size=256,
        batched=True,
        remove_columns=["patient_id", "static_measurements", "events"]
    )

    def print_summ(v):
        if type(v) is torch.Tensor:
            print(f"Tensor ({v.dtype}): {v.shape}")
        elif type(v) is list:
            print(f"List[{print_summ(v[0])}]: {len(v)}")
        else:
            print(f"{type(v)}")


    for i in range(5):
        x = ds['train'][i]
        print(f"For sample {i}, we have:")
        for k, v in x.items():
            print(f"  {k}: {print_summ(v)}")

    train_dataloader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True)
    tuning_dataloader = DataLoader(ds["tuning"], batch_size=args.batch_size, shuffle=False)
    held_out_dataloader = DataLoader(ds["held_out"], batch_size=args.batch_size, shuffle=False)

    logger.info("Building the model...")
    lit_model = LightningModel(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        lr=args.lr,
    )

    logger.info("Building the optimizer...")

    logger.info("Training the model...")
    trainer = L.Trainer(max_epochs=args.n_epochs, default_root_dir=args.output_path)
    trainer.fit(lit_model, train_dataloader, tuning_dataloader)

    logger.info("Evaluating the model...")
    tuning_results = trainer.test(lit_model, tuning_dataloader)
    held_out_results = trainer.test(lit_model, held_out_dataloader)

    with open(Path(args.output_path) / "tuning_results.json", "w") as f:
        json.dump(tuning_results, f)

    with open(Path(args.output_path) / "held_out_results.json", "w") as f:
        json.dump(held_out_results, f)


if __name__ == "__main__":
    main()
