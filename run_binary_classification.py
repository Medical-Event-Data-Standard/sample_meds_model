#!/usr/bin/env python

"""Run a simple binary classification model over an ESDS dataset."""

from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm
import datasets
import logging

from esds_transformations import JoinCohort

from .simple_model import Model

logger = logging.getLogger(__name__)

def get_vocab_size(ds: datasets.Dataset) -> int:
    """Get the size of the vocabulary."""
    vocab = set()
    for row in tqdm(ds, desc="Building vocabulary", total=len(ds)):
        vocab.update([m['code'] for m in row['static_measurements']])
        for e in row['events']:
            vocab.update([m['code'] for m in e['measurements']])
    return len(vocab)

def main():
    parser = ArgumentParser("Train a model over a binary classification task on an ESDS dataset.")
    parser.add_argument("--dataset_path", type=str, default=dataset_path)
    parser.add_argument("--task_df_path", type=str, default=task_df_path)
    parser.add_argument("--output_path", type=str, default=output_path)

    # Model params
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    args = parser.parse_args()

    logger.info("Loading the core dataset...")
    ds = datasets.load_dataset(
        "parquet",
        data_files={
            sp: str(Path(args.dataset_path) / sp / "*.parquet") for sp in ('train', 'tuning', 'held_out')
        },
    )

    logger.info("Loading the task dataframe...")
    task_df = pl.read_parquet(task_df_path)

    logger.info("Applying transformations...")
    raise NotImplementedError("TODO: Apply transformations")

    logger.info("Getting vocab size...")
    vocab_size = get_vocab_size(ds['train'])

    model = Model(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
    )

    logger.info("Training the model...")

if __name__ == "__main__":
    main()
