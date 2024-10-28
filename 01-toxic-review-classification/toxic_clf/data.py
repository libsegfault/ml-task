from pathlib import Path

import datasets
import pandas as pd


def prepare(raw_data: Path) -> datasets.Dataset:
    dataset = pd.read_excel(raw_data)
    #actual_data = dataset.set_index('message').T.to_dict()
    #
    #for key in actual_data:
    #    actual_data[key] = actual_data[key]['is_toxic']
    #return datasets.Dataset.from_dict(actual_data)
    return datasets.Dataset.from_pandas(dataset.drop_duplicates(subset=['message']))


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))
