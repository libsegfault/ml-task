from pathlib import Path

import datasets
import pandas as pd


def prepare(raw_data: Path) -> datasets.Dataset:
    dataset = pd.read_excel(raw_data)
    # Implement dataset preparation code here
    return datasets.Dataset.from_dict({'fake_data': [[1, 2], [3, 4]]})


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))
