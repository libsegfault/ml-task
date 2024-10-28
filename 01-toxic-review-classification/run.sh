#!/bin/bash

sudo pacman -S python-nltk python-openpyxl python-numpy python-scikit-learn python-tqdm

pip install -r requirements.txt
python main.py prepare-data ../../ToxiCR/models/code-review-dataset-full.xlsx
python main.py classify classic_ml
python main.py classify bert
