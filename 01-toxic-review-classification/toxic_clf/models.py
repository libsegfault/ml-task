from statistics import mean, stdev

import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def classifier(dataset, model):
    # Use real X and y from dataset
    X = np.random.rand(100, 3)
    y = np.random.randint(2, size=100)

    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    scores = []
    for train_index, test_index in tqdm(skf.split(X, y)):
        # Fit and evaluate model on each fold
        pass

    print('Some pretty description:')
    print(mean(scores), stdev(scores))
