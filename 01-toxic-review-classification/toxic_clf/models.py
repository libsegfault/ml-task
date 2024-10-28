from statistics import mean, stdev
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from nltk import word_tokenize
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from pprint import pprint
import nltk
import os

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from tqdm import tqdm, trange

import warnings
# Disable all warnings
warnings.filterwarnings("ignore")

mystop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ourselves', 'you', 'your', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'themselves', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'if', 'or', 'as', 'until', 'of', 'at', 'by', 'between', 'into', 'through', 'during', 'to', 'from', 'in', 'out', 'on', 'off', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'other', 'some', 'such', 'than', 'too', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now']

class classic_ml:
    def __init__(self, X_train, y_train, tuning=False):
        self.vectorizer = TfidfVectorizer(tokenizer=word_tokenize, sublinear_tf=True, max_df=0.5, stop_words=mystop_words, min_df=20)
        self.mapper = None
        self.y = None
        self.X = None

        self.clf = RandomForestClassifier(n_jobs=-1, max_features='sqrt', min_samples_split=3,  n_estimators=100)
        #self.clf = RandomForestClassifier(n_jobs=-1, min_samples_split=5)
        self.__prepare_data(X_train, y_train)
        if tuning:
            self.grid_search_parameter()
        else:
            self.model = self.train()

    def __prepare_data(self, X_train, y_train):
        self.mapper = DataFrameMapper([
            ('message', self.vectorizer),
        ])
        self.y = np.ravel(y_train)
        self.X = self.mapper.fit_transform(X_train)

    def grid_search_parameter(self):
        param_gid=None

        param_grid = {
            'max_depth': [ 10, 20, 50, None],
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'min_samples_split': [2,  4,  6, 7, 8, 10],
            'n_estimators': [100, 200, 300, 400, 500, 750, 1000]
        }
        grid_search_model = GridSearchCV(estimator=self.clf, param_grid=param_grid,
                                         cv=10, n_jobs=-1, verbose=3, return_train_score=True)
        grid_search_model.fit(self.X, self.y)
        pprint(grid_search_model.best_params_)
        best_grid = grid_search_model.best_estimator_

    def train(self):
        self.clf.fit(self.X, self.y)
        return self.clf

    def predict(self, X_test):
        X_test_mapped = self.mapper.transform(X_test)
        predictions = self.model.predict(X_test_mapped)
        return np.expand_dims(predictions, 1)

import torch
import pandas as pd
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_metric, Dataset, load_metric

torch.autograd.set_detect_anomaly(True)

# Define TinyTransformer model
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 512, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return self.sigmoid(x)

class TinyTransformerConfig(PretrainedConfig):
    model_type = "tiny_transformer"

    def __init__(self, vocab_size=30522, embed_dim=64, num_heads=2, ff_dim=128, num_layers=4, max_position_embeddings=512, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.max_position_embeddings = max_position_embeddings

class TinyTransformerForSequenceClassification(PreTrainedModel):
    config_class = TinyTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.loss_fn = nn.BCELoss()
        self.transformer = TinyTransformer(
            config.vocab_size,
            config.embed_dim,
            config.num_heads,
            config.ff_dim,
            config.num_layers
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids)
        loss = None

        if labels is not None:
            labels = labels.float()
            loss = self.loss_fn(outputs, labels.unsqueeze(-1))

        return {
            "logits": outputs,
            "loss": loss,
        }

# Load the Tiny-Toxic-Detector model and tokenizer
def load_model_and_tokenizer(train):
    device = torch.device("cpu") # Due to GPU overhead inference is faster on CPU!

    # Load Tiny-toxic-detector
    config = TinyTransformerConfig.from_pretrained("AssistantsLab/Tiny-Toxic-Detector")
    model = TinyTransformerForSequenceClassification.from_pretrained("AssistantsLab/Tiny-Toxic-Detector" if train else "./finetuned", config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("AssistantsLab/Tiny-Toxic-Detector")

    return model, tokenizer, device

# Prediction function
def is_toxic(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length").to(device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs["logits"].squeeze()
    return logits > 0.5

class bert:
    def __init__(self, X_train, y_train):
        train = not os.path.exists('./finetuned')
        self.model, self.tokenizer, self.device = load_model_and_tokenizer(train)
        if train:
            self.train(X_train, y_train)

    def train(self, X, y):
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

        training_args = TrainingArguments(output_dir="train", num_train_epochs=10)
        metric = load_metric("f1")

        train_ = X_train.copy()
        eval_ = X_eval.copy()

        train_['labels'] = y_train
        eval_['labels'] = y_eval

        train_ = Dataset.from_pandas(train_)
        eval_ = Dataset.from_pandas(eval_)

        def tokenize_function(examples):
            return self.tokenizer(examples['message'], return_tensors="pt", truncation=True, max_length=128, padding="max_length")

        train_tok = train_.map(tokenize_function, batched=True)
        eval_tok = eval_.map(tokenize_function, batched=True)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=train_tok,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.evaluate()
        trainer.save_model("finetuned")

    def predict(self, X_test):
        ans = []
        for text in X_test['message']:
            text = text[:128]
            ans.append(int(is_toxic(text, self.model, self.tokenizer, self.device)))
        return ans

class SplitOnce:
    def split(self, X, y):
        X_i = np.array([i for i in range(len(X))])
        y_i = np.array([i for i in range(len(X))])
        X_train, X_test, y_train, y_test = train_test_split(X_i, y_i, test_size=0.2, random_state=42)
        yield X_train, X_test

def classifier(dataset, model_name):
    data = dataset.to_pandas().dropna()

    X = data[['message']]
    y = data['is_toxic']

    skf = None
    if model_name != 'bert':
        skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    else:
        skf = SplitOnce()

    scores = {'precision': [], 'recall': [], 'f': []}
    confusions = []
    for train_index, test_index in tqdm(skf.split(X, y)):
        # Fit and evaluate model on each fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = eval(f'{model_name}(X_train, y_train)')
        res = model.predict(X_test)
        precision, recall, fscore, support = precision_recall_fscore_support(res, y_test, average='binary')
        scores['precision'] += [precision]
        scores['recall'] += [recall]
        scores['f'] += [fscore]
        confusions += [confusion_matrix(y_test, res)]

    avg_confusion = [[0, 0], [0, 0]]
    for conf in confusions:
        for i in range(len(conf)):
            for j in range(len(conf[i])):
                avg_confusion[i][j] += int(conf[i][j])

    for i in range(len(conf)):
        for j in range(len(conf[i])):
            avg_confusion[i][j] /= len(confusions)

    print('Some pretty description:')
    print(avg_confusion)
    for key in scores:
        print(key, '->', mean(scores[key]), stdev(scores[key]) if len(scores[key]) > 1 else '')
