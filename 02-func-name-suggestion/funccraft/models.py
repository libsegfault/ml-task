from collections.abc import Iterable
from pprint import pprint

import torch
from functools import lru_cache

import datasets
import evaluate

from transformers import AutoTokenizer, T5ForConditionalGeneration

import re

import funccraft.python as python
import funccraft.go as go

def append_extra_id(dataset, lang: str):
    extra_id = eval(lang).extra_id
    dataset['my_no_comm'] = extra_id + '\n' + dataset['my_no_comm']
    dataset['my_body'] = extra_id + dataset['my_body']
    return dict(dataset)

def has_invalid_identifier_start(x):
    return len(x) > 0 and x[0].isdigit()

def postprocess(ans):
    # Name cannot have spaces, so take the first word if present
    ans = ans.split(' ')
    ans = ans[1] if len(ans) > 1 else ans[0] if len(ans) == 1 else ''

    ans = ans.replace('\n', '')
    ans = ans.replace('\t', '')

    # Try to fixup invalid identifier names
    if has_invalid_identifier_start(ans):
        ans = f'_{ans}'

    res = ''

    for c in ans:
        if not c.isalnum() and c != '_':
            break
        res += c

    return res

def predict(dataset: datasets.Dataset, src: str, model_str: str) -> None:
    torch.cuda.empty_cache()
    which_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(which_device)

    tokenizer = AutoTokenizer.from_pretrained(model_str)
    model = T5ForConditionalGeneration.from_pretrained(model_str).to(device)

    inputs = tokenizer(dataset[src],
                       return_tensors='pt',
                       padding=True,
                       truncation=True,
                       max_length=80,
                       ).to(device)
    outputs = model.generate(**inputs, max_length=80)
    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    preds = [postprocess(p) for p in preds]

    eval_results = run_evaluate(predictions=preds, references=dataset['my_name'])
    print()
    print('*' * 80)
    print('Evaluation results:')
    pprint(eval_results)
    print('*' * 80)
    print()


def run_evaluate(predictions: Iterable[str], references: Iterable[str]) -> dict[str, float]:
    em = evaluate.load('exact_match')
    rouge =  evaluate.load('rouge')
    em_score = em.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    return {**rouge_scores, **em_score}
