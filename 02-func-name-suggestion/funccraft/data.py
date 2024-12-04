from pathlib import Path

import datasets

from tree_sitter import Language, Parser

import funccraft.python as python
import funccraft.go as go

def prepare(dataset, lang: str) -> datasets.Dataset:
    language = eval(lang).lang
    parser = Parser(language)

    code = dataset['whole_func_string']
    ast = parser.parse(bytes(code, "utf-8"))
    query = language.query(eval(lang).query)
    captures = query.captures(ast.root_node)

    data = {
        'name': '',
        'body': '',
        'comm': [],
        'no_comm': '',
    }

    for key, nodes in captures.items():
        for node in nodes:
            text = code[node.start_byte:node.end_byte]
            for n in data:
                if n == key:
                    if n == 'comm':
                        data[n] += [text]
                    else:
                        data[n] += text

    data['no_comm'] = data['body']

    for c in data['comm']:
        data['no_comm'] = data['no_comm'].replace(c, '')

    del data['comm']

    for k, v in data.items():
        dataset['my_' + k] = v

    return dict(dataset)

def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))
