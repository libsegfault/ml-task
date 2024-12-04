import argparse
from pathlib import Path

import datasets

from funccraft.data import load_dataset, prepare, save_dataset
from funccraft.models import append_extra_id, predict

def main():
    args = parse_args()
    args.func(args)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    default_data_path = Path('./prepared-dataset')
    prepare_data_parser = subparsers.add_parser('prepare-data')
    prepare_data_parser.set_defaults(func=prepare_data)
    prepare_data_parser.add_argument(
        '-o',
        '--output',
        help='Path to save prepared dataset to',
        type=Path,
        default=default_data_path,
    )
    prepare_data_parser.add_argument(
        '-n',
        '--num',
        help='Limit dataset to first --num functions',
        type=int,
        default=1000
    )
    prepare_data_parser.add_argument(
        '-l',
        '--language',
        help='Language (go|python)',
        default='python',
    )

    predict_parser = subparsers.add_parser('predict-names')
    predict_parser.set_defaults(func=predict_names)
    predict_parser.add_argument(
        '-d',
        '--dataset',
        help='Path to prepared dataset',
        type=Path,
        default=default_data_path,
    )
    predict_parser.add_argument(
        '-m',
        '--model',
        default='Salesforce/codet5p-220m',
    )
    predict_parser.add_argument(
        '-n',
        '--num',
        help='Limit dataset to first --num functions',
        type=int,
        default=1000
    )
    predict_parser.add_argument(
        '-l',
        '--language',
        help='Language (go|python)',
        default='python',
    )
    predict_parser.add_argument(
        '-c',
        '--comments',
        help='Use comments in predictions',
        type=bool,
        default=False
    )
    return parser.parse_args()


def prepare_data(args):
    dataset = datasets.load_dataset(
        'code_search_net',
        args.language,
        split='test',
        trust_remote_code=True
    )
    dataset = dataset.select(range(args.num))
    dataset = datasets.Dataset.from_list([prepare(x, args.language) for x in dataset])
    save_dataset(dataset, args.output)


def predict_names(args):
    dataset = load_dataset(args.dataset)
    dataset = dataset.select(range(args.num))

    src = 'my_body' if args.comments else 'my_no_comm'
    dataset = datasets.Dataset.from_list([append_extra_id(x, args.language) for x in dataset])
    predict(dataset, src, args.model)

if __name__ == '__main__':
    main()
