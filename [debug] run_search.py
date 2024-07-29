'''
Mainly used in debugging, please use `run_search.ipynb` for actually searching purpose. 
'''

from search.searcher import Searcher
from llm.openai import send_to_openai
import argparse
import logging
import os

# fmt: off
def get_args_parser():
    parser = argparse.ArgumentParser("Index documents", add_help=False)
    parser.add_argument("--input_dir", default=None, type=str, help="Output directory for generated index files.",)
    parser.add_argument("--query", default=None, type=str, help="Query used for searching questions.")
    return parser
# fmt: on


def main(args):

    if (args.input_dir is None) or (not os.path.exists(args.input_dir)):
        raise ValueError(f"The input directory [{args.input_dir}] does not exist.")

    logging.basicConfig(
        # filename=f"{args.name}.log",
        level=logging.INFO,
    )

    searcher = Searcher(
        input_dir=args.input_dir,
        send_to=send_to_openai,
    )

    result = searcher.search(
        query=args.query,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Searching script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
