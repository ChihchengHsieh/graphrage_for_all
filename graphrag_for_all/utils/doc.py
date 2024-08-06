import pandas as pd

from cleantext import clean
from langchain_core.documents.base import Document as LC_doc
from typing import List
from utils.hash import gen_md5_hash


def clean_docs(docs):
    for d in docs:
        d.page_content = clean(
            d.page_content,
            # no_line_breaks=True,
        )
    return docs


def langhchain_doc_to_df(lc_docs: List[LC_doc]) -> pd.DataFrame:
    """
    This function load the equivelent dataset for you.
    """
    return pd.DataFrame(
        [
            {
                "id": gen_md5_hash({"text": d.page_content}, ["text"]),
                "text": d.page_content,
                "title": d.metadata["title"],
            }
            for i, d in enumerate(lc_docs)
        ]
    )
