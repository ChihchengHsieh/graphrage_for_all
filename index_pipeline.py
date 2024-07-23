from collections.abc import Iterable
from hashlib import md5
from typing import Any

# this is how the id is generated.
# def gen_md5_hash(item: dict[str, Any], hashcode: Iterable[str]):
#     """Generate an md5 hash."""
#     hashed = "".join([str(item[column]) for column in hashcode])
#     return f"{md5(hashed.encode('utf-8'), usedforsecurity=False).hexdigest()}"


# new_item = {"text": text} # text of the file
# new_item["id"] = gen_md5_hash(new_item, new_item.keys())

# once this dataset is loaded, then we start running the pipelines.

from collections.abc import Iterable
from hashlib import md5
from typing import Any, List, Dict
import pandas as pd


# this is how the id is generated.
def gen_md5_hash(item: dict[str, Any], hashcode: Iterable[str]):
    """Generate an md5 hash."""
    hashed = "".join([str(item[column]) for column in hashcode])
    return f"{md5(hashed.encode('utf-8'), usedforsecurity=False).hexdigest()}"

def lc_doc_to_df(lc_docs: List[LC_doc]) -> pd.DataFrame:
    '''
    This function load the equivelent dataset for you.
    '''
    return pd.DataFrame(
        [
            {
                "id": gen_md5_hash({'text': d.page_content} ,['text']),
                "text": d.page_content,
                "title": d.metadata["title"],
            }
            for i, d in enumerate(lc_docs)
        ]
    )
    