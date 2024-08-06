from typing import Any, Iterable
from hashlib import md5
from random import Random, getrandbits
import uuid


# this is how the id is generated.
def gen_md5_hash(item: dict[str, Any], hashcode: Iterable[str]):
    """Generate an md5 hash."""
    hashed = "".join([str(item[column]) for column in hashcode])
    return f"{md5(hashed.encode('utf-8'), usedforsecurity=False).hexdigest()}"


def gen_uuid(rd: Random | None = None):
    """Generate a random UUID v4."""
    return uuid.UUID(
        int=rd.getrandbits(128) if rd is not None else getrandbits(128), version=4
    ).hex
