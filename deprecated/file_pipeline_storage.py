from pathlib import Path
from typing import Any, cast


def join_path(file_path: str, file_name: str) -> Path:
    """Join a path and a file. Independent of the OS."""
    return Path(file_path) / Path(file_name).parent / Path(file_name).name


_encoding = "utf-8"

def file_pipeline_storage_set(
    root, key: str, value: Any, encoding: str | None = None
) -> None:
    """Set method definition."""
    is_bytes = isinstance(value, bytes)
    write_type = "wb" if is_bytes else "w"
    encoding = None if is_bytes else encoding or _encoding
    with open(join_path(root, key), cast(Any, write_type), encoding=encoding) as f:
        f.write(value)

def read_file(
    path: str | Path,
    as_bytes: bool | None = False,
    encoding: str | None = None,
) -> Any:
    """Read the contents of a file."""
    read_type = "rb" if as_bytes else "r"
    encoding = None if as_bytes else (encoding or _encoding)

    with open(
        path,
        cast(Any, read_type),
        encoding=encoding,
    ) as f:
        return f.read()


def file_pipeline_storage_get(
    root: str, key: str, as_bytes: bool | None = False, encoding: str | None = None
) -> Any:
    """Get method definition."""
    file_path = join_path(root, key)
    return read_file(file_path, as_bytes, encoding)


def parquet_table_emit(root, name, df):
    filename = f"{name}.parquet"
    file_pipeline_storage_set(root, filename, df.to_parquet())


def parquet_table_load(root, name, as_byte: bool = True):
    filename = f"{name}.parquet"
    file_pipeline_storage_get(root, filename, as_byte)
