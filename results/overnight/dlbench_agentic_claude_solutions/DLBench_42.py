import sys
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, BinaryIO, Iterator, TextIO, Union
import psutil
import nncf

def fail_if_symlink(file: Path) -> None:
    if file.is_symlink():
        raise nncf.ValidationError('File {} is a symbolic link, aborting.'.format(str(file)))

@contextmanager
def safe_open(file: Path, *args, **kwargs) -> Iterator[Union[TextIO, BinaryIO, IO[Any]]]:
    """
    Safe function to open file and return a stream.

    For security reasons, should not follow symlinks. Use .resolve() on any Path
    objects before passing them here.

    :param file: The path to the file.
    :return: A file object.
    """
    fail_if_symlink(file)
    with open(str(file), *args, **kwargs) as f:
        yield f

def is_windows() -> bool:
    return 'win32' in sys.platform

def is_linux() -> bool:
    return 'linux' in sys.platform

def get_available_cpu_count(logical: bool=True) -> int:
    """Generate a python function called get_available_cpu_count that returns the number of CPUs in the system. The input parameter is a boolean called logical, which determines whether to return the number of physical cores only (if False) or the number of logical cores (if True). The output is an integer representing the number of CPUs. If an exception occurs, the function will return 1. Default value of logical is True"""
    try:
        count = psutil.cpu_count(logical=logical)
        if count is None:
            return 1
        return count
    except Exception:
        return 1

def get_available_memory_amount() -> float:
    """
    :return: Available memory amount (bytes)
    """
    try:
        return psutil.virtual_memory()[1]
    except Exception:
        return 0
