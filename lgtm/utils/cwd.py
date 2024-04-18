import os
from contextlib import contextmanager


@contextmanager
def temporary_change_cwd(destination):
    current_dir = os.getcwd()
    try:
        os.chdir(destination)
        yield
    finally:
        os.chdir(current_dir)
