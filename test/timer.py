from contextlib import contextmanager
import time


@contextmanager
def timer(name, na_sep):
    t0 = time.time()
    yield
    print(f'[{name}] done in {round((time.time() - t0), na_sep)} s')
