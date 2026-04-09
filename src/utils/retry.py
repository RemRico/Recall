# -*- coding: utf-8 -*-
import time
from typing import Callable, Type, Tuple

def retry(fn: Callable, *, tries: int = 3, delay: float = 0.5,
          exceptions: Tuple[Type[BaseException], ...] = (Exception,), backoff: float = 2.0):
    """Small retry wrapper/decorator: retry(lambda: do(), tries=3, delay=0.5)."""
    def _call(*args, **kwargs):
        t, d = tries, delay
        while True:
            try:
                return fn(*args, **kwargs)
            except exceptions:
                t -= 1
                if t <= 0:
                    raise
                time.sleep(d)
                d *= backoff
    return _call
