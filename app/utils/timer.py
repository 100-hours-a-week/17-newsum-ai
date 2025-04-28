# app/utils/timer.py

import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    """
    코드 블록 실행 시간을 측정하는 타이머
    """
    start = time.time()
    yield
    end = time.time()
    print(f"[{name}] Elapsed: {end - start:.2f}s")


'''
# Example usage
from app.utils.timer import timer

with timer("workflow-run"):
    result = await some_workflow()

'''