# app/utils/retry.py

import asyncio
from functools import wraps

def retry_async(max_retries=3, delay=2):
    """
    비동기 함수에 적용할 수 있는 재시도 데코레이터
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    await asyncio.sleep(delay)
        return wrapper
    return decorator


'''
# Example usage
from app.utils.retry import retry_async

@retry_async(max_retries=5, delay=1)
async def call_llm_server():
    ...

'''