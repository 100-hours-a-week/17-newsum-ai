# app/utils/agent_wrapper.py
import logging
import functools
import inspect
import traceback
from typing import Callable, Dict, Any, Optional, Awaitable
from app.services.storage_service import StorageService
from app.workflows.state import ComicState

logger = logging.getLogger(__name__)
storage_service = StorageService()  # 싱글턴 패턴 적용 고려

def save_agent_result(agent_func: Callable) -> Callable:
    """
    에이전트 함수 또는 메서드 데코레이터
    처리 결과를 자동으로 JSON 파일로 저장

    Args:
        agent_func: 래핑할 에이전트 함수 또는 메서드

    Returns:
        래핑된 함수
    """
    @functools.wraps(agent_func)
    async def wrapper(*args, **kwargs) -> Dict[str, Any]:
        # 1. 함수 정보 추출
        agent_name = agent_func.__qualname__.split('.')[0].lower() if '.' in agent_func.__qualname__ else agent_func.__name__
        if agent_name == 'run' and len(args) > 0 and hasattr(args[0], '__class__'):
            # 클래스 메서드인 경우 클래스 이름 가져오기
            agent_name = args[0].__class__.__name__.lower()
        
        # 2. 상태 객체 찾기
        state = None
        comic_id = None
        
        # 인자에서 상태 객체 찾기
        for arg in args:
            if isinstance(arg, ComicState):
                state = arg
                comic_id = getattr(state, 'comic_id', None)
                break
        
        # 키워드 인자에서 상태 객체 찾기
        if state is None:
            state = kwargs.get('state')
            if state and isinstance(state, ComicState):
                comic_id = getattr(state, 'comic_id', None)
        
        # 3. comic_id 찾기
        if comic_id is None and state is not None:
            # 상태 객체에서 comic_id 추출 시도
            if isinstance(state, dict):
                comic_id = state.get('comic_id')
            else:
                comic_id = getattr(state, 'comic_id', None)
        
        # comic_id가 없으면 임시 ID 생성
        if comic_id is None:
            import uuid
            comic_id = f"temp_{str(uuid.uuid4())[:8]}"
            logger.warning(f"{agent_name}: comic_id를 찾을 수 없어 임시 ID 생성: {comic_id}")
        
        # 4. 입력 상태 저장 (옵션)
        if state is not None:
            try:
                input_data = state
                if hasattr(state, 'model_dump'):
                    input_data = state.model_dump()  # Pydantic v2
                elif hasattr(state, 'dict'):
                    input_data = state.dict()  # Pydantic v1
                
                storage_service.save_agent_result(
                    comic_id=comic_id,
                    agent_name=agent_name,
                    data={'type': 'input', 'data': input_data},
                    subfolder='inputs'
                )
            except Exception as e:
                logger.error(f"{agent_name}: 입력 상태 저장 중 오류 발생: {e}")
        
        # 5. 원본 함수 실행
        start_time = None
        try:
            import time
            start_time = time.time()
            
            # 로깅
            args_str = ", ".join([str(a)[:50] + ('...' if len(str(a)) > 50 else '') for a in args[1:]] if len(args) > 1 else [])
            kwargs_str = ", ".join([f"{k}={v}"[:50] for k, v in kwargs.items()])
            logger.info(f"{agent_name} 실행 시작: args=[{args_str}], kwargs=[{kwargs_str}]")
            
            # 함수 실행
            result = await agent_func(*args, **kwargs)
            
            # 실행 시간 측정
            execution_time = time.time() - start_time if start_time else None
            
            # 로깅
            if execution_time:
                logger.info(f"{agent_name} 실행 완료: {execution_time:.2f}초 소요")
            else:
                logger.info(f"{agent_name} 실행 완료")
                
            # 6. 결과 저장
            if result is not None:
                try:
                    # 실행 시간 및 상태 정보 추가
                    result_data = {
                        'result': result,
                        'execution_time': execution_time,
                        'status': 'success'
                    }
                    
                    storage_service.save_agent_result(
                        comic_id=comic_id,
                        agent_name=agent_name,
                        data=result_data
                    )
                except Exception as e:
                    logger.error(f"{agent_name}: 결과 저장 중 오류 발생: {e}")
            
            return result
            
        except Exception as e:
            # 7. 오류 발생 시 오류 정보 저장
            error_time = time.time() - start_time if start_time else None
            logger.error(f"{agent_name} 실행 중 오류 발생: {e}")
            
            try:
                error_data = {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc(),
                    'execution_time': error_time,
                    'status': 'error'
                }
                
                storage_service.save_agent_result(
                    comic_id=comic_id,
                    agent_name=agent_name,
                    data=error_data,
                    subfolder='errors'
                )
            except Exception as save_error:
                logger.error(f"{agent_name}: 오류 정보 저장 중 추가 오류 발생: {save_error}")
            
            # 원래 예외 다시 발생
            raise
    
    return wrapper

def wrap_all_agents():
    """
    모든 에이전트 자동 래핑 함수
    app/agents 디렉토리의 모든 에이전트 클래스에 데코레이터 적용
    """
    import importlib
    import inspect
    import os
    import sys
    from pathlib import Path
    
    # 에이전트 모듈 경로
    agents_dir = Path(__file__).parent.parent / 'agents'
    
    # 모듈 가져오기
    for file_path in agents_dir.glob('*_agent.py'):
        module_name = f"app.agents.{file_path.stem}"
        try:
            # 모듈 동적 임포트
            module = importlib.import_module(module_name)
            
            for name, obj in inspect.getmembers(module):
                # 클래스 찾기
                if inspect.isclass(obj) and name.endswith('Agent') and hasattr(obj, 'run'):
                    # run 메서드에 데코레이터 적용
                    if isinstance(obj.run, property):
                        continue  # property는 건너뛰기
                    
                    if not hasattr(obj.run, '__wrapped__'):  # 중복 래핑 방지
                        logger.info(f"에이전트 {name}.run 메서드 래핑 중...")
                        original_run = obj.run
                        obj.run = save_agent_result(original_run)
                        logger.info(f"에이전트 {name}.run 메서드 래핑 완료")
        
        except Exception as e:
            logger.error(f"모듈 {module_name} 래핑 중 오류 발생: {e}")
            continue

# 함수 에이전트용 래퍼 (클래스 메서드가 아닌 독립 함수용)
def wrap_function_agent(func_name, module_path):
    """
    특정 모듈의 함수를 에이전트 래퍼로 감싸기
    
    Args:
        func_name: 래핑할 함수 이름
        module_path: 함수가 포함된 모듈 경로 (예: 'app.agents.collector_agent')
    
    Returns:
        래핑된 함수
    """
    import importlib
    
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, func_name):
            original_func = getattr(module, func_name)
            wrapped_func = save_agent_result(original_func)
            setattr(module, func_name, wrapped_func)
            logger.info(f"함수 {module_path}.{func_name} 래핑 완료")
            return wrapped_func
        else:
            logger.error(f"함수 {func_name}를 모듈 {module_path}에서 찾을 수 없음")
            return None
    except Exception as e:
        logger.error(f"함수 {module_path}.{func_name} 래핑 중 오류 발생: {e}")
        return None
