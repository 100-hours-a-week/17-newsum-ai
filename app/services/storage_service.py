# app/services/storage_service.py
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from app.config.settings import settings

logger = logging.getLogger(__name__)

class StorageService:
    """
    에이전트 결과를 JSON 파일로 저장하고 관리하는 서비스
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        StorageService 초기화
        
        Args:
            base_dir: 결과 파일 저장 기본 디렉토리. 기본값은 settings.RESULTS_DIR
        """
        # 기본 저장 경로는 settings에서 가져옴
        self.base_dir = base_dir or settings.RESULTS_DIR
        os.makedirs(self.base_dir, exist_ok=True)
        
        # 저장 설정
        self.save_results = settings.SAVE_AGENT_RESULTS
        self.save_inputs = settings.SAVE_AGENT_INPUTS
        self.save_debug = settings.SAVE_DEBUG_INFO
        
        logger.info(f"StorageService 초기화 완료. 저장 경로: {self.base_dir}, 저장 활성화: {self.save_results}")
        
    def save_agent_result(self, 
                         comic_id: str, 
                         agent_name: str, 
                         data: Dict[str, Any], 
                         step: Optional[int] = None,
                         subfolder: Optional[str] = None) -> str:
        """
        에이전트 처리 결과를 JSON 파일로 저장
        
        Args:
            comic_id: 만화 생성 작업 ID
            agent_name: 에이전트 이름 (예: collector, scraper, summarizer 등)
            data: 저장할 데이터 (dict 형태)
            step: 처리 단계 (옵션)
            subfolder: 추가 하위 폴더 (옵션)
            
        Returns:
            저장된 파일 경로
        """
        # 저장 기능이 비활성화되어 있으면 바로 리턴
        if not self.save_results:
            return ""
            
        # 'inputs' 저장이 비활성화되어 있고, subfolder가 'inputs'인 경우 저장 안 함
        if subfolder == 'inputs' and not self.save_inputs:
            return ""
            
        # 'errors'나 'debug' 저장이 비활성화되어 있는 경우
        if not self.save_debug and subfolder in ['errors', 'debug']:
            return ""
        
        # 1. 저장 디렉토리 구성 (comic_id/[subfolder])
        save_dir = os.path.join(self.base_dir, comic_id)
        if subfolder:
            save_dir = os.path.join(save_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)
        
        # 2. 파일명 구성 (agent_name_timestamp.json 또는 agent_name_step.json)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if step is not None:
            filename = f"{agent_name}_{step:03d}.json"
        else:
            filename = f"{agent_name}_{timestamp}.json"
        
        file_path = os.path.join(save_dir, filename)
        
        # 3. 결과 메타데이터 추가
        result_data = data.copy()  # 원본 데이터 복사
        result_data.update({
            "agent": agent_name,
            "comic_id": comic_id,
            "timestamp": datetime.now().isoformat(),
            "step": step
        })
        
        # 4. JSON 파일로 저장
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            logger.info(f"에이전트 결과 저장 완료: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"결과 저장 중 오류 발생: {e}")
            raise
    
    def get_agent_result(self, 
                        comic_id: str, 
                        agent_name: str, 
                        step: Optional[int] = None, 
                        latest: bool = True,
                        subfolder: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        저장된 에이전트 결과 파일 로드
        
        Args:
            comic_id: 만화 생성 작업 ID
            agent_name: 에이전트 이름
            step: 특정 단계 결과 조회 (옵션)
            latest: 가장 최근 결과 반환 여부 (기본값: True)
            subfolder: 하위 폴더 (옵션)
            
        Returns:
            에이전트 결과 데이터 또는 None (파일 없을 경우)
        """
        # 1. 조회 디렉토리 구성
        search_dir = os.path.join(self.base_dir, comic_id)
        if subfolder:
            search_dir = os.path.join(search_dir, subfolder)
        
        if not os.path.exists(search_dir):
            logger.warning(f"결과 디렉토리 없음: {search_dir}")
            return None
        
        # 2. 파일 검색 패턴 구성
        if step is not None:
            pattern = f"{agent_name}_{step:03d}.json"
            file_path = os.path.join(search_dir, pattern)
            if os.path.exists(file_path):
                logger.debug(f"에이전트 결과 파일 발견: {file_path}")
            else:
                logger.warning(f"결과 파일 없음: {file_path}")
                return None
        else:
            # 이름이 일치하는 모든 파일 찾기
            files = [f for f in os.listdir(search_dir) if f.startswith(f"{agent_name}_") and f.endswith(".json")]
            if not files:
                logger.warning(f"에이전트 '{agent_name}' 결과 파일 없음: {search_dir}")
                return None
            
            if latest:
                # 최신 파일 선택 (수정 시간 기준)
                file_path = os.path.join(search_dir, 
                                         max(files, key=lambda f: os.path.getmtime(os.path.join(search_dir, f))))
            else:
                # 첫 번째 파일 선택
                file_path = os.path.join(search_dir, files[0])
        
        # 3. 파일 로드
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"에이전트 결과 로드 완료: {file_path}")
            return data
        except Exception as e:
            logger.error(f"결과 로드 중 오류 발생: {e}")
            return None
            
    def list_results(self, comic_id: Optional[str] = None, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        저장된 결과 파일 목록 조회
        
        Args:
            comic_id: 특정 comic_id 결과만 조회 (옵션)
            agent_name: 특정 에이전트 결과만 조회 (옵션)
            
        Returns:
            결과 파일 정보 딕셔너리 (comic_id -> agent_name -> files)
        """
        results = {}
        
        # 1. 기본 디렉토리 또는 comic_id 디렉토리 탐색
        base_path = self.base_dir
        if comic_id:
            base_path = os.path.join(base_path, comic_id)
            if not os.path.exists(base_path):
                return results
            comic_ids = [comic_id]
        else:
            if not os.path.exists(base_path):
                return results
            comic_ids = [d for d in os.listdir(base_path) 
                        if os.path.isdir(os.path.join(base_path, d))]
        
        # 2. 각 comic_id 디렉토리 탐색
        for cid in comic_ids:
            results[cid] = {}
            comic_path = os.path.join(self.base_dir, cid)
            
            # JSON 파일만 필터링
            files = [f for f in os.listdir(comic_path) 
                    if f.endswith('.json') and os.path.isfile(os.path.join(comic_path, f))]
            
            # 에이전트 이름별로 분류
            for file in files:
                # 파일명에서 에이전트 이름 추출
                agent = file.split('_')[0] if '_' in file else 'unknown'
                
                if agent_name and agent != agent_name:
                    continue
                    
                if agent not in results[cid]:
                    results[cid][agent] = []
                
                file_info = {
                    'filename': file,
                    'path': os.path.join(comic_path, file),
                    'modified': datetime.fromtimestamp(
                        os.path.getmtime(os.path.join(comic_path, file))
                    ).isoformat()
                }
                results[cid][agent].append(file_info)
            
            # 하위 디렉토리 검사 (inputs, errors, debug 등)
            subdirs = [d for d in os.listdir(comic_path) 
                      if os.path.isdir(os.path.join(comic_path, d))]
            
            for subdir in subdirs:
                subdir_path = os.path.join(comic_path, subdir)
                subdir_files = [f for f in os.listdir(subdir_path) 
                               if f.endswith('.json') and os.path.isfile(os.path.join(subdir_path, f))]
                
                for file in subdir_files:
                    # 파일명에서 에이전트 이름 추출
                    agent = file.split('_')[0] if '_' in file else 'unknown'
                    
                    if agent_name and agent != agent_name:
                        continue
                    
                    agent_key = f"{agent}_{subdir}"  # 하위 폴더 구분을 위해
                    if agent_key not in results[cid]:
                        results[cid][agent_key] = []
                    
                    file_info = {
                        'filename': file,
                        'path': os.path.join(subdir_path, file),
                        'subfolder': subdir,
                        'modified': datetime.fromtimestamp(
                            os.path.getmtime(os.path.join(subdir_path, file))
                        ).isoformat()
                    }
                    results[cid][agent_key].append(file_info)
        
        return results

    def clear_results(self, comic_id: Optional[str] = None) -> bool:
        """
        결과 파일 삭제
        
        Args:
            comic_id: 특정 작업 결과만 삭제 (옵션, None이면 모든 결과 삭제)
            
        Returns:
            성공 여부
        """
        try:
            if comic_id:
                target_dir = os.path.join(self.base_dir, comic_id)
                if os.path.exists(target_dir):
                    import shutil
                    shutil.rmtree(target_dir)
                    logger.info(f"comic_id '{comic_id}' 결과 삭제 완료")
                return True
            else:
                # 모든 결과 디렉토리 삭제
                if os.path.exists(self.base_dir):
                    import shutil
                    # results 디렉토리 내부 파일만 삭제하고 디렉토리는 유지
                    for item in os.listdir(self.base_dir):
                        item_path = os.path.join(self.base_dir, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                    logger.info("모든 결과 삭제 완료")
                return True
        except Exception as e:
            logger.error(f"결과 삭제 중 오류 발생: {e}")
            return False
