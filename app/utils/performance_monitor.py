# ai/app/utils/performance_monitor.py

import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    request_count: int = 0
    total_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    max_processing_time: float = 0.0
    error_count: int = 0
    timeout_count: int = 0
    
    @property
    def avg_processing_time(self) -> float:
        """평균 처리 시간"""
        return self.total_processing_time / self.request_count if self.request_count > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """오류율"""
        return self.error_count / self.request_count if self.request_count > 0 else 0.0
    
    @property
    def timeout_rate(self) -> float:
        """타임아웃율"""
        return self.timeout_count / self.request_count if self.request_count > 0 else 0.0


class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self, window_size: int = 1000, report_interval: int = 300):
        """
        Args:
            window_size: 슬라이딩 윈도우 크기 (최근 N개 요청)
            report_interval: 보고서 출력 간격 (초)
        """
        self.window_size = window_size
        self.report_interval = report_interval
        
        # 전체 메트릭
        self.global_metrics = PerformanceMetrics()
        
        # 슬라이딩 윈도우 메트릭 (최근 N개 요청)
        self.recent_times: deque = deque(maxlen=window_size)
        self.recent_errors: deque = deque(maxlen=window_size)
        self.recent_timeouts: deque = deque(maxlen=window_size)
        
        # 서비스별 메트릭
        self.service_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        
        # 시간대별 메트릭 (시간당)
        self.hourly_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        
        # 마지막 보고서 시간
        self.last_report_time = time.time()
        
        # vLLM 관련 메트릭
        self.vllm_concurrent_requests = 0
        self.vllm_queue_size = 0
        self.vllm_response_quality_scores = deque(maxlen=100)
        
        # Redis 관련 메트릭
        self.redis_pub_count = 0
        self.redis_sub_count = 0
        self.redis_errors = 0
    
    def record_request(
        self, 
        service: str, 
        processing_time: float, 
        success: bool = True, 
        is_timeout: bool = False,
        quality_score: Optional[float] = None
    ):
        """요청 기록"""
        try:
            # 현재 시간 (시간대별 메트릭용)
            current_hour = datetime.now().strftime("%Y-%m-%d-%H")
            
            # 전체 메트릭 업데이트
            self._update_metrics(self.global_metrics, processing_time, success, is_timeout)
            
            # 서비스별 메트릭 업데이트
            self._update_metrics(self.service_metrics[service], processing_time, success, is_timeout)
            
            # 시간대별 메트릭 업데이트
            self._update_metrics(self.hourly_metrics[current_hour], processing_time, success, is_timeout)
            
            # 슬라이딩 윈도우 업데이트
            self.recent_times.append(processing_time)
            self.recent_errors.append(not success)
            self.recent_timeouts.append(is_timeout)
            
            # vLLM 품질 점수 기록
            if quality_score is not None:
                self.vllm_response_quality_scores.append(quality_score)
            
            # 주기적 보고서 출력
            if time.time() - self.last_report_time >= self.report_interval:
                self._generate_report()
                self.last_report_time = time.time()
                
        except Exception as e:
            logger.exception(f"Error recording performance metric: {e}")
    
    def _update_metrics(
        self, 
        metrics: PerformanceMetrics, 
        processing_time: float, 
        success: bool, 
        is_timeout: bool
    ):
        """메트릭 업데이트"""
        metrics.request_count += 1
        metrics.total_processing_time += processing_time
        metrics.min_processing_time = min(metrics.min_processing_time, processing_time)
        metrics.max_processing_time = max(metrics.max_processing_time, processing_time)
        
        if not success:
            metrics.error_count += 1
        
        if is_timeout:
            metrics.timeout_count += 1
    
    def get_recent_metrics(self) -> Dict[str, Any]:
        """최근 윈도우 메트릭 반환"""
        if not self.recent_times:
            return {
                "request_count": 0,
                "avg_processing_time": 0.0,
                "error_rate": 0.0,
                "timeout_rate": 0.0
            }
        
        return {
            "request_count": len(self.recent_times),
            "avg_processing_time": sum(self.recent_times) / len(self.recent_times),
            "min_processing_time": min(self.recent_times),
            "max_processing_time": max(self.recent_times),
            "error_rate": sum(self.recent_errors) / len(self.recent_errors),
            "timeout_rate": sum(self.recent_timeouts) / len(self.recent_timeouts),
            "p95_processing_time": self._calculate_percentile(list(self.recent_times), 95),
            "p99_processing_time": self._calculate_percentile(list(self.recent_times), 99)
        }
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """백분위수 계산"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def get_vllm_metrics(self) -> Dict[str, Any]:
        """vLLM 관련 메트릭 반환"""
        avg_quality = (sum(self.vllm_response_quality_scores) / len(self.vllm_response_quality_scores) 
                      if self.vllm_response_quality_scores else 0.0)
        
        return {
            "concurrent_requests": self.vllm_concurrent_requests,
            "queue_size": self.vllm_queue_size,
            "avg_quality_score": avg_quality,
            "quality_scores_count": len(self.vllm_response_quality_scores)
        }
    
    def get_redis_metrics(self) -> Dict[str, Any]:
        """Redis 관련 메트릭 반환"""
        return {
            "pub_count": self.redis_pub_count,
            "sub_count": self.redis_sub_count,
            "error_count": self.redis_errors
        }
    
    def _generate_report(self):
        """성능 보고서 생성"""
        try:
            recent_metrics = self.get_recent_metrics()
            vllm_metrics = self.get_vllm_metrics()
            redis_metrics = self.get_redis_metrics()
            
            logger.info("=== Performance Report ===")
            logger.info(f"Global Requests: {self.global_metrics.request_count}, "
                       f"Avg Time: {self.global_metrics.avg_processing_time:.3f}s, "
                       f"Error Rate: {self.global_metrics.error_rate:.2%}")
            
            logger.info(f"Recent (Last {len(self.recent_times)}): "
                       f"Avg Time: {recent_metrics['avg_processing_time']:.3f}s, "
                       f"P95: {recent_metrics['p95_processing_time']:.3f}s, "
                       f"Error Rate: {recent_metrics['error_rate']:.2%}")
            
            logger.info(f"vLLM: Concurrent: {vllm_metrics['concurrent_requests']}, "
                       f"Queue: {vllm_metrics['queue_size']}, "
                       f"Avg Quality: {vllm_metrics['avg_quality_score']:.2f}")
            
            logger.info(f"Redis: Pub: {redis_metrics['pub_count']}, "
                       f"Sub: {redis_metrics['sub_count']}, "
                       f"Errors: {redis_metrics['error_count']}")
            
            # 서비스별 상위 메트릭
            for service, metrics in list(self.service_metrics.items())[:5]:
                logger.info(f"Service '{service}': {metrics.request_count} reqs, "
                           f"{metrics.avg_processing_time:.3f}s avg, "
                           f"{metrics.error_rate:.2%} errors")
            
        except Exception as e:
            logger.exception(f"Error generating performance report: {e}")


# 전역 성능 모니터 인스턴스
performance_monitor = PerformanceMonitor()
