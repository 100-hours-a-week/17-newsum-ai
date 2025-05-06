# app/tools/analysis/text_clusterer.py
from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter, defaultdict

# scikit-learn 동적 임포트
SKLEARN_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import MiniBatchKMeans
    # 한국어 처리 필요 시 konlpy 등과 함께 사용 고려
    SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None # type: ignore
    MiniBatchKMeans = None # type: ignore

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

if not SKLEARN_AVAILABLE:
    logger.warning("scikit-learn 라이브러리가 설치되지 않았습니다. TextClusteringTool 기능이 비활성화됩니다.")

class TextClusteringTool:
    """
    TF-IDF와 MiniBatchKMeans를 사용하여 텍스트 목록을 클러스터링하고,
    각 클러스터의 대표 텍스트를 식별하는 도구입니다.
    """

    def __init__(self):
        """TextClusteringTool 초기화."""
        self.vectorizer = None
        self.kmeans_params = {}

        if SKLEARN_AVAILABLE:
            try:
                logger.info("TF-IDF Vectorizer 초기화 중...")
                # TF-IDF 벡터라이저 설정 (settings.py 값 사용)
                self.vectorizer = TfidfVectorizer(
                    max_features=settings.TFIDF_MAX_FEATURES,
                    stop_words=settings.TFIDF_STOP_WORDS if settings.TFIDF_STOP_WORDS else None, # None 또는 'english' 등
                    min_df=settings.TFIDF_MIN_DF,
                    max_df=settings.TFIDF_MAX_DF,
                    ngram_range=(settings.TFIDF_NGRAM_RANGE_MIN, settings.TFIDF_NGRAM_RANGE_MAX)
                )
                logger.info("TF-IDF Vectorizer 초기화 완료.")

                # KMeans 파라미터 설정
                self.kmeans_params = {
                    # n_clusters는 데이터 기반 동적 결정
                    'init': 'k-means++',
                    'batch_size': 1024, # 데이터 크기에 따라 조정 가능
                    'n_init': settings.KMEANS_N_INIT, # 안정성을 위한 다중 초기화
                    'max_no_improvement': settings.KMEANS_MAX_NO_IMPROVEMENT, # 조기 종료 조건
                    'reassignment_ratio': 0.05, # 지역 최적해 탈출 도움
                    'random_state': 42 # 재현성을 위한 랜덤 시드
                }
                logger.info("KMeans 파라미터 설정 완료.")

            except Exception as e:
                logger.error(f"scikit-learn 컴포넌트 초기화 실패: {e}", exc_info=True)
                self.vectorizer = None # 실패 시 None으로 설정
        else:
             logger.warning("scikit-learn 사용 불가로 클러스터링 기능 비활성화.")


    def cluster_texts(
        self,
        opinions: List[Dict[str, Any]], # 클러스터링할 원본 의견 데이터 목록
        trace_id: Optional[str] = None,
        comic_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        주어진 의견 목록을 클러스터링하고 각 의견에 cluster_id와 is_representative 플래그를 추가합니다.

        Args:
            opinions (List[Dict[str, Any]]): 클러스터링할 의견 사전 목록 (최소 'text' 키 포함).
            trace_id (Optional[str]): 로깅용 추적 ID.
            comic_id (Optional[str]): 로깅용 코믹 ID.

        Returns:
            List[Dict[str, Any]]: 입력 목록에 'cluster_id'와 'is_representative'가 추가된 목록.
                                   클러스터링 실패 시 cluster_id=0, is_representative=True가 할당됨.
        """
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}

        # 라이브러리 또는 벡터라이저 사용 불가 시 클러스터링 건너뛰기
        if not SKLEARN_AVAILABLE or not self.vectorizer:
            logger.warning("클러스터링 건너뛰기: scikit-learn 사용 불가 또는 벡터라이저 초기화 실패.", extra=extra_log_data)
            # 기본값 할당 후 반환
            for op in opinions: op.update({'cluster_id': 0, 'is_representative': True})
            return opinions

        num_opinions = len(opinions)
        min_samples = settings.KMEANS_MIN_SAMPLES # 최소 샘플 수 설정값
        # 샘플 수 부족 시 클러스터링 건너뛰기
        if num_opinions < min_samples:
            logger.info(f"클러스터링 건너뛰기: 샘플 수 부족 ({num_opinions} < {min_samples}). 모든 의견을 대표로 지정.", extra=extra_log_data)
            for op in opinions: op.update({'cluster_id': 0, 'is_representative': True})
            return opinions

        # 클러스터링 대상 텍스트 추출
        texts = [op.get('text', '') for op in opinions]
        if not any(texts): # 유효한 텍스트 없으면 건너뛰기
             logger.warning("클러스터링 위한 유효 텍스트 없음.", extra=extra_log_data)
             for op in opinions: op.update({'cluster_id': 0, 'is_representative': True})
             return opinions

        try:
            logger.info(f"{num_opinions}개 의견에 대한 TF-IDF 벡터화 시작...", extra=extra_log_data)
            # TF-IDF 벡터화 수행 (CPU-bound, 필요 시 executor 사용 고려)
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            logger.info(f"벡터화 완료. 행렬 크기: {tfidf_matrix.shape}", extra=extra_log_data)

            # 클러스터 수 결정 (데이터 기반 동적 결정, 설정값 범위 내)
            max_clusters = settings.KMEANS_DEFAULT_CLUSTERS # 설정에서 최대 클러스터 수 가져오기
            # 휴리스틱: 최소 2개, 최대 설정값, 샘플당 최소 5개 의견 가정
            n_clusters = min(max_clusters, max(2, num_opinions // 5))
            logger.info(f"MiniBatchKMeans 클러스터링 시작 (클러스터 수: {n_clusters})...", extra=extra_log_data)

            # MiniBatchKMeans 모델 생성 및 학습/예측
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, **self.kmeans_params)
            # fit_predict는 CPU-bound, 필요 시 executor 사용 고려
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            centroids = kmeans.cluster_centers_ # 각 클러스터의 중심점
            logger.info(f"클러스터링 완료. {len(np.unique(cluster_labels))}개의 클러스터 발견.", extra=extra_log_data)

            # --- 클러스터 ID 할당 및 대표 의견 찾기 ---
            cluster_indices = defaultdict(list) # 클러스터별 원본 인덱스 저장
            for i, label in enumerate(cluster_labels):
                cluster_indices[label].append(i)

            # 모든 의견의 is_representative 초기화
            for i in range(num_opinions): opinions[i]['is_representative'] = False

            representatives_found = 0 # 찾은 대표 의견 수
            # 각 클러스터 순회
            for cluster_id, indices in cluster_indices.items():
                if not indices: continue # 빈 클러스터 건너뛰기

                cluster_vectors = tfidf_matrix[indices] # 현재 클러스터의 벡터들
                centroid = centroids[cluster_id] # 현재 클러스터의 중심점

                # 중심점과의 유클리드 거리 계산
                distances = np.linalg.norm(cluster_vectors.toarray() - centroid, axis=1)

                # 가장 가까운 벡터의 인덱스 찾기 (클러스터 내 인덱스)
                min_dist_idx_in_cluster = np.argmin(distances)
                # 원본 opinions 리스트에서의 인덱스
                original_opinion_idx = indices[min_dist_idx_in_cluster]

                # 대표 의견으로 표시하고 클러스터 ID 할당
                opinions[original_opinion_idx]['is_representative'] = True
                opinions[original_opinion_idx]['cluster_id'] = int(cluster_id)
                representatives_found += 1

                # 클러스터 내 다른 의견들에게도 클러스터 ID 할당
                for idx_in_cluster, original_idx in enumerate(indices):
                    # 대표 의견이 아닌 경우에도 클러스터 ID 할당
                    if idx_in_cluster != min_dist_idx_in_cluster:
                        opinions[original_idx]['cluster_id'] = int(cluster_id)

            logger.info(f"{representatives_found}개의 대표 의견을 선정했습니다.", extra=extra_log_data)
            return opinions # 클러스터 정보 추가된 의견 목록 반환

        except ValueError as ve:
            # 주로 TF-IDF 어휘 사전이 비거나 행렬 생성 실패 시 발생
            logger.error(f"클러스터링 중 ValueError 발생 (TF-IDF 문제 가능성): {ve}. 클러스터링 건너<0xEB><0x9C><0x95>뛰기.", exc_info=True, extra=extra_log_data)
        except Exception as e:
            # 기타 클러스터링 오류
            logger.error(f"의견 클러스터링 중 오류 발생: {e}. 클러스터링 건너<0xEB><0x9C><0x95>뛰기.", exc_info=True, extra=extra_log_data)

        # 클러스터링 실패 시 기본값 할당 후 반환
        for op in opinions: op.update({'cluster_id': 0, 'is_representative': True})
        return opinions