# ai/app/nodes_v2/n05_generate_and_finalize_report_node.py

from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import jinja2
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger
from app.services.llm_service import LLMService

logger = get_logger(__name__)
NODE_ORDER = 5
MAIN_USE_THRESHOLD = 3
SECONDARY_USE_THRESHOLD = 2
EMB_MODEL = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# 프로젝트 루트 경로 설정 (이 파일 기준 상위 3단계)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_BASE_DIR = PROJECT_ROOT / "results"


class N05GenerateAndFinalizeReportNode:
    """
    N05GenerateAndFinalizeReportNode

    • 목적:
      - N04에서 수집된 raw_search_results를 '전문적인 보고서' 관점으로 필터링
      - 필터링된 결과를 바탕으로 4컷 만평 제작용 Fact Brief 보고서 HTML 생성 및 저장

    • 생성자 인자:
      - llm_service: LLMService 인스턴스 (fact-check 용도로 사용)
      - results_base_dir: 보고서 저장 기본 디렉토리 (없으면 PROJECT_ROOT/results 사용)
    """

    def __init__(self, llm_service: LLMService, results_base_dir: Optional[Path] = None):
        self.llm_service = llm_service
        # Jinja2 환경: 템플릿 로더로 함수 로더 사용 (템플릿 내용은 _load_template에서 정의)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FunctionLoader(self._load_template),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        self.template_name = "deep_research_report_template.jinja2"
        # 저장 경로 설정: 주입된 results_base_dir 또는 기본값
        self.results_base_dir = results_base_dir or DEFAULT_RESULTS_BASE_DIR
        logger.info(f"N05Initialize: results_base_dir={self.results_base_dir}", extra={"node": "N05", "order": NODE_ORDER})

    # ─────── 1) 전문성 중심 필터링 ──────────────────────────────────────────

    async def _filter_report_sources(
        self,
        raw_results: List[Dict[str, Any]],
        issue_summary: str,
        key_aspects: List[str],
        satire_targets: List[str] | None,
        work_id: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        raw_results를 순회하며 '전문성' 필터링을 수행하여
        - MAIN_USE_THRESHOLD 이상: main_sources
        - SECONDARY_USE_THRESHOLD 이상: secondary_sources
        그 외: 제외

        점수 산정:
          ① 주제 유사도 (문장 임베딩 코사인 유사도 ≥ 0.35 → +1)
          ② URL 또는 제목/스니펫에 '보고서', '백서' 등 키워드 → +1
          ③ 스니펫 길이 ≥ 120자 또는 transcript 길이 ≥ 300자 → +1
          ④ 스니펫에 날짜/숫자/통계 지표 → +1
          ⑤ satire_targets 키워드가 본문에 포함 → +1
          ⑥ 제목 중복 시 감점 -1
          ⑦ score == 2인 경우 LLM fact-check 수행(성공: +1, 실패: -1)
        """
        def _embed(text: str):
            return EMB_MODEL.encode(text, convert_to_tensor=True)

        ref_texts = [issue_summary] + (key_aspects or [])
        ref_vecs = [_embed(t) for t in ref_texts if t]

        def sim_to_refs(text: str) -> float:
            if not ref_vecs:
                return 0.0
            v = _embed(text)
            return float(max(util.cos_sim(v, rv) for rv in ref_vecs)[0][0])

        PDF_PATTERN = re.compile(r'\.pdf$', re.I)
        REPORT_KW = re.compile(
            r'(보고서|백서|white\s*paper|policy\s*paper|statistical|annual\s*report|연구\s*결과|통계|조사)',
            re.I
        )
        FACT_REGEX = re.compile(r'\d{4}[-년\.]\d{1,2}|[0-9]{3,}[ ]?(억|만|조|%|\bpeople\b|\btons\b)', re.I)

        main_use: List[Dict[str, Any]] = []
        secondary_use: List[Dict[str, Any]] = []
        seen_titles: list[str] = []

        async def llm_fact_check(snippet_text: str) -> bool:
            """
            LLM을 이용해 주어진 스니펫이 '전문 보고서 수준의 사실 진술'인지 Yes/No로 반환.
            """
            prompt = (
                "Answer with a single word 'Yes' or 'No'.\n"
                "Does the following sentence look like an objective, factual statement "
                "found in an official or expert report?\n\n"
                f"\"{snippet_text.strip()[:250]}\""
            )
            rsp = await self.llm_service.generate_text(
                messages=[{"role": "user", "content": prompt}],
                request_id=f"{work_id}_factchk",
                max_tokens=3,
                temperature=0.0
            )
            return rsp.get("generated_text", "").strip().lower().startswith("yes")

        for item in raw_results:
            # None 안전화: snippet 혹은 transcript가 None이면 빈 문자열로 대체
            title = item.get("title", "") or ""
            snippet = item.get("snippet") or ""
            transcript = item.get("transcript") or ""
            body_for_sim = f"{title} {snippet}"[:600]

            # ① 주제 유사도
            topic_sim = sim_to_refs(body_for_sim)
            score = 1 if topic_sim >= 0.35 else 0

            # ② '전문 문건' 신호: URL이 PDF이거나, 제목/스니펫에 보고서 관련 키워드 포함
            url = item.get("url", "") or ""
            if PDF_PATTERN.search(url) or REPORT_KW.search(title) or REPORT_KW.search(snippet):
                score += 1

            # ③ 정보 밀도: 스니펫 길이 또는 transcript 길이
            if len(snippet) >= 120 or len(transcript) >= 300:
                score += 1

            # ④ 팩트 수치·날짜 존재
            if FACT_REGEX.search(snippet):
                score += 1

            # ⑤ 풍자 대상 키워드 포함 → 보조 가점
            if satire_targets and any(t.lower() in body_for_sim.lower() for t in satire_targets):
                score += 1

            # ⑥ 제목 중복 감점
            if any(fuzz.QRatio(title, existing) > 80 for existing in seen_titles):
                score -= 1
            else:
                seen_titles.append(title)

            # ⑦ LLM fact-check: score == 2일 때만 실행
            if score == 2:
                try:
                    fact_ok = await llm_fact_check(snippet)
                    score += 1 if fact_ok else -1
                    item["quality_llm_checked"] = True
                except Exception as e:
                    logger.warning(f"[N05] LLM fact-check error: {e}", extra={"work_id": work_id})
                    item["quality_llm_checked"] = False

            # 점수 및 사유 기록
            item["quality_score"] = score
            item["quality_reason"] = f"sim:{topic_sim:.2f};len:{len(snippet)};score:{score}"

            # 분류
            if score >= MAIN_USE_THRESHOLD:
                main_use.append(item)
            elif score >= SECONDARY_USE_THRESHOLD:
                secondary_use.append(item)

        return main_use, secondary_use

    # ─────── 2) Jinja2 템플릿 로딩 ────────────────────────────────────────

    def _load_template(self, template_name: str) -> Optional[str]:
        """
        Jinja2 FunctionLoader를 위한 콜백. 템플릿 이름에 해당하는 스트링을 반환.
        이 예시에서는 실제 파일 로딩 대신 인라인으로 작성합니다.
        """
        if template_name != self.template_name:
            return None

        # 간단한 HTML 템플릿 예시
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <title>Satirical Comic Fact Brief</title>
            <style>
                body { font-family: Arial, Helvetica, sans-serif; padding: 20px; }
                .sec { margin-bottom: 2em; }
                .card { border: 1px solid #ddd; padding: 1em; margin-bottom: .5em; }
            </style>
        </head>
        <body>
            <h1>📑 Satirical Comic Fact Brief</h1>
            <p><strong>Issue:</strong> {{ issue_summary }}</p>
            <p><strong>Satire Target(s):</strong> {{ satire_targets | join(', ') }}</p>
            <p><strong>Tone Suggestion:</strong> {{ tone_suggestion }}</p>

            <div class="sec">
                <h2>✅ Core Fact Sources</h2>
                {% for src in main_sources %}
                    <div class="card">
                        <h3>{{ src.title }}</h3>
                        <p>{{ src.snippet }}</p>
                        <p>
                            <small>
                            <a href="{{ src.url }}" target="_blank">{{ src.source_domain }}</a>
                             • quality {{ src.quality_score }}
                            </small>
                        </p>
                    </div>
                {% endfor %}
            </div>

            {% if secondary_sources %}
            <div class="sec">
                <h2>🧐 Additional References (Need Cross-Check)</h2>
                <ul>
                {% for s in secondary_sources %}
                    <li><a href="{{ s.url }}" target="_blank">{{ s.title }}</a> ({{ s.quality_score }})</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
        </body>
        </html>
        """

    # ─────── 3) HTML 렌더링 ────────────────────────────────────────────

    def _render_html(
        self,
        main_sources: List[Dict[str, Any]],
        secondary_sources: List[Dict[str, Any]],
        issue_summary: str,
        satire_targets: List[str],
        tone_suggestion: str
    ) -> str:
        """
        Jinja2 템플릿을 이용해 HTML 문자열 생성.
        """
        template = self.jinja_env.get_template(self.template_name)
        return template.render(
            issue_summary=issue_summary,
            satire_targets=satire_targets or [],
            tone_suggestion=tone_suggestion,
            main_sources=main_sources,
            secondary_sources=secondary_sources
        )

    # ─────── 4) 보고서 저장 ────────────────────────────────────────────

    def _save_report_to_local_fs(self, report_html_content: str, work_id: str, extra_log: Dict[str, Any]) -> Optional[str]:
        """
        생성된 HTML 보고서를 로컬 파일 시스템에 저장합니다.
        - PROJECT_ROOT/results/<work_id>/generated_report.html 경로로 저장
        """
        if not work_id:
            logger.error("Work ID가 누락되어 보고서를 저장할 수 없습니다.", extra=extra_log)
            return None
        if not report_html_content or not isinstance(report_html_content, str):
            logger.warning("저장할 보고서 내용이 없거나 문자열이 아닙니다.", extra=extra_log)
            return None

        try:
            # 1) work_id 디렉토리 생성
            report_dir = self.results_base_dir / work_id
            report_dir.mkdir(parents=True, exist_ok=True)

            # 2) 파일 경로 설정
            report_file_path = report_dir / "generated_report.html"

            # 3) 파일 쓰기
            with open(report_file_path, "w", encoding="utf-8") as f:
                f.write(report_html_content)

            saved_path_str = str(report_file_path.resolve())
            logger.info(f"보고서가 로컬에 성공적으로 저장되었습니다: {saved_path_str}", extra=extra_log)
            return saved_path_str

        except Exception as e:
            logger.error(f"보고서 로컬 저장 중 오류 발생: {e}", extra={**extra_log, "exception": str(e)})
            return None

    # ─────── 5) 노드 run() ────────────────────────────────────────────

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta_sec = state.meta
        search_sec = state.search
        query_ctx = state.query.query_context

        work_id = meta_sec.work_id
        extra_log = {"work_id": work_id, "node": "N05", "order": NODE_ORDER}

        meta_sec.workflow_status[NODE_ORDER] = "PROCESSING"
        logger.info("N05: 시작 – 결과 필터링 및 보고서 생성", extra=extra_log)

        raw_results = search_sec.raw_search_results or []
        if not raw_results:
            logger.warning("N05: raw_search_results가 비어 있음 – 보고서 생성 스킵", extra=extra_log)
            meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"
            return {"meta": meta_sec.model_dump()}

        issue_summary = query_ctx.get("issue_summary_for_comic", "")
        key_aspects = query_ctx.get("key_aspects_to_search", [])
        satire_targets = query_ctx.get("satire_target", [])
        tone_suggestion = query_ctx.get("tone_suggestion", "")

        # 1) 필터링: 전문성 기준으로 main/secondary 분류
        main_sources, secondary_sources = await self._filter_report_sources(
            raw_results,
            issue_summary,
            key_aspects,
            satire_targets,
            work_id
        )

        logger.info(
            f"N05: 필터 결과 – main: {len(main_sources)}, secondary: {len(secondary_sources)}",
            extra=extra_log
        )

        # 2) HTML 렌더링 (Jinja2 템플릿 사용)
        html_report = self._render_html(
            main_sources, secondary_sources, issue_summary, satire_targets, tone_suggestion
        )

        # 3) 로컬 저장 (PROJECT_ROOT/results/<work_id>/generated_report.html)
        saved_path = self._save_report_to_local_fs(html_report, work_id, extra_log)
        if not saved_path:
            logger.error("N05: 보고서 저장에 실패했습니다.", extra=extra_log)

        # 4) state 업데이트
        #    - raw_search_results에 main + secondary를 덮어써서 이후 노드 또는 리턴 시 활용
        search_sec.raw_search_results = main_sources + secondary_sources
        state.report.report_content = html_report
        state.report.saved_report_path = saved_path
        state.report.contextual_summary = issue_summary

        meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"
        logger.info("N05: 완료 – 다음 노드로 이동", extra=extra_log)

        return {
            "meta": meta_sec.model_dump(),
            "search": search_sec.model_dump(),
            "report": state.report.model_dump()
        }
