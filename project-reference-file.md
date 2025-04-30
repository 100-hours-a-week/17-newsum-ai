# Newsom AI 서비스 프로젝트 레퍼런스 파일

## 프로젝트 개요
- **프로젝트명**: Newsom AI 서비스
- **목적**: LangGraph와 FastAPI를 사용하여 뉴스 기사로부터 4컷 만화를 생성하는 멀티스텝 AI 워크플로우
- **대화 ID**: 현재 Claude 대화 내의 파일이며, 다른 대화에서도 참조 가능

## 프로젝트 구조
```
project_root/
│
├── app/                          # 메인 애플리케이션 코드
│   ├── agents/                   # 각 작업을 수행하는 LangGraph 에이전트
│   ├── tools/                    # 에이전트가 사용하는 독립 기능 모듈
│   ├── workflows/                # LangGraph 기반 워크플로우 그래프, 상태, 전이 정의
│   ├── services/                 # 외부 API(LLM, 이미지, 스토리지, LangSmith) 통신 모듈
│   ├── api/                      # FastAPI HTTP 엔드포인트 및 스키마
│   ├── config/                   # 환경설정 및 상수 관리
│   └── utils/                    # 로거, 재시도, 타이머 등의 유틸리티 모듈
│
├── pre-test/                     # 개발 초기 단계에서의 자료 정리 및 기술 검증용 코드 저장소
│
├── main.py                       # FastAPI 서버 엔트리포인트
├── requirements.txt              # Python 패키지 의존성 목록
├── .env.sample                   # 환경설정 샘플 템플릿
├── README.md                     # 프로젝트 설명 및 아키텍처 문서
└── .gitignore                    # Git 무시 파일 설정
```

## 핵심 컴포넌트

### 1. 워크플로우 상태 (ComicState)
```python
class ComicState(BaseModel):
    """워크플로우 실행 중 전달되는 상태 정보 정의"""
    initial_query: Optional[str] = Field(default=None)
    search_results: Optional[List[Dict[str, str]]] = Field(default=None)
    news_urls: Optional[List[str]] = Field(default_factory=list)
    selected_url: Optional[str] = Field(default=None)
    articles: Optional[List[str]] = Field(default_factory=list)
    summaries: Optional[List[str]] = Field(default_factory=list)
    final_summary: Optional[str] = Field(default=None)
    additional_context: Optional[Dict[str, Any]] = Field(default=None)
    humor_texts: Optional[List[str]] = Field(default_factory=list)
    scenarios: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    image_urls: Optional[List[str]] = Field(default_factory=list)
    final_comic_url: Optional[str] = None
    translated_texts: Optional[List[str]] = Field(default_factory=list)
    error_message: Optional[str] = None
```

### 2. 메인 워크플로우
```python
def build_main_workflow() -> StateGraph:
    """뉴스 수집부터 만화 완성까지 LangGraph 기반 플로우를 구성합니다."""
    graph = StateGraph(ComicState)

    # 노드 등록
    graph.add_node("collect", collect_news)
    graph.add_node("scrape", ScraperAgent().run)
    graph.add_node("summarize_individual", IndividualSummarizerAgent().run)
    graph.add_node("summarize_synthesis", SynthesisSummarizerAgent().run)
    graph.add_node("analyze_content", ContentSummarizerAgent().run)
    graph.add_node("humor", HumoratorAgent().run)
    # graph.add_node("scenario", ScenarioWriterAgent().run)
    # graph.add_node("image", ImagerAgent().run)
    # graph.add_node("postprocess", PostProcessorAgent().run)
    # graph.add_node("translate", TranslatorAgent().run)

    # 조건부 라우팅 설정 (옵션)
    def should_continue(state):
        """에러 상태 확인하여 워크플로우 계속 진행 여부 결정"""
        if state.get("error_message"):
            return "end_with_error"
        return "continue"

    # 흐름 연결
    graph.set_entry_point("collect")
    graph.add_edge("collect", "scrape")
    graph.add_edge("scrape", "summarize_individual")
    graph.add_edge("summarize_individual", "summarize_synthesis")
    graph.add_edge("summarize_synthesis", "analyze_content")
    graph.add_edge("analyze_content", "humor")
    graph.add_edge("humor", END)  # 현재는 유머 생성까지 구현

    return graph.compile()
```

### 3. API 엔드포인트
```python
# FastAPI 비동기 만화 생성 요청
@router.post(
    "/comics",
    response_model=AsyncComicResponse,
    status_code=202
)
async def request_comic_generation(
    request: AsyncComicRequest = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
)

# 만화 생성 상태 스트리밍 구독
@router.post(
    "/comics/{comic_id}/stream",
    summary="Subscribe to Comic Generation Status Updates"
)
async def stream_comic_status(
    comic_id: str = Path(...),
)
```

### 4. 백그라운드 작업 트리거
```python
async def trigger_workflow_task(query: str, background_tasks: BackgroundTasks) -> str:
    """백그라운드로 LangGraph 워크플로우 실행을 트리거하고 DB 상태를 업데이트하는 함수."""
    comic_id = str(uuid.uuid4())
    
    async def workflow_runner(comic_id: str, initial_query: str):
        graph = build_main_workflow()
        initial_state_obj = ComicState(initial_query=initial_query)
        initial_state_dict = initial_state_obj.model_dump()
        
        try:
            await db_client.set(comic_id, {"status": "STARTED", "message": "Workflow started.", "query": initial_query})
            final_state_dict = await graph.ainvoke(initial_state_dict)
            
            final_status = "DONE" if not final_state_dict.get("error_message") else "FAILED"
            final_message = final_state_dict.get("error_message", "Workflow completed successfully.")
            result_data = {"final_comic_url": final_state_dict.get("final_comic_url")}
            
            await db_client.set(comic_id, {"status": final_status, "message": final_message, "result": result_data})
            
        except Exception as e:
            await db_client.set(comic_id, {"status": "FAILED", "message": f"Workflow execution error: {str(e)}"})
    
    background_tasks.add_task(workflow_runner, comic_id, query)
    await db_client.set(comic_id, {"status": "PENDING", "message": "Workflow task accepted."})
    
    return comic_id
```

## 구현된 에이전트 목록

1. **Collector Agent** (`collector_agent.py`)
   - 기능: 검색 엔진 API를 활용하여 뉴스 기사 URL 수집
   - 특징: 다중 검색 엔진 지원, 한국 키워드 인식, 폴백 메커니즘

2. **Scraper Agent** (`scraper_agent.py`)
   - 기능: URL 리스트를 받아 각 URL의 본문 텍스트를 비동기적으로 스크래핑
   - 특징: 병렬 처리, 다양한 웹사이트 구조 처리

3. **Individual Summarizer Agent** (`individual_summarizer_agent.py`)
   - 기능: 개별 기사 텍스트를 요약
   - 특징: LLM을 활용한 비동기 요약 처리

4. **Synthesis Summarizer Agent** (`synthesis_summarizer_agent.py`)
   - 기능: 개별 요약을 하나의 종합 요약으로 통합
   - 특징: 다중 소스의 정보 통합 능력

5. **Content Summarizer Agent** (`content_summarizer_agent.py`)
   - 기능: YouTube 비디오 및 댓글을 통해 추가 컨텍스트 수집
   - 특징: 다차원 소스에서 추가 관점 수집

6. **Humorator Agent** (`humorator_agent.py`)
   - 기능: 뉴스에서 유머러스한 관점 추출
   - 특징: 공감적 유머 생성, 컨텍스트 이해 기반 접근
   - **수정됨**: 응답 파싱 로직 개선, 정규식 패턴 인식, 폴백 메커니즘 추가

## 주요 서비스

1. **Database Client** (`database_client.py`)
   - 기능: Redis 기반 상태 저장 및 관리
   - 메서드: `set()`, `get()`

2. **LLM Server Client** (`llm_server_client.py`)
   - 기능: LLM API 호출 및 응답 처리
   - 메서드: `call_llm_api()`

3. **Search Tools**
   - 기능: 다양한 검색 엔진(Google, Naver, Kakao, Tavily) API 연동
   - 구현: `app/tools/search/` 디렉토리 내 각 파일

## 미구현/개발 중인 컴포넌트

1. **Scenario Writer Agent** - 4컷 만화 시나리오 작성
2. **Imager Agent** - 이미지 생성 API를 통한 만화 이미지 생성
3. **Post Processor Agent** - 최종 만화 형태로 조합 및 레이아웃 처리
4. **Translator Agent** - 다국어 지원을 위한 텍스트 번역

## 기술 스택

- **FastAPI**: 비동기 API 서버
- **LangGraph**: 워크플로우 오케스트레이션
- **Redis**: 상태 관리
- **httpx**: 비동기 HTTP 요청
- **BeautifulSoup**: 웹 스크래핑
- **Pydantic**: 데이터 검증

## 프로젝트 상태

현재 워크플로우는 뉴스 수집, 스크래핑, 요약, 컨텍스트 분석, 유머 추출까지 구현되어 있습니다. 시나리오 작성, 이미지 생성, 후처리, 번역 단계는 계획되어 있으나 아직 구현되지 않았습니다.

## 최근 수정사항 (2025-04-29)

### 1. 테스트 코드 개선
- **test_workflow.py**: SentimentAnalyzerAgent → ContentSummarizerAgent 참조 변경
- **run_tests.py**: 비동기 이벤트 루프 관련 DeprecationWarning 해결
- **테스트 견고성**: 모킹 객체와 검증 로직 일치화

### 2. 에러 처리 메커니즘 강화
- **transitions.py**: 에러 상태 감지 및 워크플로우 종료 로직 추가
- **main_workflow.py**: 조건부 라우팅 기능 추가 (에러 상태 처리)
- **각 에이전트**: 에러 상태 명확화 및 일관된 에러 전파

### 3. LLM 응답 파싱 개선
- **humorator_agent.py**: 향상된 응답 파싱 로직 구현
  - 정규식 패턴 인식 추가
  - 폴백 메커니즘으로 파싱 실패 시 내용 추출
  - 최종 검증 단계를 통한 결과 품질 향상

## 다음 단계 작업

1. **시나리오 작성 에이전트 완성**: 유머 포인트를 기반으로 4컷 만화 시나리오 작성
2. **이미지 생성 에이전트 구현**: 외부 이미지 생성 API 연동
3. **후처리 에이전트 개발**: 이미지와 텍스트 조합, 레이아웃 최적화
4. **번역 기능 추가**: 다국어 지원
5. **오류 처리 강화**: 각 단계별 예외 상황 대응 및 복구 메커니즘 계속 개선
6. **모니터링 시스템 구축**: LangSmith 연동 및 성능 추적
7. **API 보안 강화**: 민감한 인증 정보 관리 개선 (환경 변수 활용)
8. **통합 테스트 구축**: 실제 API 호출 없이 테스트 가능한 모의 서비스 구현

## 작업 흐름도

```
[사용자 쿼리] → [뉴스 수집] → [스크래핑] → [개별 요약] → [종합 요약] 
                  → [컨텍스트 분석] → [유머 추출] → [시나리오 작성*] 
                  → [이미지 생성*] → [후처리*] → [번역*] → [최종 만화]

* = 미구현 단계
```

## 참고 문서

- FastAPI 공식 문서: https://fastapi.tiangolo.com/
- LangGraph 공식 문서: https://python.langchain.com/docs/langgraph/
- Pydantic 공식 문서: https://docs.pydantic.dev/

---

이 레퍼런스 파일은 다른 Claude 대화에서 Newsom AI 프로젝트 개발을 이어나갈 때 참조할 수 있습니다. 새 대화에서 이 파일을 업로드하고 "이전에 분석한 Newsom AI 프로젝트를 계속 개발하고 싶습니다"라고 요청하시면 됩니다.
