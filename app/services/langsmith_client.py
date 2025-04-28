# app/services/langsmith_client.py

# LangSmith 연동 예시 (옵션)
from langgraph.tracing.langsmith import LangSmithTracer

def get_langsmith_tracer(project_name: str) -> LangSmithTracer:
    """
    LangSmith 트래커를 생성하는 함수.
    """
    return LangSmithTracer(
        project_name=project_name,
    )
