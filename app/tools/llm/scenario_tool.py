# app/tools/llm/scenario_tool.py

async def generate_scenario(summary: str, humor: str) -> list[str]:
    """
    요약과 유머를 기반으로 4컷 만화 시나리오를 생성하는 함수.
    """
    # TODO: LLM 서버 호출하여 컷별 시나리오 생성
    return [
        "1컷: 장면 설명",
        "2컷: 장면 설명",
        "3컷: 장면 설명",
        "4컷: 장면 설명",
    ]
