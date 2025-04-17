# LangGraph + FastAPI 기반 뉴스 시나리오 생성 예제 (MVP Mock 버전)

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()


# 1. FastAPI 초기화
app = FastAPI()

# 2. 입력/출력 모델 정의
class NewsInput(BaseModel):
    title: str = Field(..., example="삼성전자, 3nm GAA 공정 기반 칩 양산 개시")
    content: str = Field(..., example="삼성전자가 세계 최초로 3nm GAA 공정을 도입해 양산에 돌입했다. 기존 FinFET 구조 대비 전력 효율과 성능이 크게 향상되었으며, 주요 고객은 아직 공개되지 않았다.")

class CutItem(BaseModel):
    content: str
    image_url: str

class ScenarioFullOutput(BaseModel):
    title: str
    description: str
    keyword_tags: list[str]
    created_at: str
    thumbnail_url: str
    cuts: list[CutItem]

# 3. LLM 초기화 (OpenAI GPT)
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# 4. LangGraph 노드 정의

def create_scenario_node():
    def scenario_fn(state):
        title = state["title"]
        content = state["content"]

        prompt = f"""
        다음 뉴스 내용을 읽고, 사실을 기반으로 4컷 만화 시나리오를 만들어줘.

        ---
        [뉴스 제목]: {title}
        [뉴스 내용]: {content}
        ---

        시나리오는 다음 형식으로 출력해줘:
        컷 1: (장면)
        컷 2: (장면)
        컷 3: (장면)
        컷 4: (장면)

        각 컷은 1문장씩, 시각적으로 묘사 가능한 형태로 만들어줘.
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content

        cuts = {f"cut_{i+1}": "" for i in range(4)}
        for line in text.splitlines():
            if line.startswith("컷"):
                try:
                    cut_num = int(line[2])
                    cuts[f"cut_{cut_num}"] = line.split(":", 1)[-1].strip()
                except:
                    pass

        return {**state, **cuts}

    return scenario_fn

def create_summary_node():
    def summary_fn(state):
        cuts_text = "\n".join([state.get(f"cut_{i+1}", "") for i in range(4)])

        prompt = f"""
        다음 4컷 만화 시나리오를 요약해서 전체 만화 제목과 설명을 각각 만들어줘.

        ---
        {cuts_text}
        ---

        출력 형식:
        제목: ...
        설명: ...
        키워드: [단어1, 단어2, 단어3]
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content

        title, description, keywords = "", "", []
        for line in text.splitlines():
            if line.startswith("제목"):
                title = line.split(":", 1)[-1].strip()
            elif line.startswith("설명"):
                description = line.split(":", 1)[-1].strip()
            elif line.startswith("키워드"):
                keyword_part = line.split(":", 1)[-1].strip()
                keywords = [k.strip().strip("'") for k in keyword_part.strip("[]").split(",") if k.strip()]

        return {
            **state,
            "comic_title": title,
            "comic_description": description,
            "comic_keywords": keywords
        }

    return summary_fn

def create_thumbnail_node():
    def thumbnail_fn(state):
        desc = state.get("comic_description", "")
        prompt = f"다음 설명을 바탕으로 전체 만화를 대표할 수 있는 장면 하나를 시각적으로 묘사해줘: {desc}"
        return {
            **state,
            "thumbnail_url": "https://dummy.local/thumbnail.png"
        }

    return thumbnail_fn

# 5. LangGraph 구성
builder = StateGraph()
builder.add_node("scenario", create_scenario_node())
builder.add_node("summary", create_summary_node())
builder.add_node("thumbnail", create_thumbnail_node())

builder.set_entry_point("scenario")
builder.add_edge("scenario", "summary")
builder.add_edge("summary", "thumbnail")
builder.set_finish_point("thumbnail")

graph = builder.compile()

# 6. FastAPI POST endpoint 정의
@app.post("/generate_scenario_full", response_model=ScenarioFullOutput)
async def generate_scenario_full(input_data: NewsInput):
    result = graph.invoke({
        "title": input_data.title,
        "content": input_data.content
    })

    cuts = []
    for i in range(4):
        cuts.append(CutItem(
            content=result.get(f"cut_{i+1}", ""),
            image_url=f"https://dummy.local/cut_{i+1}.png"
        ))

    return ScenarioFullOutput(
        title=result.get("comic_title", ""),
        description=result.get("comic_description", ""),
        keyword_tags=result.get("comic_keywords", []),
        created_at=datetime.utcnow().isoformat(),
        thumbnail_url=result.get("thumbnail_url", "https://dummy.local/thumbnail.png"),
        cuts=cuts
    )
