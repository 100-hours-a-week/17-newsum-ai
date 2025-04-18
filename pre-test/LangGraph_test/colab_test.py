from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI  # pip install -U langchain-openai

# 1. 상태 정의 (LangGraph 요구 사항 대응)
class ChatState(TypedDict):
    messages: List[BaseMessage]

# 2. vLLM (로컬 서버 또는 Cloudflared 주소) 연결
llm = ChatOpenAI(
    openai_api_base="https://names-water-jc-scoring.trycloudflare.com/v1",  # 꼭 /v1 포함
    openai_api_key="EMPTY",  # vLLM 기본 설정
    model="/content/drive/MyDrive/unsloth_models/Llama_3.2_3B_test/merged_16bit",
)

# 3. 노드 함수 정의
def run_llm_node(state: ChatState) -> ChatState:
    message = state["messages"][-1]
    response = llm.invoke([message])
    return {"messages": state["messages"] + [response]}

# 4. LangGraph 구성
builder = StateGraph(ChatState)
builder.add_node("llm", run_llm_node)
builder.set_entry_point("llm")
workflow = builder.compile()

# 5. 입력 및 실행
response = workflow.invoke({"messages": [HumanMessage(content="What is QLoRA?")]})
print(response["messages"][-1].content)