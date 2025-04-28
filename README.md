# Newsom AI 서비스

뉴스 기사를 기반으로 4컷 만화를 생성하는 다단계 AI 워크플로우
(LangGraph + FastAPI 기반)

## 폴더 구조
langgraph/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── graphs/
│   ├── __init__.py
│   └── base_graph.py
├── nodes/
│   ├── __init__.py
│   └── base_node.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
└── examples/
    ├── __init__.py
    └── simple_conversation.py

### Creating New Nodes

To create new node types, extend the `BaseNode` or `LLMNode` classes in the `nodes` directory:
