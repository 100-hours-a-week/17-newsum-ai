# # app/api/v1/stream_events.py

# import asyncio
# from app.workflows.main_workflow import build_main_workflow
# from app.workflows.state import ComicState

# async def comic_stream_generator(request):
#     """
#     LangGraph 워크플로우를 실행하면서 중간 결과를 스트리밍으로 보내는 함수.
#     """
#     graph = build_main_workflow()
#     initial_state = ComicState(news_urls=[], articles=[], summaries=[], humor_texts=[], scenarios=[])

#     async for step_result in graph.astream(initial_state):
#         yield f"data: {step_result}\n\n"
#         await asyncio.sleep(0.1)  # throttle to avoid flooding
