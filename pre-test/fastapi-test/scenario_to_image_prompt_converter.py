# scenario_to_image_prompt_converter.py
# 시나리오 결과 JSON → 이미지 생성용 프롬프트 변환 테스트 스크립트 (LLM 사용)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json

# 최신 OpenAI 모델로 LLM 초기화 (gpt-4-turbo)
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.5)

# 시나리오 결과 예시 (보통은 JSON 파일로부터 읽어옴)
with open("scenario_output.json", "r", encoding="utf-8") as f:
    scenario = json.load(f)

cut_contents = [cut["content"] for cut in scenario["cuts"]]
style = "Studio Ghibli style"

# 각 컷 내용을 확장 프롬프트로 변환
def convert_to_prompt(cut_text: str, style: str) -> str:
    prompt = f"""
    다음 문장은 만화 컷 설명이야:
    "{cut_text}"

    이 장면을 이미지 생성용 프롬프트로 바꿔줘. 
    조건:
    - '{style}' 스타일로 변환
    - 등장 인물, 배경, 분위기, 조명 등을 상상해서 묘사 추가
    - 'high detail, cinematic lighting' 품질 관련 키워드도 포함
    - 최종 결과는 한 줄 프롬프트만 출력
    """

    result = llm.invoke([HumanMessage(content=prompt)])
    return result.content.strip()

# 변환 실행 및 출력
final_prompts = []
for idx, cut_text in enumerate(cut_contents):
    print(f"\n컷 {idx+1} 내용: {cut_text}")
    prompt = convert_to_prompt(cut_text, style)
    final_prompts.append(prompt)
    print(f"→ 이미지 생성 프롬프트: {prompt}")

# 프롬프트 결과 저장 (선택)
with open("converted_prompts.json", "w", encoding="utf-8") as f:
    json.dump(final_prompts, f, indent=2, ensure_ascii=False)
