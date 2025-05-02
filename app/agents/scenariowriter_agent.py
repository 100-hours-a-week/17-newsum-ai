import logging
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from app.workflows.state import ComicState
from app.services.llm_server_client import call_llm_api

logger = logging.getLogger(__name__)

# 1️⃣ LLM 응답을 위한 출력 구조 정의 (모두 영어로 응답)
class ScenarioResponse(BaseModel):
    description: str = Field(..., description="짧은 시각 장면 설명 (영어)")
    dialogue: str = Field(..., description="짧은 대사 또는 설명 (영어)")
    prompt: str = Field(..., description="Flux 스타일 프롬프트 (자연어 문장, 영어)")

# vLLM guided_json 용 schema 생성
scenario_json_schema = ScenarioResponse.model_json_schema()

# 2️⃣ few-shot 프롬프트 생성기 (출력 포함 영어로 작성됨)
# prompt 형식 : "[Subject], [Background], [Composition], [Lighting], [Emotional tone or color style]" + [LoRA Style Trigger]
def generate_prompt_template_with_fewshots(humor_text: str) -> str:
    fewshot_1 = '''
Humor: "At a climate summit, no one notices the globe is on fire."

Output:
{
  "description": "A burning globe sits in the middle of the conference table while people talk around it.",
  "dialogue": "Politician: 'That's just a visual effect, nothing to worry about!'",
  "prompt": "A burning globe on the center of a modern conference table, surrounded by politicians in suits, wide angle composition, warm cinematic lighting, surreal and dramatic atmosphere"
}'''

    fewshot_2 = '''
Humor: "A scientist faints after reading a high temperature on a thermometer outside."

Output:
{
  "description": "A sweating scientist stares at a thermometer showing a dangerously high temperature.",
  "dialogue": "Scientist: 'It's rising again? This can't be real!'",
  "prompt": "A stressed scientist holding a thermometer, standing outside a research building, close-up composition, bright midday light, tense and alarming tone"
}'''

    instruction = f"""
You are a professional visual storyteller designing scenes for a comic.

For each humorous situation, generate:

1. A short visual **description** of the scene (1–2 sentences, in English).
2. A short **dialogue or caption** (in English).
3. A **Flux 1 Dev style image generation prompt** in natural English — a full sentence with visual structure:
   - Include: main subject, background context, composition (e.g., close-up, wide angle), lighting style, emotional tone
   - Do not use key:value pairs. Instead, make it a natural descriptive sentence.
   - Keep the tone visually rich and immersive.

Respond in this strict JSON format:
{{
  "description": "...",
  "dialogue": "...",
  "prompt": "..."
}}

Here are two examples:
{fewshot_1}

{fewshot_2}

Now, write the output in the same format for this scene:

Humor: "{humor_text}"
"""
    return instruction.strip()


# 3️⃣ 시나리오 생성 에이전트
class ScenarioWriterAgent:
    async def run(self, state: ComicState) -> Dict[str, Any]:
        logger.info("--- [ScenarioWriterAgent] 실행 시작 ---")
        updates: Dict[str, Any] = {}
        humor_texts: List[str] = state.humor_texts or []

        if not humor_texts:
            logger.warning("[ScenarioWriterAgent] humor_texts가 비어 있습니다.")
            updates["scenarios"] = []
            updates["error_message"] = "No humor texts to process."
            return updates

        # 🎯 이제 style은 쓰지 않고 lora_style만 사용
        lora_style = state.lora_style  # None이면 그대로 두고, 있으면 prompt에 추가

        results = []

        for idx, humor in enumerate(humor_texts):
            prompt = generate_prompt_template_with_fewshots(humor.strip())
            logger.info(f"[ScenarioWriterAgent] ({idx+1}/{len(humor_texts)}) 시나리오 생성 중...")

            try:
                response = await call_llm_api(
                    prompt,
                    max_tokens=512,
                    temperature=0.7,
                    guided_json=scenario_json_schema
                )

                if not response or response.strip() == "":
                    raise ValueError("LLM 응답이 비어 있음")

                parsed = json.loads(response)

                required = {"description", "dialogue", "prompt"}
                if not required.issubset(parsed):
                    raise ValueError(f"응답에 필수 키 누락: {parsed.keys()}")

                # 👉 스타일 추가: lora_style만 사용
                if lora_style:
                    parsed["prompt"] = f"{parsed['prompt'].strip()}, {lora_style}"

                results.append(parsed)

            except Exception as e:
                logger.error(f"[ScenarioWriterAgent] humor[{idx}] 처리 실패: {e}")
                results.append({
                    "description": "[ERROR]",
                    "dialogue": "[ERROR]",
                    "prompt": "[ERROR]"
                })

        updates["scenarios"] = results
        updates["error_message"] = None
        logger.info("--- [ScenarioWriterAgent] 실행 종료 ---")
        return updates