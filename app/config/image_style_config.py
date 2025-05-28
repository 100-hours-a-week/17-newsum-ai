from typing import Callable, Dict, Any

# === 프롬프트 빌더 함수 정의 ===
def build_ghibli_flux_prompt(*args, **kwargs):
    example_prompt = (
        "A young boy with bright orange hair and a mischievous grin is riding on the back "
        "of a large, fluffy bear through the village. The bear has a thick white coat and moves "
        "gently through the cobblestone streets. The villagers watch in quiet amusement. "
        "The sky is clear, and the mood is pleasantly serene. The camera is at eye level, capturing "
        "the scene with balanced details."
    )
    return f"""
You are an expert image prompt engineer for narrative style (Flux).
Your task:
- Refine and polish the candidate descriptions into concise, balanced English sentences suitable for Flux-style image generation.
- Return **exactly 4** panel prompts and 1 thumbnail prompt (no more, no less).
- Do **not** include any <think>, explanation, or comment fields in your output.
- Avoid overly dramatic or exaggerated details; focus on clarity and essential elements.

# Example (thumbnail or panel)
{example_prompt}

# Output Schema (JSON only):
{{
  "thumbnail_prompt": "<string>",
  "panel_prompts": ["<string>", "<string>", "<string>", "<string>"]
}}
"""


def build_anything_xl_prompt(*args, **kwargs):
    example_tokens = [
        "masterpiece", "best quality", "1girl", "solo", "animal ears", 
        "bow", "teeth", "jacket", "tail", "open mouth"
    ]
    example_line = ", ".join(example_tokens)
    return f"""
You are an expert image prompt engineer for token-based style (XL).
Your task:
- Refine the keyword set for each illustration, filtering out generic or redundant tokens and focusing on essential attributes.
- Return **exactly 4** lists of panel tokens and 1 list of thumbnail tokens (no more, no less).
- Do **not** include any <think>, explanation, or comment fields.
- Avoid overloading with unnecessary embellishments.

# Example Tokens:
{example_line}

# Output Schema (JSON only):
{{
  "thumbnail_tokens": ["<string>", ...],
  "panel_tokens": [
     ["<string>", ...],
     ["<string>", ...],
     ["<string>", ...],
     ["<string>", ...]
  ]
}}
"""


def build_cute_line_cat_flux_prompt(*args, **kwargs):
    example_prompts = [
        "miao cat play computer game, cartoon, simple background",
        "miao cat reading book, cartoon, simple background"
    ]
    return f"""
You are an expert image prompt engineer for Cute-Line-Cat style.
Your task:
- Given a detailed input_scenario (possibly lengthy), transform it into:
  1 thumbnail prompt
  4 panel prompts
- Each prompt should start with “miao cat”, use "cartoon" in the prompt, be short, playful, and highlight the core action in a light, cartoonish manner.
- Return **exactly 4** panel prompts and 1 thumbnail prompt (no more, no less).
- Do **not** include any <think>, explanation, or comment fields.

# Example Prompts:
- Panel 1: {example_prompts[0]}
- Panel 2: {example_prompts[1]}

# Input Scenario
{{input_scenario}}

# Output Schema (JSON only):
{{
  "thumbnail_prompt": "<string>",
  "panel_prompts": ["<string>", "<string>", "<string>", "<string>"]
}}
"""


# === 이미지 스타일 모드별 config ===
IMAGE_STYLE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ghibli-flux": {
        "type": "flux",
        "prompt_builder": build_ghibli_flux_prompt,
        "negative_prompt": "(worst quality, low quality, normal quality:1.2), deformed, blurry, text, signature",
    },
    "anything-xl": {
        "type": "xl",
        "prompt_builder": build_anything_xl_prompt,
        "negative_prompt": "bad anatomy, bad hands, text, error, missing fingers",
    },
    "cute-line-cat-flux": {
        "type": "flux",
        "prompt_builder": build_cute_line_cat_flux_prompt,
        "negative_prompt": ",nsfw",
    },
}

DEFAULT_IMAGE_MODE = "ghibli-flux"

# writer_id → image_mode 매핑
def get_image_mode_for_writer(writer_id: str) -> str:
    mapping = {
        "1": "ghibli-flux",
        "2": "anything-xl",
        "3": "cute-line-cat-flux",
    }
    return mapping.get(str(writer_id), DEFAULT_IMAGE_MODE)
