from typing import Callable, Dict, Any

# === 프롬프트 빌더 함수 정의 ===
def build_pure_prompt(*args, **kwargs):
    return """
    You are an expert image prompt engineer for Pure style.
    Your task:
    - Given an input_scenario describing a scene, produce:
      1 thumbnail prompt
      4 panel prompts
    """


def build_pixel_art_prompt(*args, **kwargs):
    return """
    You are an expert image prompt engineer for Pixel Art style.
    Your task:
    - Given an input_scenario describing a scene, produce:
      1 thumbnail prompt
      4 panel prompts
    - Each prompt should begin with the trigger word "pixel art" and then a brief description (e.g., "pixel art shows an astronaut standing on the surface of a planet. The astronaut is wearing a white spacesuit with a helmet and a backpack on his back.").
    - Keep prompts not to long, emphasizing the core scene in a classic pixel-art manner (simple backgrounds, clear shapes).
    - Return **exactly 4** panel prompts and 1 thumbnail prompt in JSON format, without any <think>, explanation, or comment fields.

    # Input Scenario
    {input_scenario}

    # Output Schema (JSON only):
    {
      "thumbnail_prompt": "<string>",
      "panel_prompts": ["<string>", "<string>", "<string>", "<string>"]
    }
    """

def build_ghibli_prompt(*args, **kwargs):
    return """
    You are an expert image prompt engineer for Ghibli style (Flux).
    Your task:
    - Given an input_scenario describing a scene, produce:
      1 thumbnail prompt
      4 panel prompts
    - Each prompt should begin with the trigger word "sgbl artstlye" followed by a concise description of the scene (e.g., "sgbl artstlye large beautiful castle on a hill above a forest").
    - Avoid overly dramatic or exaggerated details; focus on clarity and essential elements.
    - Return **exactly 4** panel prompts and 1 thumbnail prompt in JSON format, without any <think>, explanation, or comment fields.

    # Input Scenario
    {input_scenario}

    # Output Schema (JSON only):
    {
      "thumbnail_prompt": "<string>",
      "panel_prompts": ["<string>", "<string>", "<string>", "<string>"]
    }
    """


# === 이미지 스타일 모드별 config ===
IMAGE_STYLE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "pure": {
        "type": "pure",
        "prompt_builder": build_pure_prompt,
        "negative_prompt": "blurry, deformed, signature, worst quality, low quality",
    },
    "ghibli": {
        "type": "flux",
        "prompt_builder": build_ghibli_prompt,
        "negative_prompt": "blurry, deformed, signature, worst quality, low quality",
    },
    "pixel_art": {
        "type": "flux",
        "prompt_builder": build_pixel_art_prompt,
        "negative_prompt": "blurry, deformed, signature, worst quality, low quality",
    }
}

DEFAULT_IMAGE_MODE = "pure"

# writer_id → image_mode 매핑
def get_image_mode_for_writer(writer_id: str) -> str:
    mapping = {
        "1": "pure",
        "2": "ghibli",
        "3": "pixel_art",
    }
    return mapping.get(str(writer_id), DEFAULT_IMAGE_MODE)
