#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
flux_tunneling_request.py

Flux Tunneling 서버에 ① Pixel Art 비교(LoRA vs Pure) ② 다양한 스타일 테스트
두 가지 모드를 선택하여 순차적으로 요청을 보내고, 받은 이미지를 로컬에 저장/표시합니다.

– 설정:
  1) IMAGE_SERVER_URL: 본인의 Flux Tunneling 서버 주소로 수정
  2) 필요 시 HEADERS 에 인증 토큰(Authorization) 추가
  3) 서버 API 스펙에 맞춰 payload 필드 이름 조정
"""

import time
import os
import io
import requests
import base64
from PIL import Image

# ==========================
# 1. 설정: 서버 URL 및 인증
# ==========================
IMAGE_SERVER_URL = "https://clam-talented-promptly.ngrok-free.app"  # 예시: 본인 서버 주소로 수정
TEXT_TO_IMAGE_ENDPOINT = f"{IMAGE_SERVER_URL}/generate/text-to-image"

HEADERS = {
    "Content-Type": "application/json",
    # 필요한 경우 인증 헤더 추가:
    # "Authorization": "Bearer YOUR_API_TOKEN",
}

# ==============================
# 2. 테스트 모드 선택 (하나만 True)
# ==============================
RUN_PIXEL_ART_COMPARISON = True
RUN_MULTISTYLE_TEST = False

# ===========================================
# 3. 스타일별 요청 정의 (모드별 딕셔너리)
# ===========================================
STYLE_REQUESTS_PIXEL_ART_COMPARISON = {
    # # ────────────────────────────────────────────────────────────────────
    # # 1) Pixel Art (LoRA-applied) – Depth 1~4
    # # ────────────────────────────────────────────────────────────────────
    # "pixel_art_lora_depth1": {
    #     "model_name": "pixel_art",
    #     "prompt": "pixel art astronaut on a planet",
    #     "negative_prompt": "blurry, deformed, signature"
    # },
    # "pixel_art_lora_depth2": {
    #     "model_name": "pixel_art",
    #     "prompt": (
    #         "pixel art shows an astronaut standing on a planet surface "
    #         "with a speech bubble saying \"how beautiful\""
    #     ),
    #     "negative_prompt": "blurry, deformed, signature"
    # },
    # "pixel_art_lora_depth3": {
    #     "model_name": "pixel_art",
    #     "prompt": (
    #         "pixel art shows an astronaut standing on a rocky planet surface, "
    #         "facing right, wearing a white spacesuit and backpack. "
    #         "Behind the astronaut is a large pinkish-orange planet against a dark, starry sky. "
    #         "A speech bubble above reads \"how beautiful\"."
    #     ),
    #     "negative_prompt": "blurry, deformed, signature"
    # },
    # "pixel_art_lora_depth4": {
    #     "model_name": "pixel_art",
    #     "prompt": (
    #         "pixel art depicts an astronaut in a detailed white spacesuit with a reflective visor "
    #         "and a bulky backpack standing on a rocky, cratered planet surface illuminated by starlight. "
    #         "The astronaut faces right, gazing at a massive glowing pinkish-orange planet on the horizon. "
    #         "Numerous tiny stars twinkle in the dark sky, and the planet surface is covered with pixelated "
    #         "rocks and small craters. A thick white speech bubble contains black pixel text: \"how beautiful.\""
    #     ),
    #     "negative_prompt": "blurry, deformed, signature"
    # },

    # # ────────────────────────────────────────────────────────────────────
    # # 2) Pixel Art (Pure Flux Schnelle) – Depth 1~4
    # # ────────────────────────────────────────────────────────────────────
    # "pixel_art_pure_depth1": {
    #     "model_name": "pure",
    #     "prompt": "pixel art astronaut on a planet",
    #     "negative_prompt": "blurry, deformed, signature"
    # },
    # "pixel_art_pure_depth2": {
    #     "model_name": "pure",
    #     "prompt": (
    #         "pixel art shows an astronaut standing on a planet surface "
    #         "with a speech bubble saying \"how beautiful\""
    #     ),
    #     "negative_prompt": "blurry, deformed, signature"
    # },
    # "pixel_art_pure_depth3": {
    #     "model_name": "pure",
    #     "prompt": (
    #         "pixel art shows an astronaut standing on a rocky planet surface, "
    #         "facing right, wearing a white spacesuit and backpack. "
    #         "Behind the astronaut is a large pinkish-orange planet against a dark, starry sky. "
    #         "A speech bubble above reads \"how beautiful\"."
    #     ),
    #     "negative_prompt": "blurry, deformed, signature"
    # },
    # "pixel_art_pure_depth4": {
    #     "model_name": "pure",
    #     "prompt": (
    #         "pixel art depicts an astronaut in a detailed white spacesuit with a reflective visor "
    #         "and a bulky backpack standing on a rocky, cratered planet surface illuminated by starlight. "
    #         "The astronaut faces right, gazing at a massive glowing pinkish-orange planet on the horizon. "
    #         "Numerous tiny stars twinkle in the dark sky, and the planet surface is covered with pixelated "
    #         "rocks and small craters. A thick white speech bubble contains black pixel text: \"how beautiful.\""
    #     ),
    #     "negative_prompt": "blurry, deformed, signature"
    # },

    # prompt strcuture test (250605_1330)
    # "pure_test1": {
    #     "model_name": "pure",
    #     "prompt": "A single tree standing in the middle of the image. The left half of the tree has bright, vibrant yellow and green leaves under a bright, sunny blue sky, while the right half has bare branches covered in frost, with a cold, dark, thunderous sky. On the left there's green, lush grass on the ground; on the right - thick snow. The split is sharp, with the transition happening right down the middle of the tree",
    #     "negative_prompt": ""
    # },
    # "pure_test2": {
    #     "model_name": "pure",
    #     "prompt": "In the foreground, a vintage car with a 'CLASSIC' plate with green text is parked on a cobblestone street. Behind it, a bustling market scene with colorful awnings. In the background, the silhouette of an old castle on a hill, shrouded in mist",
    #     "negative_prompt": ""
    # },

    "pixel_test1": {
        "model_name": "pixel_art",
        "prompt": "pixel art, A single tree standing in the middle of the image. The left half of the tree has bright, vibrant yellow and green leaves under a bright, sunny blue sky, while the right half has bare branches covered in frost, with a cold, dark, thunderous sky. On the left there's green, lush grass on the ground; on the right - thick snow. The split is sharp, with the transition happening right down the middle of the tree",
        "negative_prompt": ""
    },
    # "pixel_test2": {
    #     "model_name": "pixel_art",
    #     "prompt": "pixel art, In the foreground, a vintage car with a 'CLASSIC' plate with green text is parked on a cobblestone street. Behind it, a bustling market scene with colorful awnings. In the background, the silhouette of an old castle on a hill, shrouded in mist",
    #     "negative_prompt": ""
    # }

    # n08a test (250605_1525)
    "pixel_test0": {
        "model_name": "pure",
        "prompt": "pixel art shows an overconfident AI Doctor standing center-left in a futuristic hospital. The scene is split: the doctor confidently points at a floating screen reading 'EXISTENTIAL DREAD' while a worried Human Doctor watches from the right. Cool blue lighting, holographic monitors, and ironic undertone highlight the absurdity of AI making life decisions. Futuristic instrumentation lines the walls.",
        "negative_prompt": ""
    },
    # "pixel_test1": {
    #     "model_name": "pure",
    #     "prompt": "pixel art, medium shot of an AI Doctor in a glowing circuit-patterned lab coat standing on the left and a bewildered Patient seated on the right, both under cool blue neon light, while a Human Doctor with arms crossed looks on nervously from the background. A floating holographic screen displays 'EXISTENTIAL DREAD'.",
    #     "negative_prompt": ""
    # },
    # "pixel_test2": {
    #     "model_name": "pure",
    #     "prompt": "pixel art, over-the-shoulder view from behind the AI Doctor holding a digital tablet reading '401(k) AUDIT' as a prescription. The Patient leans forward in shock under bright clinical lighting, and the Human Doctor fumbles with a file in the background. Medical consoles glow with ironic green numbers.",
    #     "negative_prompt": ""
    # },
    # "pixel_test3": {
    #     "model_name": "pure",
    #     "prompt": "pixel art, close-up on a high-tech console screen showing health stats overlaid with financial graphs labeled 'Mood Index' and 'Retirement Fund'. The AI Doctor smirks on the left, pointing; the Patient’s frightened reflection appears on the screen. Cool digital glow, green numeric overlays, and blink effects.",
    #     "negative_prompt": ""
    # },
    # "pixel_test4": {
    #     "model_name": "pure",
    #     "prompt": "pixel art, wide shot of a chaotic hospital corridor labeled 'AI HEALTHCARE UNIT' overhead. The AI Doctor stands proudly center with a blissful smile; the confused Patient peeks from the right; the exasperated Human Doctor yells off to the left. Bright overhead lights, flickering monitors, and frantic staff fill the hallway.",
    #     "negative_prompt": ""
    # },
}

STYLE_REQUESTS_MULTISTYLE = {
    # ────────────────────────────────────────────────────────────────────
    # A) Pixel Art Style – Depth 1~4
    # ────────────────────────────────────────────────────────────────────
    "style_pixel_art_depth1": {
        "model_name": "pure",
        "prompt": "pixel art cat sitting on a chair",
        "negative_prompt": "blurry, deformed, signature"
    },
    "style_pixel_art_depth2": {
        "model_name": "pure",
        "prompt": "pixel art shows a cat with a big smile sitting on a chair in front of a fireplace",
        "negative_prompt": "blurry, deformed, signature"
    },
    "style_pixel_art_depth3": {
        "model_name": "pure",
        "prompt": (
            "pixel art shows a brown and black cat sitting on a chair in front of a wooden fireplace, "
            "flames gently flicker behind the cat, the cat has a big smile and wears a small red hat"
        ),
        "negative_prompt": "blurry, deformed, signature"
    },
    "style_pixel_art_depth4": {
        "model_name": "pure",
        "prompt": (
            "pixel art depicts a brown and black cat sitting on a wooden chair before a roaring fireplace. "
            "Flames cast warm orange light across the scene, highlighting the cat’s big grin and its red hat. "
            "The chair is pixelated with wood grain details, and fireplace logs crackle below the dancing flames. "
            "A tiny table beside holds a steaming cup of coffee."
        ),
        "negative_prompt": "blurry, deformed, signature"
    },

    # ────────────────────────────────────────────────────────────────────
    # B) Photorealistic Style – Depth 1~4
    # ────────────────────────────────────────────────────────────────────
    "style_photorealistic_depth1": {
        "model_name": "pure",
        "prompt": "photorealistic portrait of a weathered sailor",
        "negative_prompt": "blurry, low quality, cartoon, signature"
    },
    "style_photorealistic_depth2": {
        "model_name": "pure",
        "prompt": (
            "photorealistic portrait of a middle-aged sailor with a weathered face, "
            "wrinkles around his eyes, wearing a navy cap"
        ),
        "negative_prompt": "blurry, low quality, cartoon, signature"
    },
    "style_photorealistic_depth3": {
        "model_name": "pure",
        "prompt": (
            "photorealistic portrait of a middle-aged sailor with sun-tanned, weathered skin, "
            "deep wrinkles around his blue eyes, wearing a worn navy cap and a striped shirt. "
            "Soft directional lighting from the left accentuates his textured beard."
        ),
        "negative_prompt": "blurry, low quality, cartoon, signature"
    },
    "style_photorealistic_depth4": {
        "model_name": "pure",
        "prompt": (
            "ultra-detailed photorealistic portrait of a middle-aged sailor: "
            "sun-tanned, leathery skin, deep crow’s feet around ice-blue eyes, "
            "grizzled gray beard with salt stains, wearing a faded navy cap and a weathered striped shirt. "
            "Soft golden-hour lighting from the left highlights every wrinkle, pore, "
            "and individual beard hair. Background is softly blurred harbor scene."
        ),
        "negative_prompt": "blurry, low quality, cartoon, signature"
    },

    # ────────────────────────────────────────────────────────────────────
    # C) Painterly Style – Depth 1~4
    # ────────────────────────────────────────────────────────────────────
    "style_painterly_depth1": {
        "model_name": "pure",
        "prompt": "painterly flower bouquet",
        "negative_prompt": "blurry, low detail, signature"
    },
    "style_painterly_depth2": {
        "model_name": "pure",
        "prompt": "painterly painting of a colorful flower bouquet in a vase",
        "negative_prompt": "blurry, low detail, signature"
    },
    "style_painterly_depth3": {
        "model_name": "pure",
        "prompt": (
            "painterly artwork of a vibrant bouquet of roses and sunflowers in a glass vase, "
            "bold brush strokes, rich oil texture, warm ambient light"
        ),
        "negative_prompt": "blurry, low detail, signature"
    },
    "style_painterly_depth4": {
        "model_name": "pure",
        "prompt": (
            "oil painterly painting showing a lush bouquet of red roses, yellow sunflowers, and "
            "purple lilies in a crystal-clear glass vase. Thick brush strokes create visible texture, "
            "vermilion highlights catch soft afternoon light, background is muted ochre canvas."
        ),
        "negative_prompt": "blurry, low detail, signature"
    },

    # ────────────────────────────────────────────────────────────────────
    # D) Cartoon/Comic Book Style – Depth 1~4
    # ────────────────────────────────────────────────────────────────────
    "style_cartoon_depth1": {
        "model_name": "pure",
        "prompt": "cartoon superhero standing",
        "negative_prompt": "blurry, deformed, signature"
    },
    "style_cartoon_depth2": {
        "model_name": "pure",
        "prompt": "cartoon shows a superhero standing with hands on hips",
        "negative_prompt": "blurry, deformed, signature"
    },
    "style_cartoon_depth3": {
        "model_name": "pure",
        "prompt": (
            "cartoon style illustration of a superhero wearing a red cape, standing "
            "on a rooftop with hands on hips, wind blowing the cape to the side"
        ),
        "negative_prompt": "blurry, deformed, signature"
    },
    "style_cartoon_depth4": {
        "model_name": "pure",
        "prompt": (
            "comic book style illustration of a muscular superhero in a red and blue suit, "
            "standing on a tall rooftop at dusk, wind gusts flaring the red cape. "
            "City skyline with skyscrapers and neon lights behind; bold black outlines, "
            "vivid colors, dynamic shading."
        ),
        "negative_prompt": "blurry, deformed, signature"
    },

    # ────────────────────────────────────────────────────────────────────
    # E) Watercolor Style – Depth 1~4
    # ────────────────────────────────────────────────────────────────────
    "style_watercolor_depth1": {
        "model_name": "pure",
        "prompt": "watercolor forest scene",
        "negative_prompt": "blurry, muddy colors, signature"
    },
    "style_watercolor_depth2": {
        "model_name": "pure",
        "prompt": "watercolor painting of a forest with tall pine trees",
        "negative_prompt": "blurry, muddy colors, signature"
    },
    "style_watercolor_depth3": {
        "model_name": "pure",
        "prompt": (
            "watercolor artwork of a misty forest with tall pine trees, soft washes "
            "of green and blue, diffused light filtering through branches"
        ),
        "negative_prompt": "blurry, muddy colors, signature"
    },
    "style_watercolor_depth4": {
        "model_name": "pure",
        "prompt": (
            "detailed watercolor painting of a misty early morning forest: tall pine trees "
            "with delicate watercolor washes, soft transitions between emerald greens and "
            "sky blues, mist rolling between trunks, dappled sunlight creating subtle highlights."
        ),
        "negative_prompt": "blurry, muddy colors, signature"
    },
}

# ================================================
# 4. 출력 폴더 생성 (타임스탬프 포함)
# ================================================
timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
if RUN_PIXEL_ART_COMPARISON:
    OUTPUT_DIR = f"flux_output_images/{timestamp}_pixel_art_comparison"
elif RUN_MULTISTYLE_TEST:
    OUTPUT_DIR = f"flux_output_images/{timestamp}_multistyle_test"
else:
    raise RuntimeError("✗ 테스트 모드를 하나만 True로 설정하세요: RUN_PIXEL_ART_COMPARISON 또는 RUN_MULTISTYLE_TEST")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# 5. 이미지 저장 함수: 서버 응답에서 이미지 추출/저장
# ==================================================
def save_image_bytes(content: bytes, filename: str):
    """
    response.content(바이너리) 또는 base64 디코딩 후 byte 데이터를
    지정된 경로에 PNG로 저장합니다.
    """
    with open(filename, "wb") as f:
        f.write(content)
    print(f"[✔] Saved: {filename}")


def process_response(response: requests.Response, out_path: str):
    """
    HTTP 응답에서 이미지 데이터를 감지하여 로컬에 저장.
    – image/png 반환: response.content로 저장.
    – JSON 반환(Base64 포함): base64 디코딩 후 저장.
    """
    content_type = response.headers.get("Content-Type", "")
    if response.status_code != 200:
        print(f"[✗] HTTP {response.status_code} Error - {response.text}")
        return False

    if content_type.startswith("image/"):
        # 이미지 바이너리 직접 반환된 경우
        save_image_bytes(response.content, out_path)
        return True

    # JSON 반환(Base64 이미지 데이터 등) 처리
    try:
        j = response.json()
    except Exception:
        print(f"[✗] JSON 파싱 실패: {response.text}")
        return False

    # 예: { "image_base64": "iVBORw0..." }
    b64_key = "image_base64"
    if b64_key in j:
        try:
            img_data = base64.b64decode(j[b64_key])
            save_image_bytes(img_data, out_path)
            return True
        except Exception as e:
            print(f"[✗] Base64 디코딩 실패: {e}")
            return False
    else:
        print(f"[✗] 예상치 못한 JSON 구조: {j}")
        return False


# ==================================================
# 6. 요청 루프: 스타일별로 반복 요청 및 저장
# ==================================================
def run_tests(style_requests: dict):
    """
    style_requests 딕셔너리의 각 스타일 키에 대해
    여러 번(예: 3회) 서버에 요청을 보내고 결과를 저장합니다.
    """
    total_start = time.time()

    # 반복 횟수 (이미지 variability 확인용; 필요 시 조정)
    TRIES_PER_STYLE = 3

    for style_name, payload in style_requests.items():
        style_dir = os.path.join(OUTPUT_DIR, style_name)
        os.makedirs(style_dir, exist_ok=True)
        print(f"\n[→] Testing '{style_name}' ({payload['model_name']}) ...")

        for i in range(TRIES_PER_STYLE):
            try:
                resp = requests.post(TEXT_TO_IMAGE_ENDPOINT, json=payload, headers=HEADERS, timeout=300)
            except requests.exceptions.RequestException as e:
                print(f"[✗] Network error for '{style_name}' (attempt {i+1}): {e}")
                continue

            filename = os.path.join(style_dir, f"{style_name}_{i+1}.png")
            success = process_response(resp, filename)
            if success:
                # 이미지 미리보기: PIL 사용 (콜백 환경에 따라 자동 팝업이 안 될 수 있음)
                try:
                    img = Image.open(io.BytesIO(resp.content))
                    img.show()
                except Exception:
                    pass

    print(f"\n[✔] All requests done. Time elapsed: {time.time() - total_start:.2f}s\n")


# ==================================================
# 7. 메인 함수: 선택된 모드 실행
# ==================================================
def main():
    if RUN_PIXEL_ART_COMPARISON:
        print("[Start] Running Pixel Art Comparison (LoRA vs Pure)")
        run_tests(STYLE_REQUESTS_PIXEL_ART_COMPARISON)

    elif RUN_MULTISTYLE_TEST:
        print("[Start] Running Multi-Style Test")
        run_tests(STYLE_REQUESTS_MULTISTYLE)


if __name__ == "__main__":
    main()