import requests
import io
import time
from PIL import Image

# 서버 URL
SERVER_URL = "https://publicly-capable-monkfish.ngrok-free.app/"    # static url
TEXT_TO_IMAGE_ENDPOINT = f"{SERVER_URL}/generate/text-to-image"
IMAGE_TO_IMAGE_ENDPOINT = f"{SERVER_URL}/generate/image-to-image"

# 공통 설정
NEGATIVE_PROMPT = "(worst quality, low quality, normal quality:1.2), deformed, blurry, text, signature"
NUM_IMAGES_TO_GENERATE = 4  # 총 생성할 이미지 수
BASE_SEED = 42
LORA_SCALE = 0.0
CONTROLNET_SCALE = 0.7
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 3.5

# 프레임별 Prompt 목록
FRAME_PROMPTS = [
    "A futuristic cityscape with flying cars and neon lights",  # Frame 1
    "A futuristic cityscape at sunset with flying cars and glowing neon lights",  # Frame 2
    "A futuristic cityscape at night with flying cars above illuminated skyscrapers",  # Frame 3
    "A futuristic cityscape during a light rain, flying cars with reflections on the wet streets"  # Frame 4
]

# 현재 생성한 이미지
current_image = None

# 1. 첫 번째 이미지는 text-to-image
print("\n🚀 [1/4] 텍스트-이미지 생성 요청 중...")

payload = {
    "prompt": FRAME_PROMPTS[0],
    "negative_prompt": NEGATIVE_PROMPT,
    "num_inference_steps": NUM_INFERENCE_STEPS,
    "guidance_scale": GUIDANCE_SCALE,
    "lora_scale": LORA_SCALE,
    "seed": BASE_SEED
}

try:
    response = requests.post(TEXT_TO_IMAGE_ENDPOINT, json=payload, timeout=300)

    if response.status_code == 200:
        print("✅ 텍스트-이미지 생성 성공!")
        current_image = Image.open(io.BytesIO(response.content))
        filename = f"generated_frame_01.png"
        current_image.save(filename)
        print(f"💾 '{filename}'로 저장 완료")
    else:
        print(f"❌ 텍스트-이미지 생성 실패: {response.status_code}")
        print(f"오류 내용: {response.text}")
        exit(1)

except requests.exceptions.RequestException as e:
    print(f"❌ 텍스트-이미지 요청 실패: {e}")
    exit(1)

time.sleep(1)

# 2. 이후부터는 image-to-image
for idx in range(1, NUM_IMAGES_TO_GENERATE):
    print(f"\n🚀 [{idx+1}/{NUM_IMAGES_TO_GENERATE}] 이미지-이미지 생성 요청 중...")

    img_byte_arr = io.BytesIO()
    current_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # 수정된 files 구성
    files = {
        "image": ("input.png", img_byte_arr, "image/png"),
        "prompt": (None, FRAME_PROMPTS[idx]),
        "negative_prompt": (None, NEGATIVE_PROMPT),
        "lora_scale": (None, str(LORA_SCALE)),
        "controlnet_scale": (None, str(CONTROLNET_SCALE)),
        "num_inference_steps": (None, str(NUM_INFERENCE_STEPS)),
        "guidance_scale": (None, str(GUIDANCE_SCALE)),
        "seed": (None, str(BASE_SEED + idx))
    }


    try:
        response = requests.post(IMAGE_TO_IMAGE_ENDPOINT, files=files, timeout=300)

        if response.status_code == 200:
            print("✅ 이미지-이미지 생성 성공!")
            current_image = Image.open(io.BytesIO(response.content))
            filename = f"generated_frame_{idx+1:02d}.png"
            current_image.save(filename)
            print(f"💾 '{filename}'로 저장 완료")
        else:
            print(f"❌ 이미지-이미지 생성 실패: {response.status_code}")
            print(f"오류 내용: {response.text}")
            exit(1)

    except requests.exceptions.RequestException as e:
        print(f"❌ 이미지-이미지 요청 실패: {e}")
        exit(1)

    time.sleep(1)
