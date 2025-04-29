import requests
import io
import time
from PIL import Image
import os

# 서버 URL
SERVER_URL = "https://publicly-capable-monkfish.ngrok-free.app"  # 주의: 마지막 슬래시 제거
IMAGE_TO_IMAGE_ENDPOINT = f"{SERVER_URL}/generate/image-to-image"

# 설정
PROMPT = "A futuristic cityscape at night with flying cars above illuminated skyscrapers"
NEGATIVE_PROMPT = "(worst quality, low quality, normal quality:1.2), deformed, blurry, text, signature"
LORA_SCALE = 0.0
CONTROLNET_SCALE = 0.7
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 3.5
SEED = 42

# 디렉토리 및 파일 확인
print(f"현재 작업 디렉토리: {os.getcwd()}")
print(f"현재 디렉토리 파일 목록: {os.listdir()}")

# 준비: 테스트용 입력 이미지 (test_image.png 파일 읽기)
try:
    with open("i2i_test_image.png", "rb") as f:
        img_byte_arr = io.BytesIO(f.read())
        img_byte_arr.seek(0)
except FileNotFoundError:
    print("❌ 'i2i_test_image.png' 파일을 현재 디렉토리에서 찾을 수 없습니다.")
    raise

# 파일 및 폼 데이터 준비
files = {
    "image": ("i2i_test_image.png", img_byte_arr, "image/png"),
    "prompt": (None, PROMPT),
    "negative_prompt": (None, NEGATIVE_PROMPT),
    "lora_scale": (None, str(LORA_SCALE)),
    "controlnet_scale": (None, str(CONTROLNET_SCALE)),
    "num_inference_steps": (None, str(NUM_INFERENCE_STEPS)),
    "guidance_scale": (None, str(GUIDANCE_SCALE)),
    "seed": (None, str(SEED))
}

# 요청 보내기
print("\n🚀 이미지-이미지 단일 생성 요청 중...")

try:
    start_time = time.time()  # ⏱️ 요청 시작 시간 기록

    response = requests.post(IMAGE_TO_IMAGE_ENDPOINT, files=files, timeout=300)

    if response.status_code == 200:
        print("✅ 이미지-이미지 생성 성공!")
        result_image = Image.open(io.BytesIO(response.content))
        filename = "generated_i2i_test.png"
        result_image.save(filename)
        print(f"💾 '{filename}'로 저장 완료")
    else:
        print(f"❌ 이미지-이미지 생성 실패: {response.status_code}")
        print(f"오류 내용: {response.text}")

    end_time = time.time()  # ⏱️ 요청 종료 시간 기록
    elapsed_time = end_time - start_time
    print(f"\n⏳ 총 소요 시간: {elapsed_time:.2f}초")

except requests.exceptions.RequestException as e:
    print(f"❌ 이미지-이미지 요청 실패: {e}")
