import requests
import io
import time
from PIL import Image
import os

# 서버 URL (주의: 마지막 슬래시 제거)
SERVER_URL = "https://publicly-capable-monkfish.ngrok-free.app"
IP_ADAPTER_ENDPOINT = f"{SERVER_URL}/generate/ip-adapter-image"  # ✅ 엔드포인트 수정

# 설정
PROMPT = "A serene garden filled with blooming cherry blossoms, a girl with long teal twin tails sitting quietly under a sakura tree"
NEGATIVE_PROMPT = ""
LORA_SCALE = 0.0
IPADAPTER_SCALE = 0.7
NUM_INFERENCE_STEPS = 30
SEED = 42

# 디렉토리 및 파일 확인
print(f"현재 작업 디렉토리: {os.getcwd()}")
print(f"현재 디렉토리 파일 목록: {os.listdir()}")

# 준비: 테스트용 입력 이미지 (test_image.png 파일 읽기)
try:
    with open("ipa_test_image.png", "rb") as f:
        img_byte_arr = io.BytesIO(f.read())
        img_byte_arr.seek(0)
except FileNotFoundError:
    print("❌ 'ipa_test_image.png' 파일을 현재 디렉토리에서 찾을 수 없습니다.")
    raise

# 파일 및 폼 데이터 준비
files = {
    "image": ("ipa_test_image.png", img_byte_arr, "image/png"),
}

data = {
    "prompt": PROMPT,
    "negative_prompt": NEGATIVE_PROMPT,
    "lora_scale": str(LORA_SCALE),
    "ipadapter_scale": str(IPADAPTER_SCALE),
    "num_inference_steps": str(NUM_INFERENCE_STEPS),
    "seed": str(SEED),
}

# 요청 보내기
print("\n🚀 IP-Adapter 이미지 생성 요청 중...")

try:
    start_time = time.time()  # ⏱️ 요청 시작 시간 기록

    response = requests.post(IP_ADAPTER_ENDPOINT, files=files, data=data, timeout=300)

    if response.status_code == 200:
        print("✅ IP-Adapter 이미지 생성 성공!")
        result_image = Image.open(io.BytesIO(response.content))
        filename = "generated_ip_adapter_test.png"
        result_image.save(filename)
        print(f"💾 '{filename}'로 저장 완료")
    else:
        print(f"❌ IP-Adapter 이미지 생성 실패: {response.status_code}")
        print(f"오류 내용: {response.text}")

    end_time = time.time()  # ⏱️ 요청 종료 시간 기록
    elapsed_time = end_time - start_time
    print(f"\n⏳ 총 소요 시간: {elapsed_time:.2f}초")

except requests.exceptions.RequestException as e:
    print(f"❌ IP-Adapter 이미지 요청 실패: {e}")
