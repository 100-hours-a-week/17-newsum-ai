import requests
import io
from PIL import Image

# FastAPI 서버의 URL (Ngrok URL 또는 로컬 서버 주소)
# Ngrok URL 예: "https://your-ngrok-url.ngrok-free.app"
# 로컬 서버 예: "http://127.0.0.1:8000"
SERVER_URL = "https://publicly-capable-monkfish.ngrok-free.app/"    # static url

# API 엔드포인트
TEXT_TO_IMAGE_ENDPOINT = f"{SERVER_URL}/generate/text-to-image"

# 요청 데이터 (텍스트-이미지 생성용)
payload = {
    "prompt": "A futuristic cityscape with flying cars and neon lights",
    "negative_prompt": "(worst quality, low quality, normal quality:1.2), deformed, blurry, text, signature",
    "num_inference_steps": 30,
    "guidance_scale": 3.5,
    "lora_scale": 0.0,  # LoRA 사용 시 조정 가능
    "seed": 42  # 고정된 시드 값으로 재현 가능
}

try:
    # API 요청
    print("🚀 텍스트-이미지 생성 요청 중...")
    response = requests.post(TEXT_TO_IMAGE_ENDPOINT, json=payload, timeout=300)

    # 응답 처리
    if response.status_code == 200:
        print("✅ 이미지 생성 성공!")
        # 응답 데이터를 이미지로 변환
        image = Image.open(io.BytesIO(response.content))
        # 이미지 표시
        image.show()
        # 이미지 저장 (선택 사항)
        image.save("generated_image.png")
        print("💾 생성된 이미지를 'generated_image.png'로 저장했습니다.")
    else:
        print(f"❌ 이미지 생성 실패: {response.status_code}")
        try:
            print(f"오류 메시지: {response.json()}")
        except Exception:
            print(f"오류 내용: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"❌ 요청 실패: {e}")