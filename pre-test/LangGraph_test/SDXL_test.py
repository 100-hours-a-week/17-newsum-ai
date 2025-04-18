import requests
import json
import base64
from PIL import Image # 이미지 표시/저장용
import io
import os
import time

# --- 테스트 설정 ---

# 1. 접속할 API 서버의 주소를 입력하세요.
#    (ngrok/cloudflared 사용 시 해당 URL, 로컬 실행 시 http://127.0.0.1:포트번호)
target_base_url = "https://172f-35-247-173-221.ngrok-free.app/"  # <--- 여기를 서버 주소로 변경하세요!
# 예: target_base_url = "https://your-random-name.trycloudflare.com"
# 예: target_base_url = "https://your-random-id.ngrok.io"

# 2. LoRA 로딩/언로딩 테스트에 사용할 LoRA 파일의 *로컬* 경로를 입력하세요.
#    이 파일이 로컬 컴퓨터에 실제로 존재해야 합니다. 비워두면 LoRA 테스트 건너<0xEB><0x9C><0x8D>니다.
TEST_LORA_PATH = "" # 예: "C:/Users/YourUser/Documents/Loras/MyLora.safetensors" # <--- 실제 로컬 경로로 변경!
# 또는 Linux/MacOS: "/home/youruser/loras/my_lora.safetensors"

# 3. 테스트 결과(로그, 응답 JSON, 이미지)를 저장할지 여부 설정
SAVE_RESULTS = True
# 결과 저장 위치 (스크립트 실행 위치 기준)
OUTPUT_BASE_DIR = "./FastAPITestResults_local"

# --- 설정값 기반 URL 및 디렉토리 준비 ---
print(f"API Test Target URL: {target_base_url}")
print(f"Test LoRA Path: {TEST_LORA_PATH if TEST_LORA_PATH else 'N/A'}")

output_dir = None
if SAVE_RESULTS:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE_DIR, f"test_{timestamp}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        SAVE_RESULTS = False

if TEST_LORA_PATH and not os.path.exists(TEST_LORA_PATH):
    print(f"경고: 테스트 LoRA 경로가 존재하지 않습니다 ({TEST_LORA_PATH}).")
    if SAVE_RESULTS: print("   LoRA 관련 테스트 결과가 불완전할 수 있습니다.")

print("-" * 50)

# API 엔드포인트 URL
status_url = f"{target_base_url}/status"
predictions_url = f"{target_base_url}/predictions"
load_lora_url = f"{target_base_url}/load-lora"
unload_lora_url = f"{target_base_url}/unload-lora"

# 이미지 저장 및 표시 함수 (로컬용)
def save_and_show_image(base64_string, filename_base, title="Generated Image"):
    """Decode, save, and show image using default viewer."""
    global output_dir, SAVE_RESULTS
    try:
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        print(f"\n🖼️ --- {title} ---")
        try:
            img.show() # 로컬 이미지 뷰어 실행
        except Exception as show_e:
            print(f"   (Image viewer not available or failed: {show_e})")

        if SAVE_RESULTS and output_dir:
            img_path = os.path.join(output_dir, f"{filename_base}.png")
            try:
                img.save(img_path, "PNG")
                print(f"   Image saved to: {img_path}")
            except Exception as e: print(f"   Error saving image {img_path}: {e}")
    except Exception as e: print(f"Error decoding/saving/showing image: {e}")

# 결과 로깅 함수 (이전과 동일)
def log_result(test_name, success, message="", response=None):
    global output_dir, SAVE_RESULTS; status = "✅ SUCCESS" if success else "❌ FAILED"; log_entry = f"{test_name}: {status} - {message}\n"; print(log_entry.strip())
    if SAVE_RESULTS and output_dir:
        summary_file = os.path.join(output_dir, "results_summary.log"); response_file = None
        try:
            with open(summary_file, "a", encoding="utf-8") as f: f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {log_entry}")
            if response is not None:
                response_filename = f"{test_name.lower().replace('[','').replace(']','').replace(':','').replace(' ', '_')}_response.json"; response_file = os.path.join(output_dir, response_filename)
                try: response_json = response.json()
                except json.JSONDecodeError: response_json = {"raw_content": response.text}
                with open(response_file, "w", encoding="utf-8") as f: json.dump(response_json, f, indent=2, ensure_ascii=False); print(f"   Response saved to: {response_file}")
        except Exception as e: print(f"   Error writing to log/response file: {e}")

# =================================================
# 테스트 실행
# =================================================
headers = {'Content-Type': 'application/json'}
initial_lora = None

# --- 테스트 1: 상태 조회 ---
print("\n🧪 === [Test 1] Server Status Check ===")
test_name = "[Test 1] Status Check"
try:
    response_1 = requests.get(status_url, timeout=30); response_1.raise_for_status(); result_json = response_1.json(); initial_lora = result_json.get("current_lora_path")
    log_result(test_name, True, f"Status Code: {response_1.status_code}", response=response_1); print(json.dumps(result_json, indent=2, ensure_ascii=False))
except Exception as e: log_result(test_name, False, str(e))

# --- 테스트 2: 기본 추론 ---
print("\n🧪 === [Test 2] Basic Inference ===")
test_name = "[Test 2] Basic Inference"
payload_infer = {"prompt": "masterpiece, best quality, highly detailed illustration of a magical forest landscape at sunset, fantasy art", "negative_prompt": "worst quality, low quality, blurry, text, signature, watermark, username, deformed", "steps": 28, "cfg_scale": 7.0, "seed": 2025, "width": 1024, "height": 1024}
print(f"Request Payload:\n{json.dumps(payload_infer, indent=2)}")
try:
    response_2 = requests.post(predictions_url, headers=headers, json=payload_infer, timeout=400); response_2.raise_for_status(); result = response_2.json()
    log_result(test_name, True, f"Generation Time: {result.get('generation_time_ms')}ms, LoRA: {result.get('current_lora')}", response=response_2)
    if "image_base64" in result: save_and_show_image(result["image_base64"], "test2_basic_inference", "Basic Inference Image")
    else: print("  - No image in response.")
except Exception as e: log_result(test_name, False, str(e))

# --- 테스트 3: LoRA 로드 ---
lora_base_name = os.path.basename(TEST_LORA_PATH) if TEST_LORA_PATH else "N/A"
print(f"\n🧪 === [Test 3] Load Test LoRA ({lora_base_name}) ===")
test_name = f"[Test 3] Load LoRA {lora_base_name}"
# TEST_LORA_PATH 가 설정되어 있고, 해당 파일이 로컬에 존재하는 경우에만 실행
if TEST_LORA_PATH and os.path.exists(TEST_LORA_PATH):
    payload_load = {"lora_path": TEST_LORA_PATH}; print(f"Request Payload:\n{json.dumps(payload_load, indent=2)}")
    try:
        response_3 = requests.post(load_lora_url, headers=headers, json=payload_load, timeout=120); response_3.raise_for_status()
        log_result(test_name, True, f"Status Code: {response_3.status_code}", response=response_3); print(json.dumps(response_3.json(), indent=2, ensure_ascii=False)); time.sleep(2)
        status_resp_after_load = requests.get(status_url, timeout=30); print(f"Status after Load LoRA:\n{json.dumps(status_resp_after_load.json(), indent=2, ensure_ascii=False)}")
    except Exception as e: log_result(test_name, False, str(e))
else:
    if not TEST_LORA_PATH: print("ℹ️ Skipping Test 3: TEST_LORA_PATH not set.")
    else: print(f"ℹ️ Skipping Test 3: Test LoRA path not found: {TEST_LORA_PATH}")
    log_result(test_name, False, "Skipped - LoRA path invalid or not set")


# --- 테스트 4: LoRA 적용 후 추론 ---
print(f"\n🧪 === [Test 4] Inference with Test LoRA ({lora_base_name}) ===")
test_name = f"[Test 4] Inference with LoRA {lora_base_name}"
# TEST_LORA_PATH 가 설정되어 있고, 로컬에 존재하는 경우에만 실행 (Test 3 성공 여부와는 별개로 경로만 확인)
if TEST_LORA_PATH and os.path.exists(TEST_LORA_PATH):
    print(f"Request Payload:\n{json.dumps(payload_infer, indent=2)} (Same as Test 2)")
    try:
        response_4 = requests.post(predictions_url, headers=headers, json=payload_infer, timeout=400); response_4.raise_for_status(); result = response_4.json()
        log_result(test_name, True, f"Generation Time: {result.get('generation_time_ms')}ms, LoRA: {result.get('current_lora')}", response=response_4)
        if "image_base64" in result: save_and_show_image(result["image_base64"], f"test4_inference_{lora_base_name.replace('.safetensors','').replace('.pt','')}", f"Inference Image with {lora_base_name}")
        else: print("  - No image in response.")
    except Exception as e: log_result(test_name, False, str(e))
else:
    if not TEST_LORA_PATH: print("ℹ️ Skipping Test 4: TEST_LORA_PATH not set.")
    else: print(f"ℹ️ Skipping Test 4: Test LoRA path not found: {TEST_LORA_PATH}")
    log_result(test_name, False, "Skipped - LoRA path invalid or not set")


# --- 테스트 5: LoRA 언로드 ---
print("\n🧪 === [Test 5] Unload LoRA ===")
test_name = "[Test 5] Unload LoRA"
try:
    response_5 = requests.post(unload_lora_url, headers=headers, timeout=60); response_5.raise_for_status()
    log_result(test_name, True, f"Status Code: {response_5.status_code}", response=response_5); print(json.dumps(response_5.json(), indent=2, ensure_ascii=False)); time.sleep(2)
    status_resp_after_unload = requests.get(status_url, timeout=30); print(f"Status after Unload LoRA:\n{json.dumps(status_resp_after_unload.json(), indent=2, ensure_ascii=False)}")
except Exception as e: log_result(test_name, False, str(e))

# --- 테스트 6: LoRA 언로드 후 추론 ---
print("\n🧪 === [Test 6] Inference after Unload ===")
test_name = "[Test 6] Inference after Unload"
print(f"Request Payload:\n{json.dumps(payload_infer, indent=2)} (Same as Test 2)")
try:
    response_6 = requests.post(predictions_url, headers=headers, json=payload_infer, timeout=400); response_6.raise_for_status(); result = response_6.json()
    log_result(test_name, True, f"Generation Time: {result.get('generation_time_ms')}ms, LoRA: {result.get('current_lora')}", response=response_6)
    if "image_base64" in result: save_and_show_image(result["image_base64"], "test6_unloaded_inference", "Inference Image after Unload")
    else: print("  - No image in response.")
except Exception as e: log_result(test_name, False, str(e))

print("\n" + "="*50)
print("✅ API 기능 테스트 완료.")
if SAVE_RESULTS and output_dir: print(f"📄 테스트 결과가 '{output_dir}' 에 저장되었습니다.")
print("="*50)