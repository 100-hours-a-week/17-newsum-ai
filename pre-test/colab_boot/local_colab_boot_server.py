# local_colab_boot_server.py

import time, sys
import uvicorn  # uvicorn import 추가
from fastapi import FastAPI, BackgroundTasks
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

app = FastAPI()

def run_colab_with_selenium():
    # --- Selenium 설정 ---
    chrome_options = Options()
    # 매번 로그인하는 것을 피하기 위해 기존 Chrome 프로필 사용 (경로는 실제 환경에 맞게 수정)
    chrome_options.add_argument("user-data-dir=C:\\Users\\xodnr\\AppData\\Local\\Google\\Chrome\\User Data\\Default")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # 1. Colab 노트북 열기
        colab_url = "https://colab.research.google.com/drive/12Ivw_zJZ4Zk2tx9gNxzgK-LOCadM_g7M#scrollTo=2R0izjLyvHfL"
        driver.get(colab_url)
        print("Colab 노트북을 여는 중... 20초 대기")
        time.sleep(300)  # 페이지 및 내부 스크립트가 완전히 로드될 때까지 충분히 대기

        # 2. ActionChains 객체 생성
        actions = ActionChains(driver)

        # 3. 운영체제에 맞는 단축키 설정
        # sys.platform이 'darwin'이면 macOS, 그렇지 않으면 Windows/Linux로 간주
        modifier_key = Keys.COMMAND if sys.platform == 'darwin' else Keys.CONTROL

        print(f"'{modifier_key.name}' + 'F9' 단축키를 입력하여 '모두 실행'을 시작합니다.")

        # 4. 단축키 시뮬레이션
        actions.key_down(modifier_key)  # Ctrl 또는 Command 키를 누름
        actions.send_keys(Keys.F9)  # F9 키를 입력
        actions.key_up(modifier_key)  # 눌렀던 Ctrl 또는 Command 키를 뗌

        # 5. 액션 실행
        actions.perform()

        print("'모두 실행'이 시작되었습니다. 셀 실행 완료까지 대기합니다.")
        # 실제 셀들이 모두 실행될 때까지 대기하는 로직 추가 필요
        time.sleep(120)

    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        print("드라이버를 종료합니다.")
        driver.quit()

@app.get("/")
async def boot_server_on():
    """이미지 생성 요청을 받아 백그라운드에서 Selenium 작업을 실행합니다."""
    return {"message": "boot server on"}

@app.get("/boot")
async def generate_image(background_tasks: BackgroundTasks):
    """이미지 생성 요청을 받아 백그라운드에서 Selenium 작업을 실행합니다."""
    background_tasks.add_task(run_colab_with_selenium)
    return {"message": "Image generation started in the background."}

# --- 서버 실행을 위한 main 함수 추가 ---
if __name__ == "__main__":
    # uvicorn.run("파일이름:app객체이름", host="호스트주소", port=포트번호, reload=코드변경시자동재시작)
    uvicorn.run("local_colab_boot_server:app", host="0.0.0.0", port=9741, reload=False)