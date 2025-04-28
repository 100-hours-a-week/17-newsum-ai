# main.py 예시
from fastapi import FastAPI
# api 폴더 및 v1 폴더 내의 endpoints 모듈에서 router 객체를 가져옵니다.
from app.api.v1.endpoints import router as v1_router
# 필요시 다른 설정이나 미들웨어 추가
# from app.config.settings import settings # 예시

# FastAPI 앱 인스턴스 생성
app = FastAPI(title="Comic Generation API", version="0.1.0")

# --- 중요: V1 라우터 포함 ---
# endpoints.py 에서 정의한 router 객체(v1_router)를
# FastAPI 앱(app)에 포함시킵니다.
# prefix="/v1" 은 해당 라우터의 모든 경로 앞에 /v1 을 붙여줍니다.
# 이 prefix는 endpoints.py 파일 내 router 생성 시 정의된 prefix와 중첩되지 않도록 주의합니다.
# 만약 endpoints.py 에서 APIRouter(prefix="/v1") 로 정의했다면, 여기서는 prefix를 생략해야 합니다.
# 여기서는 endpoints.py 에 prefix가 있다고 가정하고 아래처럼 포함합니다.
app.include_router(v1_router) # endpoints.py에 prefix="/v1" 가 있다면 이것으로 충분

# 만약 endpoints.py에 prefix가 없다면 아래처럼 포함 시 prefix 지정
# app.include_router(v1_router, prefix="/v1")


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Comic Generation API!"}

# uvicorn 실행을 위한 부분 (선택 사항, 터미널에서 직접 실행 가능)
if __name__ == "__main__":
    import uvicorn
    # host, port, reload 등 설정은 uvicorn 명령어 옵션으로 주는 것이 일반적
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)