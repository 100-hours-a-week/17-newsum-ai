from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import db_utils
import uvicorn
import os
from dotenv import load_dotenv
from typing import Optional
import uuid

load_dotenv()

app = FastAPI()

class AiResponse(BaseModel):
    request_id: uuid.UUID
    content: str
    # 필요시 추가 필드 정의 (e.g., error_message, status)

@app.post("/receive_response")
async def receive_response(response: AiResponse):
    """AI 서버로부터 응답을 받아 DB에 저장합니다."""
    print(f"Received AI response for request_id: {response.request_id}")
    success = db_utils.update_loading_message(response.request_id, response.content)

    if success:
        print(f"Successfully updated message for request_id: {response.request_id}")
        return {"status": "success", "message": "Response processed."}
    else:
        print(f"Failed to find or update loading message for request_id: {response.request_id}")
        # 실패 시 처리 로직 (e.g., 별도 로그, 재시도 알림 등)
        # 여기서는 404 에러를 반환하여 AI 서버나 모니터링 시스템이 인지하도록 함
        raise HTTPException(status_code=404, detail="Loading message not found or already processed.")

if __name__ == "__main__":
    port = int(os.getenv("RECEPTION_API_PORT", 9090))
    uvicorn.run(app, host="0.0.0.0", port=port)