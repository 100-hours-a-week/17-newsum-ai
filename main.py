# main.py
from fastapi import FastAPI
from app.api.v1.endpoints import router as api_router

app = FastAPI()

app.include_router(api_router, prefix="/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
