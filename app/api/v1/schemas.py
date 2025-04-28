# app/api/v1/schemas.py

from pydantic import BaseModel
from typing import List, Optional

class ComicRequest(BaseModel):
    query: str
    language: Optional[str] = "ko"

class ComicResponse(BaseModel):
    comic_id: str
