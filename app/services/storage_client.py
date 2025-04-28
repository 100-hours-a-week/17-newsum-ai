# app/services/storage_client.py

import boto3
from app.config.settings import settings
from uuid import uuid4

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key
)

def upload_file_to_s3(file_path: str, bucket_name: str = None) -> str:
    """
    파일을 S3에 업로드하고 S3 URL을 반환하는 함수.
    """
    bucket = bucket_name or settings.s3_bucket_name
    object_key = f"comics/{uuid4().hex}.png"
    
    s3_client.upload_file(file_path, bucket, object_key)
    
    s3_url = f"https://{bucket}.s3.amazonaws.com/{object_key}"
    return s3_url
