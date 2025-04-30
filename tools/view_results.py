#!/usr/bin/env python
# tools/view_results.py
import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리를 sys.path에 추가
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# 앱 모듈 임포트
from app.utils.result_viewer import main

if __name__ == "__main__":
    # 결과 뷰어 실행
    main()
