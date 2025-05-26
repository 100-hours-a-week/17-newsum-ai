# tests/run_tests.py
import subprocess
import sys
import os

def run_tests():
    """PostgreSQL 테스트 실행"""
    print("🔬 PostgreSQL 테스트 시작...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_postgresql_service.py",
            "-v", "--tb=short"
        ], cwd=os.path.dirname(os.path.dirname(__file__)))
        
        if result.returncode == 0:
            print("✅ 테스트 성공!")
        else:
            print("❌ 테스트 실패!")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
