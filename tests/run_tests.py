import os
import sys
import unittest
import asyncio

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_tests():
    """Discover and run all tests"""
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

def run_single_test(test_path):
    """Run a specific test file"""
    if not os.path.exists(test_path):
        print(f"Test file not found: {test_path}")
        return
    
    # Import the test file
    sys.path.append(os.path.dirname(test_path))
    test_name = os.path.basename(test_path)
    test_module = test_name.replace('.py', '')
    
    try:
        # 수정: 이벤트 루프 생성 방식 업데이트
        try:
            # Python 3.10 이상에서 권장되는 방식
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 실행 중인 루프가 없는 경우
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        module = __import__(test_module)
        unittest.main(module=module)
    except ImportError as e:
        print(f"Error importing test module: {e}")

if __name__ == "__main__":
    # 수정: 이벤트 루프 생성 방식 업데이트
    try:
        # Python 3.10 이상에서 권장되는 방식
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 실행 중인 루프가 없는 경우
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        run_single_test(test_path)
    else:
        run_tests()
