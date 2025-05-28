import torch
try:
    print(f"PyTorch 버전: {torch.__version__}") # torch 버전도 함께 확인
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 사용 가능 여부 (torch.cuda.is_available()): {cuda_available}")

    if cuda_available:
        print(f"PyTorch 빌드 시 사용된 CUDA 버전: {torch.version.cuda}") # PyTorch가 인식하는 CUDA 버전
        print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            current_gpu_id = torch.cuda.current_device()
            print(f"현재 선택된 GPU ID: {current_gpu_id}")
            print(f"현재 GPU 이름: {torch.cuda.get_device_name(current_gpu_id)}")
    else:
        print("PyTorch에서 CUDA를 사용할 수 없습니다.")
except Exception as e:
    print(f"PyTorch CUDA 점검 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()