print("스크립트 실행 시도 중...")
try:
    print("tensorrt_llm 로거 설정 및 임포트 시도 중...")
    # 로거 설정이 임포트 전에 필요한 경우도 있으므로 먼저 시도
    from tensorrt_llm.logger import set_level
    set_level('verbose')
    print("TensorRT-LLM 로거 레벨을 'verbose'로 설정했습니다.")

    from tensorrt_llm import LLM, SamplingParams
    print("tensorrt_llm 관련 모듈 임포트 성공.")

except Exception as e:
    print(f"TensorRT-LLM 임포트 또는 로거 설정 중 심각한 오류 발생: {e}")
    import traceback
    traceback.print_exc() # 전체 오류 트레이스백 출력
    exit() # 임포트 실패 시 즉시 프로그램 종료

# --- 이전에 안내드린 main() 함수 및 나머지 디버깅 코드 ---
# main() 함수 내의 LLM() 및 generate() 호출도 try-except로 감싸주세요.
def main():
    print("main() 함수 호출됨.")
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    print("프롬프트 및 샘플링 파라미터 정의 완료.")

    print("LLM 객체 초기화 시도 중...")
    llm = None
    try:
        llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("LLM 객체 초기화 완료.")
    except Exception as e:
        print(f"LLM 초기화 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

    print("텍스트 생성 시도 중...")
    outputs = None
    try:
        outputs = llm.generate(prompts, sampling_params)
        print("텍스트 생성 완료.")
    except Exception as e:
        print(f"텍스트 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

    if not outputs:
        print("생성된 출력이 없습니다.")
        return

    print("생성된 텍스트 출력:")
    for i, output in enumerate(outputs):
        prompt = output.prompt
        if output.outputs: # output.outputs 리스트가 비어있지 않은지 확인
            generated_text = output.outputs[0].text
            print(f"출력 {i+1}:")
            print(f"  프롬프트: {prompt!r}")
            print(f"  생성된 텍스트: {generated_text!r}")
        else:
            print(f"출력 {i+1}:")
            print(f"  프롬프트: {prompt!r}")
            print(f"  생성된 텍스트 없음.")


    print("스크립트 실행 완료.")

if __name__ == '__main__':
    main()