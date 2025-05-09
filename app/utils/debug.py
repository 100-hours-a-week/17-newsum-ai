def print_final_state_debug(state, result_dict=None):
    print("\n========== [최종 워크플로우 상태 DEBUG] ==========")
    print(f"[current_stage] : {getattr(state, 'current_stage', None)}")
    print(f"[comic_id] : {getattr(state, 'comic_id', None)}")
    print(f"[trace_id] : {getattr(state, 'trace_id', None)}")
    print(f"[selected_comic_idea_for_scenario] : {getattr(state, 'selected_comic_idea_for_scenario', None)}")
    print(f"[thumbnail_image_prompt] : {getattr(state, 'thumbnail_image_prompt', None)}")
    print(f"[uploaded_image_urls] : {getattr(state, 'uploaded_image_urls', None)}")
    print(f"[uploaded_report_s3_uri] : {getattr(state, 'uploaded_report_s3_uri', None)}")
    print(f"[translated_report_content] : {getattr(state, 'translated_report_content', None)}")
    print(f"[referenced_urls] : {getattr(state, 'referenced_urls', None)}")
    print(f"[external_api_response] : {getattr(state, 'external_api_response', None)}")
    print(f"[error_log] :")
    error_log = getattr(state, 'error_log', [])
    if not error_log:
        print("  (에러 없음)")
    else:
        for err in error_log:
            print(f"  - Stage: {err.get('stage')}, Error: {err.get('error')}")
    # slides 정보도 출력
    slides = None
    if result_dict:
        slides = result_dict.get('slides')
    if not slides and hasattr(state, 'slides'):
        slides = getattr(state, 'slides')
    if slides:
        print("[slides (실제 전송)] :")
        for slide in slides:
            print(f"  - slideSeq: {slide.get('slideSeq')}, imageUrl: {slide.get('imageUrl')}, content: {slide.get('content')}")
    print("==============================================\n") 