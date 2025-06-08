import re, json
from typing import Any, Optional

# def extract_json(text: str) -> dict:
#     clean = re.sub(r"</?think>", "", text)
#     m = re.search(r"(\{[\s\S]*\})", clean)
#     return json.loads(m.group(1)) if m else {}

def remove_think_tags(text: Optional[str]) -> str:
    """
    주어진 텍스트에서 모든 <think> 태그와 그 내용을 제거합니다.

    Args:
        text: 처리할 원본 문자열. <think> 태그를 포함할 수 있습니다.

    Returns:
        <think> 태그와 그 내용이 제거된 문자열. 입력이 None이면 빈 문자열을 반환합니다.
    """
    if not text:
        return ""
    # <think>...</think> 패턴을 찾아 제거 (대소문자 구분 없음, 여러 줄 가능, 비탐욕적 매칭)
    cleaned_text = re.sub(r"<think>(.*?)</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip()


def extract_json(text_input: Optional[str]) -> Optional[Any]:
    """
    주어진 텍스트 입력에서 <think> 태그를 제거한 후, 유효한 JSON 객체를 추출합니다.
    LLM 응답에 JSON 외의 설명이나 주석이 포함된 경우에도 첫 번째 유효한 JSON을 찾으려고 시도합니다.

    Args:
        text_input: LLM 응답과 같이 JSON을 포함할 수 있는 문자열.

    Returns:
        파싱된 JSON 객체 (dict 또는 list) 또는 추출/파싱 실패 시 None.
    """
    if not text_input:
        return None

    # 1. <think> 태그와 그 내용 제거
    cleaned_text = remove_think_tags(text_input)

    # 2. 정리된 텍스트에서 JSON 객체 또는 배열 찾기 시도
    # 가장 바깥쪽 중괄호 {} 또는 대괄호 []를 기준으로 탐색

    # 중괄호로 시작하는 JSON 객체 시도
    obj_match = re.search(r"^\s*(\{.*?\})\s*$", cleaned_text, re.DOTALL)
    if obj_match:
        json_str = obj_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 전체가 완벽한 JSON이 아니면, 내부에서 첫번째 중괄호 블록을 찾으려는 시도
            pass  # 아래의 더 일반적인 검색으로 넘어감

    # 대괄호로 시작하는 JSON 배열 시도
    arr_match = re.search(r"^\s*(\[.*?\])\s*$", cleaned_text, re.DOTALL)
    if arr_match:
        json_str = arr_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # 아래의 더 일반적인 검색으로 넘어감

    # 위의 경우가 실패하면, 문자열 내에 포함된 첫번째 JSON 구조를 찾으려는 일반적인 시도
    # (주의: 이 방식은 복잡한 문자열에서 완벽하지 않을 수 있음)
    # 보통 LLM은 JSON을 코드 블록 등으로 감싸서 주기도 하므로, 그 패턴도 고려할 수 있음
    # 예: ```json\n{...}\n```
    code_block_match = re.search(r"```json\n(\{.*?\})\n```", cleaned_text, re.DOTALL)
    if not code_block_match:  # 일반 코드 블록도 확인
        code_block_match = re.search(r"```\n(\{.*?\})\n```", cleaned_text, re.DOTALL)

    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        # 가장 처음 나타나는 '{' 또는 '[' 부터 가장 마지막 '}' 또는 ']' 까지를 잠재적 JSON으로 간주
        # 이 방식은 간단하지만, 문자열 내에 여러 JSON 유사 구조가 있을 경우 부정확할 수 있음
        first_brace = cleaned_text.find('{')
        first_bracket = cleaned_text.find('[')

        start_index = -1

        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start_char = '{'
            end_char = '}'
            start_index = first_brace
        elif first_bracket != -1:
            start_char = '['
            end_char = ']'
            start_index = first_bracket
        else:
            # JSON 시작 문자를 찾지 못함
            # logger.warning(f"JSON 시작 문자 ('{{' 또는 '[')를 찾지 못했습니다. 원본(정리 후): {cleaned_text[:500]}")
            return None

        if start_index != -1:
            # 해당 괄호의 짝을 찾아 JSON 문자열 경계 설정 (간단한 방식, 중첩 구조에 취약할 수 있음)
            # 더 견고한 방식은 괄호 카운팅이지만, LLM 응답이 보통 단일 JSON 객체를 잘 생성한다고 가정
            # 여기서는 첫 시작 괄호에 대응하는 마지막 괄호를 찾는 매우 단순한 방법을 사용하지 않고,
            # LLM이 JSON을 비교적 잘 생성한다는 가정하에, 첫 시작부터 끝까지를 잠재적 JSON으로 봄.
            # 실제로는 더 정교한 파서가 필요할 수 있음.
            # 여기서는 일반성을 위해 첫번째 발견된 '{...}' 또는 '[...]' 구조를 시도
            potential_json_str = cleaned_text[start_index:]
            # 가장 마지막 닫는 괄호 찾기
            last_closing_brace = potential_json_str.rfind('}')
            last_closing_bracket = potential_json_str.rfind(']')

            if start_char == '{' and last_closing_brace != -1:
                json_str = potential_json_str[:last_closing_brace + 1]
            elif start_char == '[' and last_closing_bracket != -1:
                json_str = potential_json_str[:last_closing_bracket + 1]
            else:
                # logger.warning(f"JSON의 닫는 괄호를 찾지 못했습니다. 원본(정리 후): {cleaned_text[:500]}")
                return None  # 닫는 괄호 못 찾으면 파싱 불가

    try:
        # 최종적으로 추출된 문자열을 JSON으로 파싱
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # logger.error(f"JSON 최종 파싱 실패: {e}. 대상 문자열: {json_str[:500]}", exc_info=True)
        # 디버깅을 위해 예외를 다시 발생시킬 수도 있음
        # raise # 또는 None 반환으로 실패 처리
        return None  # 파싱 실패 시 None 반환

def extract_json_all(text: str) -> list:
    clean = re.sub(r"</?think>", "", text)
    m = re.search(r"\[\s*{[\s\S]*?}\s*]", clean)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return [normalize_plan(entry) for entry in data if isinstance(entry, dict)]
        except json.JSONDecodeError:
            pass
    return []

def normalize_plan(plan: dict) -> dict:
    def flatten(val):
        # list of list 처리
        if isinstance(val, list):
            flattened = []
            for item in val:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            return flattened
        return val

    plan["max_results"] = plan.get("max_results", 5)
    if isinstance(plan["max_results"], list):
        plan["max_results"] = plan["max_results"][0] if plan["max_results"] else 5
    if not isinstance(plan["max_results"], int):
        try:
            plan["max_results"] = int(plan["max_results"])
        except Exception:
            plan["max_results"] = 5

    plan["queries"] = flatten(plan.get("queries", []))
    if not isinstance(plan["queries"], list):
        plan["queries"] = [plan["queries"]] if plan["queries"] else []

    plan["domains"] = flatten(plan.get("domains", []))
    if not isinstance(plan["domains"], list):
        plan["domains"] = [plan["domains"]] if plan["domains"] else []

    return plan
