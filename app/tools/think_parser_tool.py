import re, json

def extract_json(text: str) -> dict:
    # <think>...</think> 제거
    clean = re.sub(r"</?think>", "", text)
    m = re.search(r"(\{[\s\S]*\})", clean)
    return json.loads(m.group(1)) if m else {}