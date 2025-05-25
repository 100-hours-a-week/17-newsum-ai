import re, json

def extract_json(text: str) -> dict:
    clean = re.sub(r"</?think>", "", text)
    m = re.search(r"(\{[\s\S]*\})", clean)
    return json.loads(m.group(1)) if m else {}

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
