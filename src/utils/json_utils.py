import json

def extract_json_by_braces(text: str) -> dict:
    if not isinstance(text, str):
        text = str(text)

    start = text.find("{")
    if start < 0:
        raise ValueError("No '{' found in LLM output")

    end = text.rfind("}")
    if end < 0 or end <= start:
        raise ValueError("No valid '}' found in LLM output")

    json_text = text[start:end + 1]
    return json.loads(json_text)