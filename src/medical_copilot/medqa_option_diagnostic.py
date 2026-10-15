"""Post hoc deterministic option extraction; never changes primary graph records."""
import json


def visible_option(record: dict) -> dict | None:
    if record["status"] == "completed":
        return {"selected_option": record["answer"]["selected_option"], "justification": record["answer"]["justification"], "basis": "Primary validated answer"}
    calls = record.get("model_calls", [])
    if not calls or calls[-1]["stage"] != "final_answer":
        return None
    call = calls[-1]
    if call.get("status") != "ok" or call.get("response_metadata", {}).get("finish_reason") != "stop":
        return None
    text = call.get("response", "")
    candidates = []
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("selected_option") in ("A", "B", "C", "D") and isinstance(value.get("justification"), str):
            candidates.append(value)
    if len(candidates) != 1:
        return None
    value = candidates[0]
    return {"selected_option": value["selected_option"], "justification": value["justification"], "basis": "Post hoc single explicit JSON object in a normally stopped response with extra prose; primary format error remains unchanged"}
