from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def analyze_turn(text: str, relationship_context: dict[str, Any]) -> dict[str, Any]:
    tags: list[str] = []
    reason_parts: list[str] = []

    if any(word in text for word in ["第一次", "纪念日", "里程碑"]):
        tags.append("milestone")
        reason_parts.append("识别到关系里程碑")
    if any(word in text for word in ["其实", "从来没告诉", "只有你", "崩溃", "哭"]):
        tags.append("vulnerable")
        reason_parts.append("识别到脆弱表达")
    if any(word in text for word in ["答应", "约定", "下次", "以后"]):
        tags.append("commitment")
        reason_parts.append("识别到未完成承诺")
    if any(word in text for word in ["谢谢", "开心", "成功", "庆祝"]):
        tags.append("celebration")
        reason_parts.append("识别到共同庆祝")

    stage = relationship_context.get("stage")
    importance = 0.88 if tags else 0.35
    if stage in {"INTENSIFYING", "INTEGRATING", "BONDING"}:
        importance = min(1.0, importance + 0.08)

    return {
        "tags": tags,
        "importance": importance,
        "memory_type": memory_type_from_tags(tags),
        "context_tag": context_tag_from_tags(tags),
        "reason": "external memory AI worker: " + ("；".join(reason_parts) if reason_parts else "普通信息，低关系权重"),
    }


def summarize_story(event_contents: list[str], relationship_context: dict[str, Any]) -> dict[str, str]:
    themes = relationship_context.get("themes", [])
    lead = f"{'、'.join(themes[:2])}：" if themes else ""
    selected = event_contents[-4:]
    return {"summary": lead + "；".join(selected)}


def evaluate_memory_value(content: str, relationship_context: dict[str, Any]) -> dict[str, float | str]:
    value = 0.25
    if any(word in content for word in ["第一次", "纪念日", "里程碑", "答应", "约定"]):
        value += 0.40
    if any(word in content for word in ["只有你", "从来没告诉", "崩溃", "哭", "谢谢", "开心"]):
        value += 0.30
    if relationship_context.get("stage") in {"INTEGRATING", "BONDING"}:
        value += 0.10
    return {"value": min(1.0, value), "reason": "external memory AI worker value estimate"}


def assess_time_conflict(candidate: dict[str, Any], relationship_context: dict[str, Any]) -> dict[str, Any]:
    relevance = float(candidate.get("relevance", 0.0) or 0.0)
    gap_days = float(candidate.get("gap_days", 0.0) or 0.0)
    first = dict(candidate.get("memory") or {})
    second = dict(candidate.get("conflicting_memory") or {})
    same_type = first.get("memory_type") == second.get("memory_type")
    is_conflict = relevance >= 0.55 and gap_days >= 1.0 and same_type
    return {
        "is_conflict": is_conflict,
        "confidence": min(0.92, 0.48 + relevance * 0.34 + min(gap_days, 30.0) / 30.0 * 0.10),
        "reason": "external memory AI worker time conflict assessment",
        "clarification_question": "这两条记忆像是在说同一件事，但时间不同，需要确认哪一个时间更准确吗？",
    }


def memory_type_from_tags(tags: list[str]) -> str | None:
    if "milestone" in tags:
        return "milestone"
    if "commitment" in tags:
        return "commitment"
    if "vulnerable" in tags or "celebration" in tags:
        return "emotional_moment"
    return None


def context_tag_from_tags(tags: list[str]) -> str | None:
    if "milestone" in tags:
        return "MILESTONE"
    if "commitment" in tags:
        return "UNRESOLVED_THREAD"
    if "vulnerable" in tags:
        return "VULNERABLE_MOMENT"
    if "celebration" in tags:
        return "SHARED_CELEBRATION"
    return None


class MemoryAIHandler(BaseHTTPRequestHandler):
    server_version = "MemoryAIWorker/0.1"

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.route(payload)
            self.send_json(200, result)
        except Exception as exc:
            self.send_json(400, {"error": str(exc)})

    def route(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = payload.get("task")
        if task == "analyze_turn":
            return analyze_turn(str(payload.get("text", "")), dict(payload.get("relationship_context") or {}))
        if task == "summarize_story":
            contents = [str(item) for item in payload.get("event_contents", [])]
            return summarize_story(contents, dict(payload.get("relationship_context") or {}))
        if task == "evaluate_memory_value":
            return evaluate_memory_value(str(payload.get("content", "")), dict(payload.get("relationship_context") or {}))
        if task == "assess_time_conflict":
            return assess_time_conflict(dict(payload.get("candidate") or {}), dict(payload.get("relationship_context") or {}))
        raise ValueError(f"unsupported task: {task}")

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Example external AI worker for the friend memory project.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MemoryAIHandler)
    print(f"memory AI worker listening on http://{args.host}:{args.port}/memory-ai")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
