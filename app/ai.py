from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from env_config import load_dotenv


class MemoryAI(Protocol):
    """AI decision surface for the memory system.

    The project core is deterministic, but these methods are the intended AI
    participation points: turn understanding, story synthesis, and value
    re-evaluation during offline consolidation.
    """

    def analyze_turn(self, text: str, relationship_context: dict[str, Any]) -> dict[str, Any]:
        ...

    def summarize_story(self, event_contents: list[str], relationship_context: dict[str, Any]) -> str:
        ...

    def evaluate_memory_value(self, content: str, relationship_context: dict[str, Any]) -> float:
        ...

    def assess_time_conflict(self, candidate: dict[str, Any], relationship_context: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class HeuristicMemoryAI:
    """Local fallback that mimics the AI interface without external services."""

    def analyze_turn(self, text: str, relationship_context: dict[str, Any]) -> dict[str, Any]:
        tags: list[str] = []
        if any(word in text for word in ["第一次", "纪念日"]):
            tags.append("milestone")
        if any(word in text for word in ["其实", "从来没告诉", "只有你", "崩溃", "哭"]):
            tags.append("vulnerable")
        if any(word in text for word in ["答应", "约定", "以后", "下次"]):
            tags.append("commitment")
        if any(word in text for word in ["谢谢", "开心", "成功", "庆祝"]):
            tags.append("celebration")
        importance = 0.75 if tags else 0.35
        return {
            "tags": tags,
            "importance": importance,
            "memory_type": self._memory_type(tags),
            "context_tag": self._context_tag(tags),
            "reason": "heuristic local AI fallback",
        }

    def summarize_story(self, event_contents: list[str], relationship_context: dict[str, Any]) -> str:
        if not event_contents:
            return ""
        if len(event_contents) == 1:
            return event_contents[0]
        themes = relationship_context.get("themes", [])
        prefix = f"{'、'.join(themes[:2])}：" if themes else ""
        return prefix + " / ".join(event_contents[-3:])

    def evaluate_memory_value(self, content: str, relationship_context: dict[str, Any]) -> float:
        value = 0.25
        if any(word in content for word in ["第一次", "纪念日", "答应", "约定", "只有你", "从来没告诉"]):
            value += 0.45
        if any(word in content for word in ["谢谢", "开心", "成功", "崩溃", "哭", "焦虑"]):
            value += 0.20
        if relationship_context.get("stage") in {"INTEGRATING", "BONDING"}:
            value += 0.10
        return min(1.0, value)

    def assess_time_conflict(self, candidate: dict[str, Any], relationship_context: dict[str, Any]) -> dict[str, Any]:
        relevance = float(candidate.get("relevance", 0.0) or 0.0)
        gap_days = float(candidate.get("gap_days", 0.0) or 0.0)
        is_conflict = relevance >= 0.55 and gap_days >= 1.0
        return {
            "is_conflict": is_conflict,
            "confidence": min(0.90, 0.45 + relevance * 0.35 + min(gap_days, 30.0) / 30.0 * 0.10),
            "reason": "heuristic local AI fallback time conflict assessment",
        }

    def _memory_type(self, tags: list[str]) -> str | None:
        if "milestone" in tags:
            return "milestone"
        if "commitment" in tags:
            return "commitment"
        if "vulnerable" in tags or "celebration" in tags:
            return "emotional_moment"
        return None

    def _context_tag(self, tags: list[str]) -> str | None:
        if "milestone" in tags:
            return "MILESTONE"
        if "commitment" in tags:
            return "UNRESOLVED_THREAD"
        if "vulnerable" in tags:
            return "VULNERABLE_MOMENT"
        if "celebration" in tags:
            return "SHARED_CELEBRATION"
        return None


@dataclass
class HttpMemoryAI:
    """Generic HTTP adapter for an external LLM/AI service.

    The endpoint is intentionally generic. It receives JSON with a `task` field
    and should return a JSON object. This avoids coupling the project to one
    provider while still making AI participation explicit.
    """

    endpoint: str
    api_key: str | None = None
    timeout_seconds: float = 10.0

    def analyze_turn(self, text: str, relationship_context: dict[str, Any]) -> dict[str, Any]:
        return self._post({"task": "analyze_turn", "text": text, "relationship_context": relationship_context})

    def summarize_story(self, event_contents: list[str], relationship_context: dict[str, Any]) -> str:
        payload = self._post(
            {"task": "summarize_story", "event_contents": event_contents, "relationship_context": relationship_context}
        )
        return str(payload.get("summary", " / ".join(event_contents[-3:])))

    def evaluate_memory_value(self, content: str, relationship_context: dict[str, Any]) -> float:
        payload = self._post({"task": "evaluate_memory_value", "content": content, "relationship_context": relationship_context})
        try:
            return max(0.0, min(1.0, float(payload.get("value", 0.5))))
        except (TypeError, ValueError):
            return 0.5

    def assess_time_conflict(self, candidate: dict[str, Any], relationship_context: dict[str, Any]) -> dict[str, Any]:
        return self._post(
            {"task": "assess_time_conflict", "candidate": candidate, "relationship_context": relationship_context}
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=data, method="POST")
        request.add_header("content-type", "application/json")
        if self.api_key:
            request.add_header("authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


@dataclass
class OpenAICompatibleMemoryAI:
    """OpenAI-compatible chat completions adapter for real model participation."""

    endpoint: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 20.0

    def analyze_turn(self, text: str, relationship_context: dict[str, Any]) -> dict[str, Any]:
        return self._json_task(
            "analyze_turn",
            {
                "text": text,
                "relationship_context": relationship_context,
                "schema": {
                    "importance": "number from 0 to 1",
                    "memory_type": "one of milestone, emotional_moment, commitment, preference, identity, context_detail, or null",
                    "context_tag": "one supported context tag or null",
                    "tags": "short list of semantic tags",
                    "reason": "brief reason in Chinese",
                },
            },
        )

    def summarize_story(self, event_contents: list[str], relationship_context: dict[str, Any]) -> str:
        payload = self._json_task(
            "summarize_story",
            {
                "event_contents": event_contents,
                "relationship_context": relationship_context,
                "schema": {"summary": "concise shared-story consensus in Chinese"},
            },
        )
        return str(payload.get("summary") or " / ".join(event_contents[-3:]))

    def evaluate_memory_value(self, content: str, relationship_context: dict[str, Any]) -> float:
        payload = self._json_task(
            "evaluate_memory_value",
            {
                "content": content,
                "relationship_context": relationship_context,
                "schema": {"value": "number from 0 to 1", "reason": "brief reason in Chinese"},
            },
        )
        try:
            return max(0.0, min(1.0, float(payload.get("value", 0.5))))
        except (TypeError, ValueError):
            return 0.5

    def assess_time_conflict(self, candidate: dict[str, Any], relationship_context: dict[str, Any]) -> dict[str, Any]:
        return self._json_task(
            "assess_time_conflict",
            {
                "candidate": candidate,
                "relationship_context": relationship_context,
                "schema": {
                    "is_conflict": "boolean",
                    "confidence": "number from 0 to 1",
                    "reason": "brief reason in Chinese",
                },
            },
        )

    def _json_task(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        system = (
            "You are the AI reasoning component for a friend-style relational memory system. "
            "Return only valid JSON. Do not include markdown."
        )
        user = {
            "task": task,
            "instruction": "Use the provided schema and relationship context. Keep reasons short and auditable.",
            **payload,
        }
        response = self._post_chat(
            {
                "model": self.model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
            }
        )
        content = self._message_content(response)
        return self._parse_json_content(content)

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=data, method="POST")
        request.add_header("content-type", "application/json")
        if self.api_key:
            request.add_header("authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _message_content(self, response: dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            return "{}"
        message = choices[0].get("message") or {}
        content = message.get("content", "{}")
        if isinstance(content, list):
            text_parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") in {None, "text", "output_text"}
            ]
            return "".join(text_parts) or "{}"
        return str(content)

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise
            parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}


@dataclass
class FallbackMemoryAI:
    """Use an external AI first, then fall back to local heuristics with trace metadata."""

    primary: MemoryAI
    fallback: MemoryAI = field(default_factory=HeuristicMemoryAI)
    last_call_metadata: dict[str, Any] | None = None

    def analyze_turn(self, text: str, relationship_context: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.primary.analyze_turn(text, relationship_context)
            self._record("analyze_turn", ai_provider_name(self.primary), fallback_used=False)
            return result
        except Exception as exc:
            result = self.fallback.analyze_turn(text, relationship_context)
            self._record(
                "analyze_turn",
                ai_provider_name(self.fallback),
                fallback_used=True,
                primary_provider=ai_provider_name(self.primary),
                error=exc,
            )
            reason = str(result.get("reason", ""))
            result["reason"] = f"{reason}; external AI failed, used local fallback: {type(exc).__name__}"
            return result

    def summarize_story(self, event_contents: list[str], relationship_context: dict[str, Any]) -> str:
        try:
            result = self.primary.summarize_story(event_contents, relationship_context)
            self._record("summarize_story", ai_provider_name(self.primary), fallback_used=False)
            return result
        except Exception as exc:
            self._record(
                "summarize_story",
                ai_provider_name(self.fallback),
                fallback_used=True,
                primary_provider=ai_provider_name(self.primary),
                error=exc,
            )
            return self.fallback.summarize_story(event_contents, relationship_context)

    def evaluate_memory_value(self, content: str, relationship_context: dict[str, Any]) -> float:
        try:
            result = self.primary.evaluate_memory_value(content, relationship_context)
            self._record("evaluate_memory_value", ai_provider_name(self.primary), fallback_used=False)
            return result
        except Exception as exc:
            self._record(
                "evaluate_memory_value",
                ai_provider_name(self.fallback),
                fallback_used=True,
                primary_provider=ai_provider_name(self.primary),
                error=exc,
            )
            return self.fallback.evaluate_memory_value(content, relationship_context)

    def assess_time_conflict(self, candidate: dict[str, Any], relationship_context: dict[str, Any]) -> dict[str, Any]:
        try:
            assessor = getattr(self.primary, "assess_time_conflict")
            result = assessor(candidate, relationship_context)
            self._record("assess_time_conflict", ai_provider_name(self.primary), fallback_used=False)
            return result
        except Exception as exc:
            fallback_assessor = getattr(self.fallback, "assess_time_conflict", None)
            if callable(fallback_assessor):
                result = fallback_assessor(candidate, relationship_context)
            else:
                result = HeuristicMemoryAI().assess_time_conflict(candidate, relationship_context)
            self._record(
                "assess_time_conflict",
                ai_provider_name(self.fallback),
                fallback_used=True,
                primary_provider=ai_provider_name(self.primary),
                error=exc,
            )
            reason = str(result.get("reason", ""))
            result["reason"] = f"{reason}; external AI failed, used local fallback: {type(exc).__name__}"
            return result

    def consume_last_call_metadata(self) -> dict[str, Any] | None:
        metadata = self.last_call_metadata
        self.last_call_metadata = None
        return metadata

    def _record(
        self,
        task: str,
        used_provider: str,
        *,
        fallback_used: bool,
        primary_provider: str | None = None,
        error: Exception | None = None,
    ) -> None:
        metadata: dict[str, Any] = {
            "task": task,
            "used_provider": used_provider,
            "fallback_used": fallback_used,
        }
        if primary_provider:
            metadata["primary_provider"] = primary_provider
        if error:
            metadata["error_type"] = type(error).__name__
            metadata["error"] = str(error)
        self.last_call_metadata = metadata


def ai_provider_name(ai: MemoryAI) -> str:
    return ai.__class__.__name__


def provider_participation_kind(provider: str | None) -> str:
    if provider == "HeuristicMemoryAI":
        return "local_heuristic"
    if provider == "HttpMemoryAI":
        return "external_http_worker"
    if provider == "MemoryAIWorkerAdapter":
        return "external_http_worker"
    if provider == "OpenAICompatibleMemoryAI":
        return "external_model"
    if provider == "FallbackMemoryAI":
        return "external_with_local_fallback"
    return "custom"


def consume_ai_call_metadata(ai: MemoryAI) -> dict[str, Any] | None:
    consumer = getattr(ai, "consume_last_call_metadata", None)
    if callable(consumer):
        return consumer()
    return None


def describe_memory_ai(ai: MemoryAI) -> dict[str, Any]:
    if isinstance(ai, FallbackMemoryAI):
        return {
            "provider": ai_provider_name(ai),
            "participation_kind": "external_with_local_fallback",
            "fallback_enabled": True,
            "primary": describe_memory_ai(ai.primary),
            "fallback": describe_memory_ai(ai.fallback),
        }
    if isinstance(ai, HttpMemoryAI):
        return {
            "provider": ai_provider_name(ai),
            "participation_kind": "external_http_worker",
            "endpoint": ai.endpoint,
            "timeout_seconds": ai.timeout_seconds,
            "api_key_configured": bool(ai.api_key),
        }
    if isinstance(ai, OpenAICompatibleMemoryAI):
        return {
            "provider": ai_provider_name(ai),
            "participation_kind": "external_model",
            "endpoint": ai.endpoint,
            "model": ai.model,
            "timeout_seconds": ai.timeout_seconds,
            "api_key_configured": bool(ai.api_key),
        }
    if isinstance(ai, HeuristicMemoryAI):
        return {
            "provider": ai_provider_name(ai),
            "participation_kind": "local_heuristic",
            "fallback_enabled": False,
            "external_model": False,
        }
    provider = ai_provider_name(ai)
    return {"provider": provider, "participation_kind": provider_participation_kind(provider), "fallback_enabled": False}


def build_memory_ai_from_env() -> MemoryAI:
    load_dotenv()
    provider = os.environ.get("MEMORY_AI_PROVIDER", "heuristic").lower().replace("_", "-")
    if provider == "http":
        endpoint = os.environ["MEMORY_AI_ENDPOINT"]
        timeout = float(os.environ.get("MEMORY_AI_TIMEOUT_SECONDS", "10"))
        primary = HttpMemoryAI(endpoint=endpoint, api_key=os.environ.get("MEMORY_AI_API_KEY"), timeout_seconds=timeout)
        if os.environ.get("MEMORY_AI_DISABLE_FALLBACK", "").lower() in {"1", "true", "yes", "on"}:
            return primary
        return FallbackMemoryAI(primary=primary)
    if provider in {"openai", "openai-compatible", "chat-completions"}:
        endpoint = os.environ.get("MEMORY_AI_ENDPOINT", "https://api.openai.com/v1/chat/completions")
        model = os.environ.get("MEMORY_AI_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
        api_key = os.environ.get("MEMORY_AI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        timeout = float(os.environ.get("MEMORY_AI_TIMEOUT_SECONDS", "20"))
        primary = OpenAICompatibleMemoryAI(endpoint=endpoint, model=model, api_key=api_key, timeout_seconds=timeout)
        if os.environ.get("MEMORY_AI_DISABLE_FALLBACK", "").lower() in {"1", "true", "yes", "on"}:
            return primary
        return FallbackMemoryAI(primary=primary)
    return HeuristicMemoryAI()
