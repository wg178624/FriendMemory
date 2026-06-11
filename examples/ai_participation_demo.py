from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "examples"))

from memory_ai_worker import analyze_turn, assess_time_conflict, evaluate_memory_value, summarize_story  # noqa: E402
from main import ai_observation_report_for_project  # noqa: E402
from system import FriendMemoryProject  # noqa: E402


class MemoryAIWorkerAdapter:
    """In-process adapter for the example external worker tasks.

    The real HTTP worker is `examples/memory_ai_worker.py`. This adapter avoids
    opening a local socket, so the participation proof can run in restricted
    CI/sandbox environments while exercising the same structured task outputs.
    """

    def analyze_turn(self, text: str, relationship_context: dict[str, Any]) -> dict[str, Any]:
        return analyze_turn(text, relationship_context)

    def summarize_story(self, event_contents: list[str], relationship_context: dict[str, Any]) -> str:
        return summarize_story(event_contents, relationship_context)["summary"]

    def evaluate_memory_value(self, content: str, relationship_context: dict[str, Any]) -> float:
        return float(evaluate_memory_value(content, relationship_context)["value"])

    def assess_time_conflict(self, candidate: dict[str, Any], relationship_context: dict[str, Any]) -> dict[str, Any]:
        return assess_time_conflict(candidate, relationship_context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate auditable external MemoryAI participation.")
    parser.add_argument(
        "--output-ai-observation",
        help="Write a release-gate compatible ai-observation.json file for this demo run.",
    )
    args = parser.parse_args()

    project = FriendMemoryProject(ai=MemoryAIWorkerAdapter())
    result = project.ingest_turn(
        "demo-user",
        "demo-ai",
        "其实我从来没告诉别人，第一次失败那天我哭了，但后来我们一起庆祝重新开始。",
        timestamp=datetime.now(timezone.utc),
    )
    decision = next(
        item
        for item in reversed(project.ai_decision_log)
        if item.get("relationship_id") == result.relationship_id and item.get("task") == "analyze_turn"
    )
    summary = project.ai_decision_summary(decision)
    status = project.ai_status(result.relationship_id)
    report = project.decision_report(result.relationship_id)
    observation = ai_observation_report_for_project(project, result.relationship_id)
    observation_path = None
    if args.output_ai_observation:
        target = Path(args.output_ai_observation)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(observation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        observation_path = str(target)

    print(
        json.dumps(
            {
                "relationship_id": result.relationship_id,
                "memory_id": result.memory_id,
                "ai_decision_summary": summary,
                "ai_status": {
                    "provider": status["provider"],
                    "participation_kind": status["participation_kind"],
                    "readiness_status": status["readiness_status"],
                    "readiness_label": status["readiness_label"],
                    "external_ai_configured": status["external_ai_configured"],
                    "external_ai_used_recently": status["external_ai_used_recently"],
                    "external_success_count": status["readiness"]["external_success_count"],
                    "fallback_event_count": status["fallback_event_count"],
                },
                "decision_report_ai_readiness": report["ai_readiness"],
                "ai_observation": {
                    "formal_ready": observation["formal_ready"],
                    "readiness_status": observation["readiness_status"],
                    "external_success_count": observation["external_success_count"],
                    "output_path": observation_path,
                },
                "proof": {
                    "external_worker_called": summary["used_provider"] == "MemoryAIWorkerAdapter",
                    "participation_kind": summary["used_participation_kind"],
                    "fallback_used": summary["fallback_used"],
                    "reason": summary["reason"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
