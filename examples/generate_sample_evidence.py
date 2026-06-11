from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from main import EVIDENCE_DATASET_SCHEMA, EVIDENCE_FILENAMES  # noqa: E402

TASK_FILENAMES = dict(EVIDENCE_FILENAMES)


def with_schema(task: str, payload: dict) -> dict:
    return {"schema": EVIDENCE_DATASET_SCHEMA, "task": task, **payload}


def sample_datasets() -> dict[str, dict]:
    return {
        "stage_detection": with_schema("stage_detection", {
            "config": {"required_samples": 2, "target_accuracy": 0.75},
            "examples": [
                {"text": "你不懂我，我很失望。", "expected_stage": "DIFFERENTIATING"},
                {"text": "第一次一起庆祝成功，太开心了！", "expected_stage": "INTENSIFYING"},
            ],
        }),
        "self_disclosure_capture": with_schema("self_disclosure_capture", {
            "config": {"required_samples": 2, "target_recall": 0.90},
            "examples": [
                {"text": "其实我从来没告诉别人，我那天哭了很久。", "expected_self_disclosure": True},
                {"text": "今天只是普通聊项目进展。", "expected_self_disclosure": False},
            ],
        }),
        "inside_joke_detection": with_schema("inside_joke_detection", {
            "config": {"required_samples": 2, "target_accuracy": 0.70},
            "examples": [
                {
                    "turns": [
                        "我们先把这个叫做“样例暗号A”。",
                        "今天继续说“样例暗号A”。",
                        "又到了“样例暗号A”的时候。",
                    ],
                    "expected_detected": True,
                    "expected_phrase": "样例暗号A",
                },
                {"turns": ["今天只是普通聊项目进展。"], "expected_detected": False},
            ],
        }),
        "emotional_resonance_retrieval": with_schema("emotional_resonance_retrieval", {
            "config": {"required_samples": 1, "target_p5": 0.65},
            "examples": [
                {
                    "memories": ["我那天崩溃哭了很久，压力特别大。"],
                    "query": "我今天也崩溃难过，压力很大。",
                    "expected_relevant_indices": [0],
                }
            ],
        }),
        "story_quality": with_schema("story_quality", {
            "config": {"required_samples": 2, "target_average_score": 4.0},
            "examples": [
                {"story_id": "story_1", "title": "样例暗号A", "score": 4.5, "note": "来源清晰"},
                {"story_id": "story_2", "title": "第一次庆祝", "score": 4.0, "note": "叙事一致"},
            ],
        }),
        "friend_mode_ab": with_schema("friend_mode_ab", {
            "config": {"duration_weeks": 12},
            "examples": [
                {
                    "cohort": "control",
                    "users": 1200,
                    "nps": 10,
                    "retention_rate": 0.40,
                    "avg_session_minutes": 10,
                    "avg_intimacy_delta": 0.4,
                },
                {
                    "cohort": "friend",
                    "users": 1250,
                    "nps": 32,
                    "retention_rate": 0.45,
                    "avg_session_minutes": 13.5,
                    "avg_intimacy_delta": 2.0,
                },
            ],
        }),
        "production_telemetry": with_schema("production_telemetry", {
            "config": {"duration_days": 7},
            "metrics": {
                "active_users": 50,
                "active_complaint_rate": 0.04,
                "hard_delete_success_rate": 1.0,
                "transparency_ack_rate": 0.35,
                "crisis_escalation_review_rate": 1.0,
            },
        }),
    }


def decision_report_command(output_dir: Path) -> str:
    return (
        "uv --cache-dir .uv-cache run python app/main.py decision-report "
        f"--evidence-dir {output_dir} --run-benchmarks --benchmark-iterations 20 --json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate small sample evidence JSON files for decision-report demos.")
    parser.add_argument("--output-dir", default="/tmp/friend-memory-evidence")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for task, dataset in sample_datasets().items():
        path = output_dir / TASK_FILENAMES[task]
        path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    command = decision_report_command(output_dir)
    if args.quiet:
        print(str(output_dir))
        return
    print(json.dumps({"output_dir": str(output_dir), "files": TASK_FILENAMES, "decision_report_command": command}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
