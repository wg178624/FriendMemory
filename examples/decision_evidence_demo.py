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

from audit_coverage_demo import build_project  # noqa: E402
from generate_sample_evidence import sample_datasets  # noqa: E402


DEMO_NOW = datetime(2026, 5, 11, tzinfo=timezone.utc)


def build_evaluation_results(project: Any) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for task, dataset in sample_datasets().items():
        evaluation = project.evaluate_labeled_dataset(dataset, task=task)
        results[evaluation["task"]] = evaluation
    return results


def not_pass_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for phase in report["phases"].values():
        for criterion in phase["criteria"]:
            if criterion["status"] != "pass":
                items.append(
                    {
                        "phase": phase["id"],
                        "criterion_id": criterion["id"],
                        "status": criterion["status"],
                        "target": criterion["target"],
                        "evidence": criterion.get("evidence", {}),
                    }
                )
    return items


def demo_status(report: dict[str, Any]) -> str:
    status_counts = report["summary"].get("status_counts", {})
    if status_counts.get("fail") or status_counts.get("missing_evidence"):
        return "CHECK"
    if status_counts.get("external_required"):
        return "NEEDS_EXTERNAL_OR_BENCHMARK_EVIDENCE"
    return "ALL_DEMO_CRITERIA_PASS_BUT_NOT_FORMAL"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a runtime + labelled-sample decision evidence report for the friend memory system."
    )
    parser.add_argument("--run-benchmarks", action="store_true", help="include local latency/write benchmarks")
    parser.add_argument("--benchmark-iterations", type=int, default=20)
    parser.add_argument("--full-report", action="store_true", help="include the full decision_report payload")
    args = parser.parse_args()

    project = build_project()
    evaluation_results = build_evaluation_results(project)
    report = project.decision_report(
        now=DEMO_NOW,
        run_benchmarks=args.run_benchmarks,
        benchmark_iterations=args.benchmark_iterations,
        evaluation_results=evaluation_results,
    )
    audit = project.audit_report(now=DEMO_NOW)

    payload: dict[str, Any] = {
        "status": demo_status(report),
        "evidence_mode": "synthetic_runtime_plus_sample_fixtures",
        "formal_completion_claim": "not_proven_sample_fixture",
        "report_completion_claim": report["summary"].get("completion_claim"),
        "sample_evidence_notice": (
            "Labelled/A-B evidence in this demo is a tiny synthetic fixture for wiring proof; "
            "formal completion still requires real 200-sample reviews, production telemetry, and A/B data."
        ),
        "audit_status": audit["status"],
        "coverage_summary": audit["coverage_summary"],
        "decision_summary": report["summary"],
        "evaluation_tasks": sorted(evaluation_results),
        "not_pass_items": not_pass_items(report),
        "ai_readiness": report["ai_readiness"],
    }
    if args.full_report:
        payload["decision_report"] = report

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
