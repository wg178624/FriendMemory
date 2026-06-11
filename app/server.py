from __future__ import annotations

import argparse
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from env_config import load_dotenv
from models import ContextTag, MemoryLayer, MemoryType, Mode, ResetMode
from system import FriendMemoryProject
from main import (
    EVALUATION_TASKS,
    ai_observation_report_for_project,
    decision_report_with_evidence,
    evidence_input_warnings,
    evidence_manifest,
    evaluation_template,
    project_doctor_report,
    release_bundle_for_project,
    release_gate_report_for_project,
    validate_evidence_dir_for_project,
)

load_dotenv()
STATE_PATH = Path(os.environ.get("FRIEND_MEMORY_STATE", "data/friend_memory_state.json"))

HTTP_ROUTES = [
    {"method": "GET", "path": "/healthz", "category": "system", "writes_state": False, "description": "process health check"},
    {"method": "GET", "path": "/doctor", "category": "system", "writes_state": False, "description": "read-only project readiness report"},
    {"method": "GET", "path": "/routes", "category": "system", "writes_state": False, "description": "machine-readable HTTP route catalog"},
    {"method": "GET", "path": "/browser", "category": "relationship", "writes_state": False, "description": "relationship browser snapshot"},
    {"method": "GET", "path": "/mode-suggestions", "category": "relationship", "writes_state": False, "description": "mode switching suggestions"},
    {"method": "GET", "path": "/retrieve", "category": "memory", "writes_state": True, "description": "retrieve relationship memories"},
    {"method": "GET", "path": "/export", "category": "privacy", "writes_state": True, "description": "export relationship data"},
    {"method": "GET", "path": "/transparency", "category": "transparency", "writes_state": True, "description": "relationship transparency panel"},
    {"method": "GET", "path": "/ai-status", "category": "ai", "writes_state": False, "description": "AI provider status and recent decisions"},
    {"method": "GET", "path": "/ai-observation", "category": "ai", "writes_state": False, "description": "external AI observation evidence"},
    {"method": "GET", "path": "/health", "category": "safety", "writes_state": True, "description": "health and safety alerts"},
    {"method": "GET", "path": "/audit", "category": "evidence", "writes_state": False, "description": "project or relationship audit report"},
    {"method": "GET", "path": "/decision-report", "category": "evidence", "writes_state": False, "description": "decision evidence report"},
    {"method": "GET", "path": "/evidence-manifest", "category": "evidence", "writes_state": False, "description": "evidence manifest"},
    {"method": "GET", "path": "/evidence-template", "category": "evidence", "writes_state": False, "description": "formal evidence template"},
    {"method": "GET", "path": "/evidence-validate", "category": "evidence", "writes_state": False, "description": "formal evidence validation"},
    {"method": "GET", "path": "/release-gate", "category": "evidence", "writes_state": False, "description": "release readiness gate"},
    {"method": "GET", "path": "/release-bundle", "category": "evidence", "writes_state": True, "description": "write release evidence bundle"},
    {"method": "GET", "path": "/deletion-compliance", "category": "privacy", "writes_state": True, "description": "auditor-only deletion compliance log"},
    {"method": "GET", "path": "/reminders", "category": "memory", "writes_state": True, "description": "commitment reminders"},
    {"method": "POST", "path": "/ingest", "category": "memory", "writes_state": True, "description": "record a user turn"},
    {"method": "POST", "path": "/ingest-exchange", "category": "memory", "writes_state": True, "description": "record a complete user/assistant exchange"},
    {"method": "POST", "path": "/ai-probe", "category": "ai", "writes_state": False, "description": "call MemoryAI without writing memory"},
    {"method": "POST", "path": "/evaluate-labels", "category": "evidence", "writes_state": False, "description": "evaluate labeled evidence dataset"},
    {"method": "POST", "path": "/consolidate", "category": "memory", "writes_state": True, "description": "run offline consolidation"},
    {"method": "POST", "path": "/migrate", "category": "migration", "writes_state": True, "description": "migrate legacy turns"},
    {"method": "POST", "path": "/migrate/rollback", "category": "migration", "writes_state": True, "description": "rollback a migration batch"},
    {"method": "POST", "path": "/health", "category": "safety", "writes_state": True, "description": "run health and safety review"},
    {"method": "POST", "path": "/guardian-summary", "category": "safety", "writes_state": True, "description": "generate minor guardian summary"},
    {"method": "POST", "path": "/reset/request", "category": "privacy", "writes_state": True, "description": "request relationship reset"},
    {"method": "POST", "path": "/reset/confirm", "category": "privacy", "writes_state": True, "description": "confirm relationship reset"},
    {"method": "POST", "path": "/reset/cancel", "category": "privacy", "writes_state": True, "description": "cancel relationship reset"},
    {"method": "POST", "path": "/user/age", "category": "safety", "writes_state": True, "description": "set user age"},
    {"method": "POST", "path": "/interaction/minutes", "category": "safety", "writes_state": True, "description": "record daily interaction minutes"},
    {"method": "POST", "path": "/mode", "category": "controls", "writes_state": True, "description": "switch relationship mode"},
    {"method": "POST", "path": "/preference", "category": "controls", "writes_state": True, "description": "update relationship preference"},
    {"method": "POST", "path": "/decay-curve", "category": "controls", "writes_state": True, "description": "set memory decay curve"},
    {"method": "POST", "path": "/custom-profile", "category": "controls", "writes_state": True, "description": "update custom mode profile"},
    {"method": "POST", "path": "/stage/rollback", "category": "relationship", "writes_state": True, "description": "rollback relationship stage"},
    {"method": "POST", "path": "/story/correct", "category": "story", "writes_state": True, "description": "correct story consensus"},
    {"method": "POST", "path": "/story/confirm", "category": "story", "writes_state": True, "description": "confirm story consensus"},
    {"method": "POST", "path": "/story/rollback", "category": "story", "writes_state": True, "description": "rollback story narrative"},
    {"method": "POST", "path": "/transparency/ack", "category": "transparency", "writes_state": True, "description": "acknowledge transparency disclosure"},
    {"method": "POST", "path": "/memory-writes", "category": "controls", "writes_state": True, "description": "pause or resume persistent memory writes"},
    {"method": "POST", "path": "/health/ack", "category": "safety", "writes_state": True, "description": "acknowledge health alert"},
    {"method": "POST", "path": "/health/feedback", "category": "safety", "writes_state": True, "description": "record health alert feedback"},
    {"method": "POST", "path": "/active-feedback", "category": "active-recall", "writes_state": True, "description": "record active recall feedback"},
    {"method": "POST", "path": "/active-type/mute", "category": "active-recall", "writes_state": True, "description": "mute an active recall type"},
    {"method": "POST", "path": "/active-type/unmute", "category": "active-recall", "writes_state": True, "description": "unmute an active recall type"},
    {"method": "POST", "path": "/implicit-topic/feedback", "category": "active-recall", "writes_state": True, "description": "record implicit topic feedback"},
    {"method": "POST", "path": "/memory/inject", "category": "memory", "writes_state": True, "description": "manual memory injection"},
    {"method": "POST", "path": "/memory/edit", "category": "memory", "writes_state": True, "description": "edit memory content"},
    {"method": "POST", "path": "/memory/retag", "category": "memory", "writes_state": True, "description": "retag memory type or context"},
    {"method": "POST", "path": "/memory/mark-milestone", "category": "memory", "writes_state": True, "description": "mark memory as milestone"},
    {"method": "POST", "path": "/inside-joke/status", "category": "memory", "writes_state": True, "description": "control inside joke status"},
    {"method": "POST", "path": "/thread/resolve", "category": "memory", "writes_state": True, "description": "resolve unresolved thread"},
    {"method": "POST", "path": "/memory/downgrade", "category": "memory", "writes_state": True, "description": "downgrade memory"},
    {"method": "POST", "path": "/memory/suppress", "category": "memory", "writes_state": True, "description": "suppress memory recall"},
    {"method": "POST", "path": "/memory/unsuppress", "category": "memory", "writes_state": True, "description": "unsuppress memory recall"},
    {"method": "POST", "path": "/memory/restore-archive", "category": "memory", "writes_state": True, "description": "restore archived memory"},
    {"method": "POST", "path": "/memory/verify", "category": "memory", "writes_state": True, "description": "mark memory as user verified"},
    {"method": "POST", "path": "/memory/calibrate", "category": "memory", "writes_state": True, "description": "calibrate memory correctness"},
    {"method": "POST", "path": "/retention/feedback", "category": "memory", "writes_state": True, "description": "record retention feedback"},
    {"method": "POST", "path": "/time-conflict/resolve", "category": "memory", "writes_state": True, "description": "resolve time conflict"},
    {"method": "POST", "path": "/milestone/confirm", "category": "story", "writes_state": True, "description": "confirm milestone"},
    {"method": "POST", "path": "/milestone/edit", "category": "story", "writes_state": True, "description": "edit milestone"},
    {"method": "POST", "path": "/milestone/reject", "category": "story", "writes_state": True, "description": "reject milestone"},
    {"method": "POST", "path": "/batch-downgrade", "category": "memory", "writes_state": True, "description": "batch downgrade memories"},
    {"method": "POST", "path": "/reminder/complete", "category": "memory", "writes_state": True, "description": "complete reminder"},
    {"method": "POST", "path": "/memory/delete-request", "category": "privacy", "writes_state": True, "description": "request memory deletion"},
    {"method": "POST", "path": "/memory/delete-confirm", "category": "privacy", "writes_state": True, "description": "confirm memory deletion"},
    {"method": "POST", "path": "/memory/delete-cancel", "category": "privacy", "writes_state": True, "description": "cancel memory deletion"},
    {"method": "POST", "path": "/l4/delete-request", "category": "privacy", "writes_state": True, "description": "request core identity deletion"},
    {"method": "POST", "path": "/l4/delete-confirm", "category": "privacy", "writes_state": True, "description": "confirm core identity deletion"},
    {"method": "POST", "path": "/l4/delete-cancel", "category": "privacy", "writes_state": True, "description": "cancel core identity deletion"},
    {"method": "POST", "path": "/l4/review", "category": "memory", "writes_state": True, "description": "review core identity"},
]


def load_project() -> FriendMemoryProject:
    if STATE_PATH.exists():
        return FriendMemoryProject.load(STATE_PATH)
    return FriendMemoryProject()


def save_project(project: FriendMemoryProject) -> None:
    project.save(STATE_PATH)


class FriendMemoryHandler(BaseHTTPRequestHandler):
    project = load_project()
    project_lock = threading.RLock()

    def do_GET(self) -> None:
        with self.project_lock:
            self._do_GET_unlocked()

    def _do_GET_unlocked(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/healthz":
                self._json({"ok": True})
            elif parsed.path == "/doctor":
                self._json(project_doctor_report())
            elif parsed.path == "/routes":
                methods = sorted({route["method"] for route in HTTP_ROUTES})
                categories = sorted({route["category"] for route in HTTP_ROUTES})
                self._json(
                    {
                        "schema": "friend-memory-http-routes-v1",
                        "route_count": len(HTTP_ROUTES),
                        "methods": methods,
                        "categories": categories,
                        "routes": HTTP_ROUTES,
                    }
                )
            elif parsed.path == "/browser":
                relationship_id = self._relationship_id(query)
                self._json(self.project.browser_snapshot(relationship_id))
            elif parsed.path == "/mode-suggestions":
                relationship_id = self._relationship_id(query)
                self._json({"suggestions": self.project.mode_suggestions(relationship_id)})
            elif parsed.path == "/retrieve":
                relationship_id = self._relationship_id(query)
                q = self._one(query, "q", "")
                limit = self._int_query(query, "limit", 5, minimum=1)
                include_archived = self._one(query, "include_archived", "false").lower() in {"1", "true", "yes", "on"}
                retrieved = self.project.retrieve(relationship_id, q, limit=limit, include_archived=include_archived)
                results = [
                    {
                        "memory_id": item.memory.memory_id,
                        "type": item.memory.memory_type.value,
                        "score": item.score,
                        "content": item.explanation.get("display_content", item.memory.content),
                        "raw_content_available": item.explanation.get("trust_presentation", {}).get("original_preserved", True),
                        "time": item.presentation_time,
                        "explanation": item.explanation,
                    }
                    for item in retrieved
                ]
                query_metacognition = (
                    retrieved[0].explanation.get("query_metacognition")
                    if retrieved
                    else self.project.retrieval_audit_log[-1].get("query_metacognition")
                    if self.project.retrieval_audit_log
                    else None
                )
                association_expansions = retrieved[0].explanation.get("association_expansions", []) if retrieved else []
                save_project(self.project)
                self._json(
                    {
                        "results": results,
                        "query_metacognition": query_metacognition,
                        "association_expansions": association_expansions,
                    }
                )
            elif parsed.path == "/export":
                relationship_id = query.get("relationship_id", [None])[0]
                export_format = self._one(query, "format", "json")
                anonymized = self._one(query, "anonymize", "false").lower() in {"1", "true", "yes", "on"}
                purpose = self._one(query, "purpose", "user_archive")
                if export_format in {"json", "anonymous-json"}:
                    relationship_id = relationship_id or (self._relationship_id(query) if "user" in query or "ai" in query else None)
                    anonymized = anonymized or export_format == "anonymous-json" or self.project.export_requires_anonymization(relationship_id)
                    payload = self.project.generate_export(
                        relationship_id=relationship_id,
                        export_format=export_format,
                        anonymized=anonymized,
                        destination="http_response",
                        purpose=purpose,
                    )
                    save_project(self.project)
                    self._json(payload)
                else:
                    relationship_id = relationship_id or self._relationship_id(query)
                    if export_format == "narrative":
                        payload = self.project.generate_export(
                            relationship_id=relationship_id,
                            export_format="narrative",
                            destination="http_response",
                            purpose=purpose,
                        )
                        save_project(self.project)
                        self._text(str(payload), "text/markdown; charset=utf-8")
                    elif export_format == "milestones":
                        payload = self.project.generate_export(
                            relationship_id=relationship_id,
                            export_format="milestones",
                            destination="http_response",
                            purpose=purpose,
                        )
                        save_project(self.project)
                        self._json(payload)
                    elif export_format == "timeline":
                        payload = self.project.generate_export(
                            relationship_id=relationship_id,
                            export_format="timeline",
                            destination="http_response",
                            purpose=purpose,
                        )
                        save_project(self.project)
                        self._text(str(payload), "text/csv; charset=utf-8")
                    else:
                        self._json({"error": "unknown export format"}, HTTPStatus.BAD_REQUEST)
            elif parsed.path == "/transparency":
                relationship_id = self._relationship_id(query)
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                    save_project(self.project)
                self._json(self.project.transparency_panel(relationship_id))
            elif parsed.path == "/ai-status":
                relationship_id = None
                if self._one(query, "all", "false").lower() not in {"1", "true", "yes", "on"}:
                    relationship_id = self._relationship_id(query)
                self._json(self.project.ai_status(relationship_id))
            elif parsed.path == "/ai-observation":
                relationship_id = query.get("relationship_id", [None])[0]
                if relationship_id is None and ("user" in query or "ai" in query):
                    relationship_id = self._relationship_id(query)
                self._json(ai_observation_report_for_project(self.project, relationship_id))
            elif parsed.path == "/health":
                relationship_id = self._relationship_id(query)
                self._json(self._health_payload(relationship_id))
            elif parsed.path == "/audit":
                relationship_id = query.get("relationship_id", [None])[0]
                if relationship_id is None and ("user" in query or "ai" in query):
                    relationship_id = self._relationship_id(query)
                self._json(self.project.audit_report(relationship_id))
            elif parsed.path == "/decision-report":
                relationship_id = query.get("relationship_id", [None])[0]
                if relationship_id is None and ("user" in query or "ai" in query):
                    relationship_id = self._relationship_id(query)
                run_benchmarks = self._one(query, "run_benchmarks", "false").lower() in {"1", "true", "yes", "on"}
                benchmark_iterations = self._int_query(query, "benchmark_iterations", 20, minimum=1)
                self._json(
                    decision_report_with_evidence(
                        self.project,
                        relationship_id=relationship_id,
                        run_benchmarks=run_benchmarks,
                        benchmark_iterations=benchmark_iterations,
                        evidence_dir=Path(self._one(query, "evidence_dir", ""))
                        if self._one(query, "evidence_dir", "")
                        else None,
                        manifest_path=Path(self._one(query, "manifest", ""))
                        if self._one(query, "manifest", "")
                        else None,
                    )
                )
            elif parsed.path == "/evidence-manifest":
                evidence_dir_value = self._one(query, "evidence_dir", "")
                if not evidence_dir_value:
                    self._json({"error": "evidence_dir is required"}, HTTPStatus.BAD_REQUEST)
                else:
                    self._json(evidence_manifest(Path(evidence_dir_value)))
            elif parsed.path == "/evidence-template":
                if self._one(query, "all", "false").lower() in {"1", "true", "yes", "on"}:
                    self._json({task: evaluation_template(task) for task in EVALUATION_TASKS})
                else:
                    task = self._one(query, "task", "stage_detection")
                    if task not in EVALUATION_TASKS:
                        self._json({"error": f"task must be one of: {', '.join(EVALUATION_TASKS)}"}, HTTPStatus.BAD_REQUEST)
                    else:
                        self._json(evaluation_template(task))
            elif parsed.path == "/evidence-validate":
                evidence_dir_value = self._one(query, "evidence_dir", "")
                if not evidence_dir_value:
                    self._json({"error": "evidence_dir is required"}, HTTPStatus.BAD_REQUEST)
                else:
                    evidence_dir = Path(evidence_dir_value)
                    manifest_value = self._one(query, "manifest", "")
                    self._json(
                        validate_evidence_dir_for_project(
                            self.project,
                            evidence_dir,
                            manifest_path=Path(manifest_value) if manifest_value else None,
                        )
                    )
            elif parsed.path == "/release-gate":
                evidence_dir_value = self._one(query, "evidence_dir", "")
                if not evidence_dir_value:
                    self._json({"error": "evidence_dir is required"}, HTTPStatus.BAD_REQUEST)
                else:
                    evidence_dir = Path(evidence_dir_value)
                    relationship_id = query.get("relationship_id", [None])[0]
                    if relationship_id is None and ("user" in query or "ai" in query):
                        relationship_id = self._relationship_id(query)
                    run_benchmarks = self._one(query, "run_benchmarks", "false").lower() in {"1", "true", "yes", "on"}
                    benchmark_iterations = self._int_query(query, "benchmark_iterations", 20, minimum=1)
                    require_external_ai = self._one(query, "require_external_ai", "false").lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                    self._json(
                        release_gate_report_for_project(
                            self.project,
                            evidence_dir=evidence_dir,
                            manifest_path=Path(self._one(query, "manifest", ""))
                            if self._one(query, "manifest", "")
                            else None,
                            relationship_id=relationship_id,
                            run_benchmarks=run_benchmarks,
                            benchmark_iterations=benchmark_iterations,
                            require_external_ai=require_external_ai,
                        )
                    )
            elif parsed.path == "/release-bundle":
                evidence_dir_value = self._one(query, "evidence_dir", "")
                output_dir_value = self._one(query, "output_dir", "")
                if not evidence_dir_value:
                    self._json({"error": "evidence_dir is required"}, HTTPStatus.BAD_REQUEST)
                elif not output_dir_value:
                    self._json({"error": "output_dir is required"}, HTTPStatus.BAD_REQUEST)
                else:
                    relationship_id = query.get("relationship_id", [None])[0]
                    if relationship_id is None and ("user" in query or "ai" in query):
                        relationship_id = self._relationship_id(query)
                    run_benchmarks = self._one(query, "run_benchmarks", "false").lower() in {"1", "true", "yes", "on"}
                    benchmark_iterations = self._int_query(query, "benchmark_iterations", 20, minimum=1)
                    require_external_ai = self._one(query, "require_external_ai", "false").lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                    self._json(
                        release_bundle_for_project(
                            self.project,
                            evidence_dir=Path(evidence_dir_value),
                            output_dir=Path(output_dir_value),
                            relationship_id=relationship_id,
                            run_benchmarks=run_benchmarks,
                            benchmark_iterations=benchmark_iterations,
                            require_external_ai=require_external_ai,
                        )
                    )
            elif parsed.path == "/deletion-compliance":
                relationship_id = query.get("relationship_id", [None])[0] or self._relationship_id(query)
                token = self._one(query, "auditor_token", "")
                report = self.project.deletion_compliance_audit(relationship_id, auditor_token=token)
                save_project(self.project)
                self._json(report)
            elif parsed.path == "/reminders":
                relationship_id = self._relationship_id(query)
                window_days = self._int_query(query, "window_days", 1, minimum=0)
                reminders = self.project.check_commitment_reminders(relationship_id, window_days=window_days)
                save_project(self.project)
                self._json({"reminders": reminders, "count": len(reminders)})
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except PermissionError as exc:
            self._json({"error": str(exc), "error_type": "PermissionError"}, HTTPStatus.FORBIDDEN)
        except Exception as exc:  # pragma: no cover - handler safety path
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        with self.project_lock:
            self._do_POST_unlocked()

    def _do_POST_unlocked(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._body()
            if parsed.path == "/ingest":
                result = self.project.ingest_turn(
                    payload.get("user", "user"),
                    payload.get("ai", "companion"),
                    payload["text"],
                    metadata=payload.get("metadata"),
                )
                save_project(self.project)
                ai_decision = next(
                    (
                        item
                        for item in reversed(self.project.ai_decision_log)
                        if item.get("relationship_id") == result.relationship_id and item.get("task") == "analyze_turn"
                    ),
                    None,
                )
                self._json(
                    {
                        "relationship_id": result.relationship_id,
                        "memory_id": result.memory_id,
                        "emotional_memory_id": result.emotional_memory_id,
                        "score": result.score,
                        "stage": result.stage.value,
                        "active_suggestions": result.active_suggestions,
                        "active_events": result.active_events,
                        "memory_paused": result.memory_paused,
                        "ai_decision_summary": self.project.ai_decision_summary(ai_decision),
                        "ai_decision": ai_decision,
                    }
                )
            elif parsed.path == "/ingest-exchange":
                result = self.project.ingest_exchange(
                    payload.get("user", "user"),
                    payload.get("ai", "companion"),
                    payload["user_text"],
                    payload["assistant_text"],
                    metadata=payload.get("metadata"),
                )
                save_project(self.project)
                ai_decision = next(
                    (
                        item
                        for item in reversed(self.project.ai_decision_log)
                        if item.get("relationship_id") == result.relationship_id and item.get("task") == "analyze_turn"
                    ),
                    None,
                )
                self._json(
                    {
                        "relationship_id": result.relationship_id,
                        "memory_id": result.memory_id,
                        "emotional_memory_id": result.emotional_memory_id,
                        "score": result.score,
                        "stage": result.stage.value,
                        "active_suggestions": result.active_suggestions,
                        "active_events": result.active_events,
                        "memory_paused": result.memory_paused,
                        "ai_decision_summary": self.project.ai_decision_summary(ai_decision),
                        "ai_decision": ai_decision,
                    }
                )
            elif parsed.path == "/ai-probe":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                text = str(payload.get("text") or "第一次一起庆祝成功，太开心了！")
                probe = self.project.probe_ai(relationship_id, text)
                external = probe["external_ai_participation"]
                require_external = str(payload.get("require_external_ai", "false")).lower() in {
                    "1",
                    "true",
                    "yes",
                }
                requirement_met = bool(external["external_ai_participated"])
                probe["external_ai_requirement"] = {
                    "required": require_external,
                    "met": requirement_met,
                    "verdict": external["verdict"],
                    "explanation": external["explanation"],
                }
                status = HTTPStatus.OK if not require_external or requirement_met else HTTPStatus.PRECONDITION_FAILED
                self._json(probe, status)
            elif parsed.path == "/evaluate-labels":
                task = payload.get("task", "stage_detection")
                if task not in EVALUATION_TASKS:
                    self._json({"error": f"task must be one of: {', '.join(EVALUATION_TASKS)}"}, HTTPStatus.BAD_REQUEST)
                else:
                    dataset = payload.get("dataset", payload)
                    result = self.project.evaluate_labeled_dataset(dataset, task=task)
                    warnings = evidence_input_warnings(dataset, task)
                    if warnings:
                        result["input_warnings"] = warnings
                    self._json(result)
            elif parsed.path == "/consolidate":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                report = self.project.consolidate_relationship(relationship_id)
                save_project(self.project)
                self._json(
                    {
                        "relationship_id": report.relationship_id,
                        "replayed_memories": report.replayed_memories,
                        "upgraded_stories": report.upgraded_stories,
                        "downgraded_memories": report.downgraded_memories,
                        "archived_memories": report.archived_memories,
                        "compressed_stories": report.compressed_stories,
                        "health_alerts": report.health_alerts,
                        "implicit_topics": self.project.relationships[relationship_id].implicit_topics[-20:],
                    }
                )
            elif parsed.path == "/migrate":
                turns = payload.get("turns", [])
                report = self.project.migrate_legacy_turns(
                    turns,
                    default_user=payload.get("user", "user"),
                    default_ai=payload.get("ai", "companion"),
                    relationship_certificate=payload.get("relationship_certificate"),
                    require_certificate=self._bool_value(payload.get("require_certificate", False)),
                    target_mode=payload.get("target_mode"),
                )
                save_project(self.project)
                self._json(
                    {
                        "migration_id": report.migration_id,
                        "imported_turns": report.imported_turns,
                        "relationship_ids": report.relationship_ids,
                        "created_memories": report.created_memories,
                        "created_emotional_memories": report.created_emotional_memories,
                        "recognized_milestones": report.recognized_milestones,
                        "rollback_expires_at": report.rollback_expires_at.isoformat(),
                        "relationship_certificate": self.project.migration_batches[report.migration_id].get("relationship_certificate"),
                        "target_mode": self.project.migration_batches[report.migration_id].get("target_mode"),
                    }
                )
            elif parsed.path == "/migrate/rollback":
                ok = self.project.rollback_migration(payload["migration_id"])
                save_project(self.project)
                self._json({"rolled_back": ok})
            elif parsed.path == "/health":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                self._json(self._health_payload(relationship_id))
            elif parsed.path == "/guardian-summary":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                summary = self.project.generate_guardian_summary(
                    relationship_id,
                    period_start=datetime.fromisoformat(payload["start"]) if payload.get("start") else None,
                    period_end=datetime.fromisoformat(payload["end"]) if payload.get("end") else None,
                )
                save_project(self.project)
                self._json(
                    {
                        "summary_id": summary.summary_id,
                        "relationship_id": summary.relationship_id,
                        "period_start": summary.period_start.isoformat(),
                        "period_end": summary.period_end.isoformat(),
                        "generated_at": summary.generated_at.isoformat(),
                        "user_age": summary.user_age,
                        "stage": summary.stage.value,
                        "interaction_count": summary.interaction_count,
                        "total_minutes": summary.total_minutes,
                        "memory_count": summary.memory_count,
                        "emotional_memory_count": summary.emotional_memory_count,
                        "active_behavior_count": summary.active_behavior_count,
                        "health_alert_ids": summary.health_alert_ids,
                        "milestone_count": summary.milestone_count,
                        "core_identity_count": summary.core_identity_count,
                        "recommendation": summary.recommendation,
                        "privacy_boundary": summary.privacy_boundary,
                        "resource_summary": summary.resource_summary,
                    }
                )
            elif parsed.path == "/reset/request":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    self._json({"error": "relationship not found"}, HTTPStatus.BAD_REQUEST)
                    return
                request = self.project.request_reset(relationship_id, ResetMode(payload["mode"]))
                save_project(self.project)
                self._json(
                    {
                        "request_id": request.request_id,
                        "relationship_id": request.relationship_id,
                        "mode": request.mode.value,
                        "status": request.status.value,
                        "execute_after": request.execute_after.isoformat(),
                    }
                )
            elif parsed.path == "/reset/confirm":
                ok = self.project.confirm_reset(payload["request_id"], force=self._bool_value(payload.get("force", False)))
                save_project(self.project)
                self._json(
                    {
                        "confirmed": ok,
                        "relationship_ending_support": self.project.relationship_ending_support(payload["request_id"]),
                    }
                )
            elif parsed.path == "/reset/cancel":
                self.project.cancel_reset(payload["request_id"])
                save_project(self.project)
                self._json({"cancelled": payload["request_id"]})
            elif parsed.path == "/user/age":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                self.project.set_user_age(relationship_id, self._int_body(payload, "age", minimum=0))
                save_project(self.project)
                relationship = self.project.relationships[relationship_id]
                self._json(
                    {
                        "relationship_id": relationship_id,
                        "user_age": relationship.user_age,
                        "stage": relationship.stage.value,
                    }
                )
            elif parsed.path == "/interaction/minutes":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                self.project.record_interaction_minutes(
                    relationship_id,
                    payload["date"],
                    self._int_body(payload, "minutes", minimum=0),
                )
                save_project(self.project)
                self._json(
                    {
                        "relationship_id": relationship_id,
                        "date": payload["date"],
                        "minutes": self.project.relationships[relationship_id].daily_interaction_minutes[payload["date"]],
                    }
                )
            elif parsed.path == "/mode":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                event = self.project.set_mode(
                    relationship_id,
                    Mode(payload["mode"]),
                    custom_profile=payload.get("custom_profile"),
                    reason=payload.get("reason", "user_mode_switch"),
                )
                save_project(self.project)
                relationship = self.project.relationships[relationship_id]
                self._json(
                    {
                        **event,
                        "relationship_id": relationship_id,
                        "mode": relationship.preferences.mode.value,
                        "custom_profile": relationship.preferences.custom_profile,
                        "preferences": self.project._to_json(relationship.preferences),
                    }
                )
            elif parsed.path == "/preference":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                event = self.project.set_preference(
                    relationship_id,
                    payload["key"],
                    str(payload["value"]),
                    reason=payload.get("reason", "user_preference"),
                )
                save_project(self.project)
                self._json(
                    {
                        **event,
                        "preferences": self.project._to_json(self.project.relationships[relationship_id].preferences),
                    }
                )
            elif parsed.path == "/decay-curve":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                event = self.project.set_decay_curve_type(
                    relationship_id,
                    payload["curve"],
                    reason=payload.get("reason", "privacy_panel"),
                )
                save_project(self.project)
                self._json(
                    {
                        **event,
                        "decay_curve_type": self.project.relationships[relationship_id].decay_curve_type.value,
                        "preferences": self.project._to_json(self.project.relationships[relationship_id].preferences),
                    }
                )
            elif parsed.path == "/custom-profile":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                event = self.project.update_custom_mode_profile(
                    relationship_id,
                    dict(payload.get("profile") or {}),
                    reason=payload.get("reason", "user_custom_profile"),
                )
                save_project(self.project)
                self._json(
                    {
                        **event,
                        "preferences": self.project._to_json(self.project.relationships[relationship_id].preferences),
                    }
                )
            elif parsed.path == "/stage/rollback":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                result = self.project.rollback_stage(
                    relationship_id,
                    history_index=payload.get("history_index"),
                    reason=payload.get("reason", "user_stage_rollback"),
                )
                save_project(self.project)
                self._json(result)
            elif parsed.path == "/story/correct":
                correction = self.project.correct_story_consensus(
                    payload["story_id"],
                    payload["consensus"],
                    reason=payload.get("reason", "user_correction"),
                )
                save_project(self.project)
                self._json(correction)
            elif parsed.path == "/story/confirm":
                event = self.project.confirm_story_consensus(
                    payload["story_id"],
                    note=payload.get("note"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/story/rollback":
                event = self.project.rollback_story_narrative(
                    payload["story_id"],
                    version_index=payload.get("version_index"),
                    reason=payload.get("reason", "user_story_rollback"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/transparency/ack":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                self.project.acknowledge_transparency(relationship_id)
                save_project(self.project)
                self._json(self.project.transparency_panel(relationship_id))
            elif parsed.path == "/memory-writes":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                event = self.project.set_memory_writes(
                    relationship_id,
                    self._bool_value(payload.get("enabled", True)),
                    reason=payload.get("reason", "user_control"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/health/ack":
                alert = self.project.acknowledge_health_alert(payload["alert_id"], note=payload.get("note"))
                save_project(self.project)
                self._json(
                    {
                        "alert_id": alert.alert_id,
                        "acknowledged": alert.acknowledged,
                        "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                        "acknowledgement_note": alert.acknowledgement_note,
                    }
                )
            elif parsed.path == "/health/feedback":
                event = self.project.record_health_alert_feedback(
                    payload["alert_id"],
                    payload["feedback"],
                    note=payload.get("note"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/active-feedback":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                result = self.project.record_active_feedback(relationship_id, payload["active_id"], payload["reaction"])
                save_project(self.project)
                self._json(result)
            elif parsed.path == "/active-type/mute":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                event = self.project.mute_active_type(
                    relationship_id,
                    payload["active_type"],
                    days=self._int_body(payload, "days", default=90, minimum=1),
                    reason=payload.get("reason", "user_active_type_mute"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/active-type/unmute":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                event = self.project.unmute_active_type(
                    relationship_id,
                    payload["active_type"],
                    reason=payload.get("reason", "user_active_type_unmute"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/implicit-topic/feedback":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                result = self.project.record_implicit_topic_feedback(relationship_id, payload["topic_id"], payload["reaction"])
                save_project(self.project)
                self._json(result)
            elif parsed.path == "/memory/inject":
                relationship_id = payload.get("relationship_id") or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                if relationship_id not in self.project.relationships:
                    user, ai = relationship_id.split(":", 1)
                    self.project.get_or_create_relationship(user, ai)
                memory_id = self.project.inject_memory(
                    relationship_id,
                    payload["text"],
                    memory_type=MemoryType(payload.get("memory_type", MemoryType.SHARED_EPISODE.value)),
                    context_tag=ContextTag(payload.get("context_tag", ContextTag.GENERAL.value)),
                    milestone=self._bool_value(payload.get("milestone", False)),
                    core_identity=self._bool_value(payload.get("core_identity", False)),
                    force_memory_write=self._bool_value(payload.get("force_memory_write", False)),
                )
                save_project(self.project)
                memory = self.project.memories[memory_id]
                self._json(
                    {
                        "memory_id": memory_id,
                        "relationship_id": relationship_id,
                        "memory_type": memory.memory_type.value,
                        "context_tag": memory.context_tag.value,
                        "storage_layer": memory.storage_layer.value,
                    }
                )
            elif parsed.path == "/memory/edit":
                self.project.edit_memory(
                    payload["memory_id"],
                    payload["text"],
                    reason=payload.get("reason", "user_edit"),
                )
                save_project(self.project)
                memory = self.project.memories[payload["memory_id"]]
                self._json(
                    {
                        "memory_id": memory.memory_id,
                        "content": memory.content,
                        "version_count": len(memory.metadata.get("versions", [])),
                    }
                )
            elif parsed.path == "/memory/retag":
                self.project.retag_memory(
                    payload["memory_id"],
                    memory_type=MemoryType(payload["memory_type"]) if payload.get("memory_type") else None,
                    context_tag=ContextTag(payload["context_tag"]) if payload.get("context_tag") else None,
                    reason=payload.get("reason", "user_retag"),
                )
                save_project(self.project)
                memory = self.project.memories[payload["memory_id"]]
                self._json(
                    {
                        "memory_id": memory.memory_id,
                        "memory_type": memory.memory_type.value,
                        "context_tag": memory.context_tag.value,
                        "storage_layer": memory.storage_layer.value,
                    }
                )
            elif parsed.path == "/memory/mark-milestone":
                self.project.mark_milestone(payload["memory_id"])
                save_project(self.project)
                memory = self.project.memories[payload["memory_id"]]
                self._json(
                    {
                        "memory_id": memory.memory_id,
                        "memory_type": memory.memory_type.value,
                        "context_tag": memory.context_tag.value,
                        "storage_layer": memory.storage_layer.value,
                    }
                )
            elif parsed.path == "/inside-joke/status":
                event = self.project.set_inside_joke_status(
                    payload["memory_id"],
                    active=str(payload.get("action", "deactivate")).lower() == "reactivate",
                    reason=payload.get("reason", "user_inside_joke_control"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/thread/resolve":
                event = self.project.resolve_unresolved_thread(
                    payload["memory_id"],
                    resolution=payload.get("resolution", "completed"),
                    reason=payload.get("reason", "user_thread_control"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/memory/downgrade":
                self.project.downgrade_memory(payload["memory_id"], reason=payload.get("reason", "user_downgrade"))
                save_project(self.project)
                memory = self.project.memories[payload["memory_id"]]
                self._json(
                    {
                        "memory_id": memory.memory_id,
                        "memory_type": memory.memory_type.value,
                        "context_tag": memory.context_tag.value,
                        "storage_layer": memory.storage_layer.value,
                    }
                )
            elif parsed.path == "/memory/suppress":
                event = self.project.suppress_memory(payload["memory_id"], reason=payload.get("reason", "user_boundary"))
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/memory/unsuppress":
                event = self.project.unsuppress_memory(
                    payload["memory_id"],
                    reason=payload.get("reason", "user_boundary_removed"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/memory/restore-archive":
                event = self.project.restore_archived_memory(
                    payload["memory_id"],
                    reason=payload.get("reason", "user_restore_archive"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/memory/verify":
                event = self.project.verify_memory(payload["memory_id"], reason=payload.get("reason", "user_verified"))
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/memory/calibrate":
                event = self.project.calibrate_memory(
                    payload["memory_id"],
                    payload["outcome"],
                    reason=payload.get("reason", "user_calibration"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/retention/feedback":
                event = self.project.record_retention_feedback(
                    payload["memory_id"],
                    payload["outcome"],
                    reason=payload.get("reason", "user_retention_feedback"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/time-conflict/resolve":
                event = self.project.resolve_time_conflict(
                    payload["conflict_id"],
                    resolution=payload["resolution"],
                    preferred_memory_id=payload.get("preferred_memory_id"),
                    note=payload.get("note"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/milestone/confirm":
                self.project.confirm_milestone(
                    payload["memory_id"],
                    title=payload.get("title"),
                    description=payload.get("description"),
                )
                save_project(self.project)
                self._json({"confirmed": payload["memory_id"]})
            elif parsed.path == "/milestone/edit":
                event = self.project.edit_milestone(
                    payload["memory_id"],
                    title=payload.get("title"),
                    description=payload.get("description"),
                )
                save_project(self.project)
                self._json(event)
            elif parsed.path == "/milestone/reject":
                self.project.reject_milestone(payload["memory_id"], reason=payload.get("reason", "user_rejected"))
                save_project(self.project)
                self._json({"rejected": payload["memory_id"]})
            elif parsed.path == "/batch-downgrade":
                memory_ids = payload.get("memory_ids") or None
                relationship_id = payload.get("relationship_id")
                if not memory_ids:
                    relationship_id = relationship_id or f"{payload.get('user', 'user')}:{payload.get('ai', 'companion')}"
                downgraded = self.project.batch_downgrade_memories(
                    relationship_id,
                    memory_ids=memory_ids,
                    memory_type=MemoryType(payload["memory_type"]) if payload.get("memory_type") else None,
                    context_tag=ContextTag(payload["context_tag"]) if payload.get("context_tag") else None,
                    storage_layer=MemoryLayer(payload["storage_layer"]) if payload.get("storage_layer") else None,
                    reason=payload.get("reason", "user_batch_downgrade"),
                )
                save_project(self.project)
                self._json({"downgraded": downgraded, "count": len(downgraded)})
            elif parsed.path == "/reminder/complete":
                self.project.complete_commitment_reminder(payload["reminder_id"])
                save_project(self.project)
                self._json({"completed": payload["reminder_id"]})
            elif parsed.path == "/memory/delete-request":
                request = self.project.request_memory_delete(
                    payload["memory_id"],
                    reason=payload.get("reason", "user_delete"),
                )
                save_project(self.project)
                self._json(
                    {
                        "request_id": request.request_id,
                        "memory_id": request.memory_id,
                        "status": request.status.value,
                        "execute_after": request.execute_after.isoformat(),
                    }
                )
            elif parsed.path == "/memory/delete-confirm":
                ok = self.project.confirm_memory_delete(
                    payload["request_id"],
                    force=self._bool_value(payload.get("force", False)),
                )
                save_project(self.project)
                self._json({"confirmed": ok})
            elif parsed.path == "/memory/delete-cancel":
                self.project.cancel_memory_delete(payload["request_id"])
                save_project(self.project)
                self._json({"cancelled": payload["request_id"]})
            elif parsed.path == "/l4/delete-request":
                request = self.project.request_core_identity_delete(
                    payload["identity_id"],
                    reason=payload.get("reason", "user_delete"),
                )
                save_project(self.project)
                self._json(
                    {
                        "request_id": request.request_id,
                        "identity_id": request.identity_id,
                        "status": request.status.value,
                        "execute_after": request.execute_after.isoformat(),
                    }
                )
            elif parsed.path == "/l4/delete-confirm":
                ok = self.project.confirm_core_identity_delete(
                    payload["request_id"],
                    force=self._bool_value(payload.get("force", False)),
                )
                save_project(self.project)
                self._json({"confirmed": ok})
            elif parsed.path == "/l4/delete-cancel":
                self.project.cancel_core_identity_delete(payload["request_id"])
                save_project(self.project)
                self._json({"cancelled": payload["request_id"]})
            elif parsed.path == "/l4/review":
                entry = self.project.confirm_core_identity_review(
                    payload["identity_id"],
                    decision=payload.get("decision", "confirm"),
                    reason=payload.get("reason", "user_confirmed"),
                )
                save_project(self.project)
                self._json(entry)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except PermissionError as exc:
            self._json({"error": str(exc), "error_type": "PermissionError"}, HTTPStatus.FORBIDDEN)
        except Exception as exc:  # pragma: no cover - handler safety path
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _relationship_id(self, query: dict[str, list[str]]) -> str:
        if "relationship_id" in query:
            return query["relationship_id"][0]
        return f"{self._one(query, 'user', 'user')}:{self._one(query, 'ai', 'companion')}"

    def _one(self, query: dict[str, list[str]], key: str, default: str) -> str:
        return query.get(key, [default])[0]

    def _int_query(self, query: dict[str, list[str]], key: str, default: int, *, minimum: int | None = None) -> int:
        raw = self._one(query, key, str(default))
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer") from None
        if minimum is not None and value < minimum:
            raise ValueError(f"{key} must be >= {minimum}")
        return value

    def _int_body(self, payload: dict, key: str, default: int | None = None, *, minimum: int | None = None) -> int:
        if key not in payload:
            if default is None:
                raise ValueError(f"{key} is required")
            raw = default
        else:
            raw = payload[key]
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer") from None
        if minimum is not None and value < minimum:
            raise ValueError(f"{key} must be >= {minimum}")
        return value

    def _body(self) -> dict:
        try:
            size = int(self.headers.get("content-length", "0"))
        except (TypeError, ValueError):
            raise ValueError("content-length must be an integer") from None
        raw = self.rfile.read(size).decode("utf-8") if size else "{}"
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            raise ValueError("request body must be valid JSON") from None
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _bool_value(self, value: object) -> bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on", "enable", "enabled", "resume", "开"}
        return bool(value)

    def _health_payload(self, relationship_id: str) -> dict:
        if relationship_id not in self.project.relationships:
            user, ai = relationship_id.split(":", 1)
            self.project.get_or_create_relationship(user, ai)
        self.project.evaluate_health(relationship_id)
        save_project(self.project)
        return {
            "relationship_id": relationship_id,
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "risk_type": alert.risk_type,
                    "level": alert.level.value,
                    "message": alert.message,
                    "acknowledged": alert.acknowledged,
                    "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    "acknowledgement_note": alert.acknowledgement_note,
                    "resources": alert.resources,
                }
                for alert in self.project.health_alerts.values()
                if alert.relationship_id == relationship_id
            ],
        }

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _text(self, payload: str, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        return


def configure_state(state_path: str | Path) -> None:
    global STATE_PATH
    STATE_PATH = Path(state_path)
    with FriendMemoryHandler.project_lock:
        FriendMemoryHandler.project = load_project()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the friend memory HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--state",
        help=(
            "state JSON path for this server process; overrides FRIEND_MEMORY_STATE "
            "without changing the default project state file"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.state:
        configure_state(args.state)
    server = ThreadingHTTPServer((args.host, args.port), FriendMemoryHandler)
    print(f"serving http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
