from __future__ import annotations

import json
import re
import statistics
import hashlib
import time
from copy import deepcopy
from csv import DictWriter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from ai import (
    MemoryAI,
    ai_provider_name,
    build_memory_ai_from_env,
    consume_ai_call_metadata,
    describe_memory_ai,
    provider_participation_kind,
)
from models import (
    CommitmentReminder,
    ContextTag,
    CoreIdentityDeleteRequest,
    DecayCurve,
    EmotionalBaseline,
    EmotionalMemory,
    EmotionalTrajectory,
    EmotionalTrajectoryWindow,
    EmotionLabel,
    GuardianSummary,
    HealthAlert,
    HealthRiskLevel,
    InteractionPatterns,
    CoreIdentityRecord,
    MemoryDeleteRequest,
    MemoryGraphEdge,
    MemoryLayer,
    MemoryRecord,
    MemoryType,
    Mode,
    NarrativeLevel,
    Relationship,
    RelationshipNarrative,
    RelationshipStage,
    ReminderStatus,
    ResetMode,
    ResetRequest,
    ResetRequestStatus,
    SharedStoryNode,
    UserPreferences,
    new_id,
    utcnow,
)
from scoring import (
    apply_trust_bias,
    clamp,
    is_trust_bias_protected,
    lexical_similarity,
    memory_weight,
    retention_calibration_multiplier,
    temporal_fuzz,
    tokenize,
    trust_bias_stage_enabled,
)
from signals import TurnSignals, detect_turn_signals


@dataclass
class IngestResult:
    relationship_id: str
    memory_id: str | None
    emotional_memory_id: str | None
    score: float
    stage: RelationshipStage
    active_suggestions: list[str]
    signals: TurnSignals
    active_events: list[dict[str, Any]]
    memory_paused: bool = False


@dataclass
class RetrievalResult:
    memory: MemoryRecord
    score: float
    weight: float
    presentation_time: dict[str, Any]
    explanation: dict[str, Any]


@dataclass
class ConsolidationReport:
    relationship_id: str
    replayed_memories: int
    upgraded_stories: int
    downgraded_memories: list[str]
    archived_memories: list[str]
    compressed_stories: list[str]
    health_alerts: list[str]


@dataclass
class MigrationReport:
    migration_id: str
    imported_turns: int
    relationship_ids: list[str]
    created_memories: list[str]
    created_emotional_memories: list[str]
    recognized_milestones: list[str]
    rollback_expires_at: datetime


RETENTION_MULTIPLIER_BY_STAGE = {
    RelationshipStage.INITIATING: 1.0,
    RelationshipStage.EXPERIMENTING: 1.0,
    RelationshipStage.INTENSIFYING: 1.2,
    RelationshipStage.INTEGRATING: 1.8,
    RelationshipStage.BONDING: 2.5,
    RelationshipStage.DIFFERENTIATING: 1.2,
    RelationshipStage.CIRCUMSCRIBING: 1.0,
    RelationshipStage.STAGNATING: 0.8,
    RelationshipStage.AVOIDING: 0.6,
    RelationshipStage.TERMINATING: 0.4,
}


class FriendMemoryProject:
    """Runnable project core for the friend-style relational memory system.

    This is intentionally implemented as application code, not as a packaged SDK.
    It uses deterministic rule-based detectors so the system is runnable without
    external models. The boundaries mirror the design document and can later be
    swapped for LLM/embedding-backed implementations.
    """

    PERSISTED_STATE_FIELDS = (
        "relationships",
        "memories",
        "memory_graph_edges",
        "emotional_memories",
        "story_nodes",
        "emotional_trajectories",
        "core_identity",
        "core_identity_delete_requests",
        "memory_delete_requests",
        "reset_requests",
        "health_alerts",
        "guardian_summaries",
        "commitment_reminders",
        "retrieval_audit_log",
        "ai_decision_log",
        "deviation_log",
        "deletion_compliance_log",
        "relationship_ending_support_log",
        "migration_batches",
    )
    USER_MUTABLE_ACTIVE_TYPES = {
        "anniversary",
        "inside_joke",
        "implicit_topic",
        "shared_topic_reactivation",
        "emotional_resonance",
        "commitment_reminder",
    }

    def __init__(self, ai: MemoryAI | None = None) -> None:
        self.ai = ai or build_memory_ai_from_env()
        self.relationships: dict[str, Relationship] = {}
        self.memories: dict[str, MemoryRecord] = {}
        self.memory_graph_edges: dict[str, MemoryGraphEdge] = {}
        self.emotional_memories: dict[str, EmotionalMemory] = {}
        self.story_nodes: dict[str, SharedStoryNode] = {}
        self.emotional_trajectories: dict[str, EmotionalTrajectory] = {}
        self.core_identity: dict[str, CoreIdentityRecord] = {}
        self.core_identity_delete_requests: dict[str, CoreIdentityDeleteRequest] = {}
        self.memory_delete_requests: dict[str, MemoryDeleteRequest] = {}
        self.reset_requests: dict[str, ResetRequest] = {}
        self.health_alerts: dict[str, HealthAlert] = {}
        self.guardian_summaries: dict[str, GuardianSummary] = {}
        self.commitment_reminders: dict[str, CommitmentReminder] = {}
        self.retrieval_audit_log: list[dict[str, Any]] = []
        self.ai_decision_log: list[dict[str, Any]] = []
        self.deviation_log: list[dict[str, Any]] = []
        self.deletion_compliance_log: list[dict[str, Any]] = []
        self.relationship_ending_support_log: list[dict[str, Any]] = []
        self.migration_batches: dict[str, dict[str, Any]] = {}
        self._pending_active_metadata: dict[str, dict[str, Any]] = {}

    def get_or_create_relationship(self, user_id: str, ai_id: str) -> Relationship:
        relationship_id = f"{user_id}:{ai_id}"
        if relationship_id not in self.relationships:
            relationship = Relationship(user_id=user_id, ai_id=ai_id, relationship_id=relationship_id)
            relationship.retention_multiplier = self._retention_multiplier_for_stage(relationship.stage)
            relationship.stage_history.append(
                {
                    "from": None,
                    "to": relationship.stage.value,
                    "at": relationship.created_at.isoformat(),
                    "reason": "relationship_created",
                    "strength": relationship.strength,
                    "trust_level": relationship.trust_level,
                    "intimacy_level": relationship.intimacy_level,
                    "retention_multiplier": relationship.retention_multiplier,
                }
            )
            self.relationships[relationship_id] = relationship
        return self.relationships[relationship_id]

    def ingest_turn(
        self,
        user_id: str,
        ai_id: str,
        text: str,
        *,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        now = timestamp or utcnow()
        metadata = metadata or {}
        relationship = self.get_or_create_relationship(user_id, ai_id)
        self._update_relationship_clock(relationship, now)

        signals = detect_turn_signals(text, relationship.strength, relationship.trust_level, relationship.intimacy_level)
        if not relationship.preferences.memory_writes_enabled:
            self.deviation_log.append(
                {
                    "type": "memory_write_skipped",
                    "relationship_id": relationship.relationship_id,
                    "reason": relationship.preferences.memory_pause_reason or "memory_writes_disabled",
                    "text_chars": len(text),
                    "at": now.isoformat(),
                }
            )
            return IngestResult(
                relationship_id=relationship.relationship_id,
                memory_id=None,
                emotional_memory_id=None,
                score=0.0,
                stage=relationship.stage,
                active_suggestions=[],
                signals=signals,
                active_events=[],
                memory_paused=True,
            )
        self._record_maintenance_signal(relationship, text, now)
        boundary_request = self._detect_boundary_request(text)
        if boundary_request:
            boundary_result = self._apply_boundary_request(relationship, text, boundary_request, now)
            self.deviation_log.append(boundary_result)
            return IngestResult(
                relationship_id=relationship.relationship_id,
                memory_id=None,
                emotional_memory_id=None,
                score=0.0,
                stage=relationship.stage,
                active_suggestions=[],
                signals=signals,
                active_events=[],
                memory_paused=False,
            )
        raw_ai_analysis = self.ai.analyze_turn(text, self._relationship_context(relationship))
        ai_analysis = self._sanitize_ai_analysis(raw_ai_analysis)
        ai_decision = self._log_ai_decision(
            relationship.relationship_id,
            task="analyze_turn",
            input_summary={"text_chars": len(text), "stage": relationship.stage.value},
            output_summary=ai_analysis,
            now=now,
        )
        ai_decision_summary = self.ai_decision_summary(ai_decision)
        score = self._friend_score(signals)
        score = max(score, float(ai_analysis.get("importance", 0.0) or 0.0))
        memory_type = self._memory_type_from_ai(ai_analysis) or signals.memory_type
        context_tag = self._context_tag_from_ai(ai_analysis) or signals.context_tag
        stage_encoding = self._stage_encoding_strategy(relationship, signals, memory_type, context_tag)
        memory_type = stage_encoding["memory_type"]
        context_tag = stage_encoding["context_tag"]
        decay_curve = self._decay_curve_for(memory_type, relationship)
        memory = MemoryRecord(
            memory_id=new_id("mem"),
            relationship_id=relationship.relationship_id,
            content=text,
            memory_type=memory_type,
            context_tag=context_tag,
            created_at=now,
            updated_at=now,
            base_weight=self._base_weight(score, memory_type),
            importance=score,
            emotion_intensity=signals.emotion_intensity,
            emotional_valence=signals.sentiment,
            decay_curve=decay_curve,
            relationship_stage_at_creation=relationship.stage,
            relationship_age_at_creation=relationship.relationship_age,
            trust_level_at_creation=relationship.trust_level,
            storage_layer=self._storage_layer_for(
                memory_type=memory_type,
                context_tag=context_tag,
                score=score,
                relationship=relationship,
            ),
            metadata={
                **metadata,
                **stage_encoding["metadata"],
                "ai_analysis": ai_analysis,
                "ai_decision_source": ai_decision_summary,
            },
        )
        memory.metadata["metacognition"] = self._memory_metacognition(
            memory,
            source_kind=metadata.get("source", "user_turn"),
            ai_analysis=ai_analysis,
            now=now,
        )
        self._apply_criticality_protection(relationship, memory, now=now)
        self._mark_critical_tombstone_remention(relationship, memory, now)
        self._attach_source_time(memory, text, now)
        memory.metadata["embeddings"] = self._memory_embedding_features(memory, signals)
        self.memories[memory.memory_id] = memory
        self._detect_time_conflicts(relationship, memory, now)
        self._detect_preference_supersession(relationship, memory, now)
        self._maybe_promote_major_shared_decision(relationship, memory, signals, now)
        self._maybe_promote_shared_celebration(relationship, memory, signals, now)

        emotional_memory_id = None
        if relationship.preferences.emotional_layer_enabled and signals.emotion_intensity >= self._emotional_layer_threshold(relationship):
            emotional = self._create_emotional_memory(relationship, memory, signals, now)
            self.emotional_memories[emotional.emotion_id] = emotional
            emotional_memory_id = emotional.emotion_id

        self._attach_relationship_indexes(relationship, memory, signals)
        self._maybe_create_commitment_reminder(relationship, memory, text, now)
        self._maybe_promote_core_identity(relationship, memory, signals, now)
        self._update_shared_story(relationship, memory, signals, now)
        self._maybe_attach_inside_joke_to_story(relationship, memory)
        self._update_memory_graph(relationship, memory, signals, now)
        self._maybe_set_origin_story(relationship, memory, signals, now)
        self._update_relationship_state(relationship, signals, now)
        self._update_emotional_baseline(relationship, text, signals, now)
        self._update_emotional_trajectory(relationship, memory, signals, now)
        self._evaluate_turn_safety(relationship, memory, now)
        self._apply_minor_stage_limit(relationship)
        if metadata.get("benchmark_disable_active_recall"):
            active = []
            active_events = []
        else:
            active = self.active_suggestions(relationship.relationship_id, current_text=text, now=now, current_signals=signals)
            active_events = self._log_active_behavior(relationship, active, now)

        return IngestResult(
            relationship_id=relationship.relationship_id,
            memory_id=memory.memory_id,
            emotional_memory_id=emotional_memory_id,
            score=score,
            stage=relationship.stage,
            active_suggestions=active,
            signals=signals,
            active_events=active_events,
        )

    def ingest_exchange(
        self,
        user_id: str,
        ai_id: str,
        user_text: str,
        assistant_text: str,
        *,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        metadata = dict(metadata or {})
        metadata.update(
            {
                "source": "dialogue_exchange",
                "exchange": {
                    "user": user_text,
                    "assistant": assistant_text,
                },
            }
        )
        combined_text = f"用户：{user_text}\n{ai_id}：{assistant_text}"
        result = self.ingest_turn(
            user_id,
            ai_id,
            combined_text,
            timestamp=timestamp,
            metadata=metadata,
        )
        if result.memory_id:
            self._apply_assistant_reply_memory_effects(
                result.relationship_id,
                result.memory_id,
                user_text,
                assistant_text,
                timestamp or utcnow(),
            )
            memory = self.memories[result.memory_id]
            if memory.metadata.get("assistant_reply_outcome") in {"declined", "cancelled"}:
                relationship = self.relationships[result.relationship_id]
                active_ids = {event.get("active_id") for event in result.active_events}
                relationship.active_behavior_log = [
                    event for event in relationship.active_behavior_log if event.get("active_id") not in active_ids
                ]
                result.active_suggestions.clear()
                result.active_events.clear()
        return result

    def retrieve(
        self,
        relationship_id: str,
        query: str,
        *,
        now: datetime | None = None,
        limit: int = 5,
        audit: bool = True,
        include_archived: bool = False,
    ) -> list[RetrievalResult]:
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        current = detect_turn_signals(query, relationship.strength, relationship.trust_level, relationship.intimacy_level)
        retrieval_weights = self._retrieval_weights(relationship, current, query)
        results: list[RetrievalResult] = []
        for memory in self.memories.values():
            if memory.relationship_id != relationship_id:
                continue
            if memory.metadata.get("archived") and not include_archived:
                continue
            if self._memory_pending_sensitive_confirmation(memory):
                continue
            if self._memory_is_recall_suppressed(memory):
                continue
            composite = self._composite_similarity(query, current, memory)
            semantic = composite["composite"]
            emotional = self._emotional_resonance(current, memory)
            relation = self._relationship_relevance(relationship, memory)
            precious = self._time_preciousness(relationship, memory, now)
            component_score = (
                retrieval_weights["emotional_resonance"] * emotional
                + retrieval_weights["relationship_relevance"] * relation
                + retrieval_weights["time_preciousness"] * precious
                + retrieval_weights["semantic"] * semantic
            )
            weight = memory_weight(memory, relationship, now)
            weighted_score = component_score * weight
            metacognition = self._ensure_memory_metacognition(memory, now)
            weighted_score *= metacognition["score_multiplier"]
            weighted_score *= self._timestamp_score_multiplier(metacognition)
            score = apply_trust_bias(weighted_score, memory, relationship)
            trust_presentation = self._trust_presentation(memory, relationship, score, weighted_score)
            results.append(
                RetrievalResult(
                    memory=memory,
                    score=score,
                    weight=weight,
                    presentation_time=self._presentation_time(relationship, memory, now),
                    explanation={
                        "semantic": semantic,
                        "lexical_similarity": composite["semantic"],
                        "composite_embedding": composite,
                        "emotional_resonance": emotional,
                        "relationship_relevance": relation,
                        "time_preciousness": precious,
                        "component_score": component_score,
                        "memory_weight": weight,
                        "weighted_score": weighted_score,
                        "final_score": score,
                        "metacognition": metacognition,
                        "source_time": memory.metadata.get("source_time"),
                        "trust_bias_applied": abs(score - weighted_score) > 1e-9,
                        "trust_presentation": trust_presentation,
                        "display_content": trust_presentation["display_content"],
                        "trust_level": relationship.trust_level,
                        "weights": retrieval_weights,
                    },
                )
            )
        self._apply_graph_retrieval_boost(results)
        ranked = sorted(results, key=lambda item: item.score, reverse=True)[:limit]
        retrieval_adaptation = self._apply_retrieval_adaptation(relationship, query, results, ranked, now)
        query_metacognition = self._query_metacognition(relationship, query, ranked)
        association_expansions = self._association_expansions(relationship, ranked, now)
        story_clusters = self._story_clusters_for_results(relationship, ranked)
        for item in ranked:
            item.explanation["query_metacognition"] = query_metacognition
            item.explanation["association_expansions"] = association_expansions
            item.explanation["story_clusters"] = story_clusters
            item.explanation["retrieval_adaptation"] = retrieval_adaptation
        if audit:
            self._log_retrieval(
                relationship,
                query,
                ranked,
                now,
                include_archived=include_archived,
                query_metacognition=query_metacognition,
                association_expansions=association_expansions,
                story_clusters=story_clusters,
                retrieval_adaptation=retrieval_adaptation,
            )
        return ranked

    def consolidate_relationship(self, relationship_id: str, *, now: datetime | None = None) -> ConsolidationReport:
        """Run an offline consolidation pass.

        This implements the project-level equivalent of the design's four-stage
        offline pipeline: replay/re-evaluate, abstract stories, reorganize
        relationship state, and compress/archive low-value details.
        """
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        self._apply_inactivity_trust_decay(relationship, now)
        memories = [memory for memory in self.memories.values() if memory.relationship_id == relationship_id]
        replayed = 0
        downgraded: list[str] = []
        archived: list[str] = []

        for memory in memories:
            replayed += 1
            if (
                relationship.preferences.trust_bias_enabled
                and relationship.trust_level >= 0.8
                and trust_bias_stage_enabled(relationship)
                and not is_trust_bias_protected(memory)
                and (
                    memory.context_tag == ContextTag.CONFLICT
                    or memory.memory_type == MemoryType.CONFLICT
                    or memory.emotional_valence <= -0.5
                )
            ):
                memory.metadata["trust_soft_cooldown"] = {
                    "at": now.isoformat(),
                    "trust_level": relationship.trust_level,
                    "retention_multiplier": 1.0,
                    "original_preserved": True,
                }
            weight = memory_weight(memory, relationship, now)
            protected_content, input_protection = self._external_ai_memory_content(memory)
            ai_value = self.ai.evaluate_memory_value(protected_content, self._relationship_context(relationship))
            memory.metadata["last_ai_value"] = ai_value
            memory.metadata["last_ai_value_source"] = {
                "provider": self._ai_provider_name(),
                "task": "evaluate_memory_value",
                "at": now.isoformat(),
                "input_protection": input_protection,
            }
            self._log_ai_decision(
                relationship_id,
                task="evaluate_memory_value",
                input_summary={
                    "memory_id": memory.memory_id,
                    "content_chars": len(memory.content),
                    "weight": weight,
                    "input_protection": input_protection,
                },
                output_summary={"value": ai_value},
                now=now,
            )
            self._review_cold_information(relationship, memory, ai_value, now)
            self._evaluate_perturbation_replay(memory, now)
            weight = memory_weight(memory, relationship, now)
            if (
                memory.decay_curve == DecayCurve.REVERSE_DECAY
                and memory.memory_type not in {MemoryType.MILESTONE, MemoryType.IDENTITY}
                and weight < 0.30
                and memory.importance < 0.50
                and ai_value < 0.30
            ):
                self.downgrade_memory(memory.memory_id, reason="offline_low_value", now=now)
                downgraded.append(memory.memory_id)
            if (now - memory.created_at).days >= 365 and weight < 0.30:
                memory.metadata["archived"] = True
                memory.metadata["archived_at"] = now.isoformat()
                memory.metadata["cold_archive"] = self._cold_archive_reference(
                    memory,
                    relationship,
                    weight=weight,
                    ai_value=ai_value,
                    now=now,
                )
                archived.append(memory.memory_id)

        upgraded = 0
        compressed: list[str] = []
        for story in [item for item in self.story_nodes.values() if item.relationship_id == relationship_id]:
            if isinstance(story.consensus_provenance, dict) and story.consensus_provenance.get("requires_schema_rebuild"):
                if not self._rebuild_story_after_deleted_source(story, relationship, now):
                    continue
            before = story.narrative_level
            previous_consensus = story.consensus_version
            next_level = self._narrative_level(story, relationship)
            if next_level != before:
                story.narrative_level = next_level
                upgraded += 1
            summary = self._summarize_story(story)
            if summary and summary != story.consensus_version:
                story.consensus_version = summary
                compressed.append(story.story_id)
            if story.narrative_level != before or story.consensus_version != previous_consensus:
                self._record_story_narrative_version(
                    story,
                    previous_level=before,
                    previous_consensus=previous_consensus,
                    reason="offline_consolidation",
                    now=now,
                )

        self._discover_implicit_topics(relationship, memories, now)
        self._recompute_relationship_from_current_state(relationship, now)
        alerts = [alert.alert_id for alert in self.evaluate_health(relationship_id, now=now)]
        return ConsolidationReport(
            relationship_id=relationship_id,
            replayed_memories=replayed,
            upgraded_stories=upgraded,
            downgraded_memories=downgraded,
            archived_memories=archived,
            compressed_stories=compressed,
            health_alerts=alerts,
        )

    def migrate_legacy_turns(
        self,
        turns: list[dict[str, Any] | str],
        *,
        default_user: str = "user",
        default_ai: str = "companion",
        relationship_certificate: dict[str, Any] | None = None,
        require_certificate: bool = False,
        target_mode: Mode | str | None = None,
        now: datetime | None = None,
    ) -> MigrationReport:
        """Replay legacy raw conversation logs into the relationship-memory system."""
        now = now or utcnow()
        migration_id = new_id("migration")
        rollback_snapshot = deepcopy(self.export())
        rollback_snapshot["migration_batches"] = {
            key: {field: value for field, value in batch.items() if field != "rollback_snapshot"}
            for key, batch in rollback_snapshot.get("migration_batches", {}).items()
        }
        normalized = sorted(
            [self._normalize_legacy_turn(item, default_user, default_ai, now) for item in turns],
            key=lambda item: item["timestamp"],
        )
        expected_certificate = self._build_migration_certificate(normalized, now=now)
        certificate_status = self._verify_migration_certificate(
            expected_certificate,
            relationship_certificate,
            require_certificate=require_certificate,
            now=now,
        )

        relationship_ids: set[str] = set()
        created_memories: list[str] = []
        created_emotional_memories: list[str] = []
        recognized_milestones: list[str] = []

        for item in normalized:
            result = self.ingest_turn(
                item["user"],
                item["ai"],
                item["text"],
                timestamp=item["timestamp"],
                metadata={"source": "legacy_migration", "migration_id": migration_id, **item["metadata"]},
            )
            relationship_ids.add(result.relationship_id)
            created_memories.append(result.memory_id)
            if result.emotional_memory_id:
                created_emotional_memories.append(result.emotional_memory_id)
            if item["milestone"]:
                self.mark_milestone(result.memory_id)
                recognized_milestones.append(result.memory_id)

        normalized_target_mode = None
        if target_mode is not None:
            normalized_target_mode = target_mode if isinstance(target_mode, Mode) else Mode(str(target_mode))

        for relationship_id in sorted(relationship_ids):
            if normalized_target_mode is not None:
                self.set_mode(
                    relationship_id,
                    normalized_target_mode,
                    reason="legacy_migration_mode_choice",
                    now=now,
                )
            self.consolidate_relationship(relationship_id, now=now)

        rollback_expires_at = now + timedelta(days=30)
        self.migration_batches[migration_id] = {
            "migration_id": migration_id,
            "status": "APPLIED",
            "created_at": now.isoformat(),
            "rollback_expires_at": rollback_expires_at.isoformat(),
            "relationship_ids": sorted(relationship_ids),
            "created_memories": created_memories,
            "created_emotional_memories": created_emotional_memories,
            "recognized_milestones": recognized_milestones,
            "relationship_certificate": {
                **expected_certificate,
                "status": certificate_status["status"],
                "verified": certificate_status["verified"],
                "provided": relationship_certificate is not None,
            },
            "target_mode": normalized_target_mode.value if normalized_target_mode else None,
            "rollback_snapshot": rollback_snapshot,
        }
        self.deviation_log.append(
            {
                "type": "legacy_migration_applied",
                "migration_id": migration_id,
                "imported_turns": len(normalized),
                "relationship_ids": sorted(relationship_ids),
                "certificate_status": certificate_status["status"],
                "target_mode": normalized_target_mode.value if normalized_target_mode else None,
                "at": now.isoformat(),
            }
        )
        return MigrationReport(
            migration_id=migration_id,
            imported_turns=len(normalized),
            relationship_ids=sorted(relationship_ids),
            created_memories=created_memories,
            created_emotional_memories=created_emotional_memories,
            recognized_milestones=recognized_milestones,
            rollback_expires_at=rollback_expires_at,
        )

    def rollback_migration(self, migration_id: str, *, now: datetime | None = None) -> bool:
        now = now or utcnow()
        batch = self.migration_batches[migration_id]
        if batch.get("status") != "APPLIED":
            return False
        if now > _dt(batch["rollback_expires_at"]):
            return False
        restored = self._from_data(batch["rollback_snapshot"])
        rolled_back = {
            **{field: value for field, value in batch.items() if field != "rollback_snapshot"},
            "status": "ROLLED_BACK",
            "rolled_back_at": now.isoformat(),
        }
        self._restore_persisted_state_from(restored)
        self.migration_batches[migration_id] = rolled_back
        self.deviation_log.append(
            {
                "type": "legacy_migration_rolled_back",
                "migration_id": migration_id,
                "at": now.isoformat(),
            }
        )
        return True

    def _restore_persisted_state_from(self, restored: "FriendMemoryProject") -> None:
        for field_name in self.PERSISTED_STATE_FIELDS:
            setattr(self, field_name, getattr(restored, field_name))

    def request_core_identity_delete(
        self, identity_id: str, *, reason: str = "user_delete", now: datetime | None = None
    ) -> CoreIdentityDeleteRequest:
        now = now or utcnow()
        identity = self.core_identity[identity_id]
        existing = next(
            (
                request
                for request in self.core_identity_delete_requests.values()
                if request.identity_id == identity_id and request.status == ResetRequestStatus.PENDING
            ),
            None,
        )
        if existing:
            return existing
        request = CoreIdentityDeleteRequest(
            request_id=new_id("l4del"),
            identity_id=identity_id,
            relationship_id=identity.relationship_id,
            memory_id=identity.memory_id,
            requested_at=now,
            execute_after=now + timedelta(hours=24),
            reason=reason,
        )
        self.core_identity_delete_requests[request.request_id] = request
        identity.pending_delete = True
        identity.change_log.append(
            {
                "at": now.isoformat(),
                "reason": "delete_requested",
                "request_id": request.request_id,
                "delete_reason_sealed": self._seal_audit_text(reason),
            }
        )
        self._refresh_l4_replicas(identity, now=now, reason="delete_requested")
        self.deviation_log.append(
            {
                "type": "l4_delete_requested",
                "request_id": request.request_id,
                "identity_id": identity_id,
                "relationship_id": identity.relationship_id,
                "execute_after": request.execute_after.isoformat(),
                "delete_reason_sealed": self._seal_audit_text(reason),
            }
        )
        return request

    def confirm_core_identity_delete(
        self, request_id: str, *, now: datetime | None = None, force: bool = False
    ) -> bool:
        now = now or utcnow()
        request = self.core_identity_delete_requests[request_id]
        if request.status != ResetRequestStatus.PENDING:
            return False
        if not force and now < request.execute_after:
            return False
        identity = self.core_identity.pop(request.identity_id, None)
        relationship = self.relationships.get(request.relationship_id)
        if relationship and request.identity_id in relationship.core_identity:
            relationship.core_identity.remove(request.identity_id)
        memory = self.memories.get(request.memory_id)
        if memory:
            memory.metadata.setdefault("l4_delete_audit", []).append(
                {
                    "at": now.isoformat(),
                    "request_id": request_id,
                    "delete_reason_sealed": self._seal_audit_text(request.reason),
                }
            )
            memory.memory_type = MemoryType.FACT
            memory.decay_curve = DecayCurve.STANDARD_POWER_LAW
            memory.importance = min(memory.importance, 0.4)
            memory.base_weight = min(memory.base_weight, 0.5)
        request.status = ResetRequestStatus.EXECUTED
        request.executed_at = now
        self.deviation_log.append(
            {
                "type": "l4_delete_confirmed",
                "request_id": request_id,
                "identity_id": request.identity_id,
                "relationship_id": request.relationship_id,
                "executed_at": now.isoformat(),
                "identity_found": identity is not None,
            }
        )
        self._record_deletion_compliance(
            relationship_id=request.relationship_id,
            deletion_type="L4_CORE_IDENTITY_DELETE",
            request_id=request_id,
            reason=request.reason,
            now=now,
            summary={
                "identity_found": identity is not None,
                "memory_id_hash": self._seal_audit_text(request.memory_id),
                "memory_downgraded": memory is not None,
                "core_identity_removed": identity is not None,
            },
        )
        return True

    def cancel_core_identity_delete(self, request_id: str, *, now: datetime | None = None) -> None:
        now = now or utcnow()
        request = self.core_identity_delete_requests[request_id]
        request.status = ResetRequestStatus.CANCELLED
        identity = self.core_identity.get(request.identity_id)
        if identity:
            identity.pending_delete = False
            identity.change_log.append({"at": now.isoformat(), "reason": "delete_cancelled", "request_id": request_id})
            self._refresh_l4_replicas(identity, now=now, reason="delete_cancelled")
        self.deviation_log.append(
            {
                "type": "l4_delete_cancelled",
                "request_id": request_id,
                "identity_id": request.identity_id,
                "relationship_id": request.relationship_id,
                "at": now.isoformat(),
            }
        )

    def confirm_core_identity_review(
        self,
        identity_id: str,
        *,
        decision: str = "confirm",
        reason: str = "user_confirmed",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        identity = self.core_identity[identity_id]
        normalized = decision.lower().replace("-", "_")
        if normalized not in {"confirm", "reject", "needs_review"}:
            raise ValueError("decision must be confirm, reject, or needs_review")
        status_map = {
            "confirm": "USER_CONFIRMED",
            "reject": "USER_REJECTED",
            "needs_review": "NEEDS_USER_CONFIRMATION",
        }
        status = status_map[normalized]
        if normalized == "confirm":
            identity.user_confirmed_at = now
        identity.review_status = status
        entry = {
            "at": now.isoformat(),
            "change_type": "user_review",
            "review_status": status,
            "decision": normalized,
            "reason_sealed": self._seal_audit_text(reason),
            "content_sealed": self._seal_audit_text(identity.content),
        }
        identity.review_history.append(entry)
        identity.change_log.append(
            self._l4_change_entry(
                f"user_review_{normalized}",
                now=now,
                new_content=identity.content,
                extra={"review_status": status, "reason_sealed": self._seal_audit_text(reason)},
            )
        )
        self._refresh_l4_replicas(identity, now=now, reason=f"user_review_{normalized}")
        memory = self.memories.get(identity.memory_id)
        if memory:
            memory.metadata["l4_review"] = {
                "identity_id": identity.identity_id,
                "review_status": status,
                "ai_score": identity.review_score,
                "at": now.isoformat(),
                "user_confirmed": normalized == "confirm",
            }
        self.deviation_log.append(
            {
                "type": "l4_review_user_confirmed",
                "relationship_id": identity.relationship_id,
                "identity_id": identity.identity_id,
                "memory_id": identity.memory_id,
                "decision": normalized,
                "review_status": status,
                "at": now.isoformat(),
            }
        )
        return entry

    def request_memory_delete(
        self,
        memory_id: str,
        *,
        reason: str = "user_delete",
        now: datetime | None = None,
    ) -> MemoryDeleteRequest:
        now = now or utcnow()
        memory = self.memories[memory_id]
        existing = next(
            (
                request
                for request in self.memory_delete_requests.values()
                if request.memory_id == memory_id and request.status == ResetRequestStatus.PENDING
            ),
            None,
        )
        if existing:
            return existing
        request = MemoryDeleteRequest(
            request_id=new_id("memdel"),
            memory_id=memory_id,
            relationship_id=memory.relationship_id,
            requested_at=now,
            execute_after=now + timedelta(hours=24),
            reason=reason,
        )
        self.memory_delete_requests[request.request_id] = request
        memory.metadata["pending_delete"] = True
        self.deviation_log.append(
            {
                "type": "memory_delete_requested",
                "request_id": request.request_id,
                "memory_id": memory_id,
                "relationship_id": memory.relationship_id,
                "execute_after": request.execute_after.isoformat(),
                "delete_reason_sealed": self._seal_audit_text(reason),
            }
        )
        return request

    def confirm_memory_delete(
        self,
        request_id: str,
        *,
        now: datetime | None = None,
        force: bool = False,
    ) -> bool:
        now = now or utcnow()
        request = self.memory_delete_requests[request_id]
        if request.status != ResetRequestStatus.PENDING:
            return False
        if not force and now < request.execute_after:
            return False
        tombstone = self._critical_memory_tombstone_preview(request.memory_id, request.reason, now)
        deleted = self._delete_memory_and_derivatives(request.memory_id, now=now, reason=request.reason)
        request.status = ResetRequestStatus.EXECUTED
        request.executed_at = now
        self.deviation_log.append(
            {
                "type": "memory_delete_executed",
                "request_id": request_id,
                "memory_id": request.memory_id,
                "relationship_id": request.relationship_id,
                "executed_at": now.isoformat(),
                "deleted": deleted,
            }
        )
        self._record_deletion_compliance(
            relationship_id=request.relationship_id,
            deletion_type="MEMORY_DELETE",
            request_id=request_id,
            reason=request.reason,
            now=now,
            summary={
                "memory_found": deleted,
                "memory_id_hash": self._seal_audit_text(request.memory_id),
                "critical_memory_tombstone": tombstone if deleted and tombstone else None,
            },
        )
        if deleted and tombstone:
            self.deviation_log.append(
                {
                    "type": "critical_memory_tombstone_created",
                    "relationship_id": request.relationship_id,
                    "request_id": request_id,
                    "memory_id_hash": tombstone["memory_id_hash"],
                    "criticality": tombstone["criticality"],
                    "reason_categories": tombstone["reason_categories"],
                    "at": now.isoformat(),
                }
            )
        return True

    def cancel_memory_delete(self, request_id: str, *, now: datetime | None = None) -> None:
        now = now or utcnow()
        request = self.memory_delete_requests[request_id]
        request.status = ResetRequestStatus.CANCELLED
        memory = self.memories.get(request.memory_id)
        if memory:
            memory.metadata.pop("pending_delete", None)
        self.deviation_log.append(
            {
                "type": "memory_delete_cancelled",
                "request_id": request_id,
                "memory_id": request.memory_id,
                "relationship_id": request.relationship_id,
                "at": now.isoformat(),
            }
        )

    def inject_memory(
        self,
        relationship_id: str,
        content: str,
        *,
        memory_type: MemoryType = MemoryType.SHARED_EPISODE,
        context_tag: ContextTag = ContextTag.GENERAL,
        milestone: bool = False,
        core_identity: bool = False,
        force_memory_write: bool = False,
        now: datetime | None = None,
    ) -> str:
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        if not relationship.preferences.memory_writes_enabled and not force_memory_write:
            self.deviation_log.append(
                {
                    "type": "manual_memory_write_blocked",
                    "relationship_id": relationship_id,
                    "reason": relationship.preferences.memory_pause_reason or "memory_writes_disabled",
                    "text_chars": len(content),
                    "at": now.isoformat(),
                }
            )
            raise PermissionError("memory writes are paused for this relationship")
        if milestone:
            memory_type = MemoryType.MILESTONE
            context_tag = ContextTag.MILESTONE
        if core_identity:
            memory_type = MemoryType.IDENTITY
        memory = MemoryRecord(
            memory_id=new_id("mem"),
            relationship_id=relationship_id,
            content=content,
            memory_type=memory_type,
            context_tag=context_tag,
            created_at=now,
            updated_at=now,
            base_weight=0.95 if memory_type in {MemoryType.IDENTITY, MemoryType.MILESTONE} else 0.80,
            importance=0.95 if memory_type in {MemoryType.IDENTITY, MemoryType.MILESTONE} else 0.80,
            emotion_intensity=0.5 if memory_type != MemoryType.FACT else 0.0,
            decay_curve=DecayCurve.PERMANENT if memory_type == MemoryType.IDENTITY else DecayCurve.REVERSE_DECAY,
            relationship_stage_at_creation=relationship.stage,
            relationship_age_at_creation=relationship.relationship_age,
            trust_level_at_creation=relationship.trust_level,
            storage_layer=self._storage_layer_for(
                memory_type=memory_type,
                context_tag=context_tag,
                score=0.95 if memory_type in {MemoryType.IDENTITY, MemoryType.MILESTONE} else 0.80,
                relationship=relationship,
            ),
            metadata={"manual_injection": True, "memory_pause_override": force_memory_write},
        )
        memory.metadata["metacognition"] = self._memory_metacognition(
            memory,
            source_kind="manual_injection",
            human_verified=True,
            now=now,
        )
        self._apply_criticality_protection(relationship, memory, now=now)
        self._mark_critical_tombstone_remention(relationship, memory, now)
        self._attach_source_time(memory, content, now)
        memory.metadata["embeddings"] = self._memory_embedding_features(
            memory,
            detect_turn_signals(content, relationship.strength, relationship.trust_level, relationship.intimacy_level),
        )
        self.memories[memory.memory_id] = memory
        self._detect_time_conflicts(relationship, memory, now)
        self._detect_preference_supersession(relationship, memory, now)
        if milestone:
            relationship.milestones.append(memory.memory_id)
        if memory_type in {MemoryType.MILESTONE, MemoryType.SHARED_EPISODE, MemoryType.EMOTIONAL_MOMENT}:
            relationship.shared_episodes.append(memory.memory_id)
        if core_identity:
            self._create_core_identity(relationship, memory, title=content[:24], now=now)
        if milestone:
            self._set_milestone_confirmation(memory, "CONFIRMED", reason="manual_injection", now=now)
        self._update_shared_story(relationship, memory, detect_turn_signals(content, relationship.strength, relationship.trust_level, relationship.intimacy_level), now)
        self._maybe_set_origin_story(
            relationship,
            memory,
            detect_turn_signals(content, relationship.strength, relationship.trust_level, relationship.intimacy_level),
            now,
        )
        return memory.memory_id

    def edit_memory(self, memory_id: str, new_content: str, *, reason: str = "user_edit", now: datetime | None = None) -> None:
        now = now or utcnow()
        memory = self.memories[memory_id]
        history = memory.metadata.setdefault("versions", [])
        history.append({"content": memory.content, "updated_at": memory.updated_at.isoformat(), "reason": reason})
        memory.content = new_content
        memory.updated_at = now
        memory.metadata["last_edit_reason"] = reason
        self._mark_memory_verified(memory, reason=reason, now=now)
        for identity in self.core_identity.values():
            if identity.memory_id == memory_id:
                old_content = identity.content
                identity.change_log.append(
                    self._l4_change_entry(reason, now=now, old_content=old_content, new_content=new_content)
                )
                identity.content = new_content
                identity.updated_at = now
                self._refresh_l4_replicas(identity, now=now, reason=reason)
                self._review_l4_change(identity, memory, change_type=reason, now=now, previous_content=old_content)

    def retag_memory(
        self,
        memory_id: str,
        *,
        memory_type: MemoryType | None = None,
        context_tag: ContextTag | None = None,
        reason: str = "user_retag",
        now: datetime | None = None,
    ) -> None:
        now = now or utcnow()
        memory = self.memories[memory_id]
        relationship = self.relationships[memory.relationship_id]
        memory.metadata.setdefault("tag_versions", []).append(
            {
                "memory_type": memory.memory_type.value,
                "context_tag": memory.context_tag.value,
                "reason": reason,
                "at": now.isoformat(),
            }
        )
        if memory_type:
            memory.memory_type = memory_type
            memory.decay_curve = DecayCurve.REVERSE_DECAY if memory_type != MemoryType.FACT else DecayCurve.STANDARD_POWER_LAW
            memory.storage_layer = self._storage_layer_for(
                memory_type=memory.memory_type,
                context_tag=memory.context_tag,
                score=memory.importance,
                relationship=relationship,
            )
            memory.metadata["embeddings"] = self._memory_embedding_features(memory)
        if context_tag:
            memory.context_tag = context_tag
            memory.storage_layer = self._storage_layer_for(
                memory_type=memory.memory_type,
                context_tag=memory.context_tag,
                score=memory.importance,
                relationship=relationship,
            )
            memory.metadata["embeddings"] = self._memory_embedding_features(memory)
        self._refresh_relationship_indexes_for_memory(relationship, memory)
        self.deviation_log.append(
            {
                "type": "memory_retagged",
                "relationship_id": memory.relationship_id,
                "memory_id": memory_id,
                "memory_type": memory.memory_type.value,
                "context_tag": memory.context_tag.value,
                "reason": reason,
                "at": now.isoformat(),
            }
        )

    def set_inside_joke_status(
        self,
        memory_id: str,
        *,
        active: bool,
        reason: str = "user_control",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        memory = self.memories[memory_id]
        if memory.memory_type != MemoryType.INSIDE_JOKE and not memory.metadata.get("inside_joke_phrase"):
            raise ValueError("memory is not an inside joke")
        relationship = self.relationships[memory.relationship_id]
        if memory_id not in relationship.inside_jokes:
            relationship.inside_jokes.append(memory_id)
        before = bool(memory.metadata.get("inside_joke_inactive", False))
        memory.metadata["inside_joke_inactive"] = not active
        if active:
            memory.metadata["inside_joke_reactivated_at"] = now.isoformat()
            memory.metadata["inside_joke_negative_feedback"] = 0
            memory.metadata.pop("inside_joke_weight_multiplier", None)
        else:
            memory.metadata["inside_joke_deactivated_at"] = now.isoformat()
            memory.metadata["inside_joke_weight_multiplier"] = 0.3
        event = {
            "type": "inside_joke_reactivated" if active else "inside_joke_deactivated",
            "relationship_id": memory.relationship_id,
            "memory_id": memory_id,
            "phrase": memory.metadata.get("inside_joke_phrase"),
            "inactive_before": before,
            "inactive_after": bool(memory.metadata.get("inside_joke_inactive", False)),
            "reason": reason,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def resolve_unresolved_thread(
        self,
        memory_id: str,
        *,
        resolution: str = "completed",
        reason: str = "user_control",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        normalized = resolution.lower().replace("-", "_")
        if normalized not in {"completed", "no_longer_track", "muted"}:
            raise ValueError("resolution must be completed, no_longer_track, or muted")
        memory = self.memories[memory_id]
        relationship = self.relationships[memory.relationship_id]
        before = memory_id in relationship.unresolved_threads
        relationship.unresolved_threads = [mid for mid in relationship.unresolved_threads if mid != memory_id]
        state = memory.metadata.setdefault("unresolved_thread_resolution", {})
        state.update(
            {
                "status": normalized.upper(),
                "reason": reason,
                "resolved_at": now.isoformat(),
            }
        )
        if normalized in {"no_longer_track", "muted"}:
            memory.metadata["thread_recall_suppressed"] = True
        event = {
            "type": "unresolved_thread_resolved",
            "relationship_id": memory.relationship_id,
            "memory_id": memory_id,
            "resolution": normalized,
            "was_unresolved": before,
            "reason": reason,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def _apply_assistant_reply_memory_effects(
        self,
        relationship_id: str,
        exchange_memory_id: str,
        user_text: str,
        assistant_text: str,
        now: datetime,
    ) -> None:
        exchange_memory = self.memories[exchange_memory_id]
        assistant_outcome = self._assistant_reply_outcome(assistant_text)
        exchange_memory.metadata["assistant_reply_outcome"] = assistant_outcome
        if assistant_outcome not in {"declined", "cancelled"}:
            return
        if not self._looks_like_invitation_or_commitment(user_text):
            return

        resolved_memory_ids = {exchange_memory_id}
        relationship = self.relationships[relationship_id]
        if exchange_memory.memory_type == MemoryType.MILESTONE and exchange_memory.metadata.get("major_shared_decision"):
            exchange_memory.memory_type = MemoryType.CONTEXT_DETAIL
            exchange_memory.context_tag = ContextTag.GENERAL
            exchange_memory.decay_curve = self._decay_curve_for(exchange_memory.memory_type, relationship)
            exchange_memory.storage_layer = self._storage_layer_for(
                memory_type=exchange_memory.memory_type,
                context_tag=exchange_memory.context_tag,
                score=exchange_memory.importance,
                relationship=relationship,
            )
            relationship.milestones = [mid for mid in relationship.milestones if mid != exchange_memory_id]
            relationship.shared_episodes = [mid for mid in relationship.shared_episodes if mid != exchange_memory_id]
            for story in self.story_nodes.values():
                if story.relationship_id == relationship_id and (
                    exchange_memory_id in story.core_events or exchange_memory_id in story.key_moments
                ):
                    story.title = f"已拒绝邀约：{user_text[:18]}"
                    story.narrative_level = NarrativeLevel.FRAGMENT
        if exchange_memory_id in relationship.unresolved_threads:
            self.resolve_unresolved_thread(
                exchange_memory_id,
                resolution="no_longer_track",
                reason=f"assistant_{assistant_outcome}_in_exchange",
                now=now,
            )
        for candidate_id in self._related_pending_commitment_memory_ids(relationship_id, user_text, exclude={exchange_memory_id}):
            resolved_memory_ids.add(candidate_id)
            if candidate_id in relationship.unresolved_threads:
                self.resolve_unresolved_thread(
                    candidate_id,
                    resolution="no_longer_track",
                    reason=f"assistant_{assistant_outcome}_related_exchange",
                    now=now,
                )

        archived_reminders = []
        for reminder in self.commitment_reminders.values():
            if reminder.relationship_id != relationship_id:
                continue
            if reminder.status not in {ReminderStatus.PENDING, ReminderStatus.REMINDER_SENT}:
                continue
            if reminder.memory_id not in resolved_memory_ids:
                continue
            reminder.status = ReminderStatus.ARCHIVED
            reminder.archived_at = now
            archived_reminders.append(reminder.reminder_id)

        exchange_memory.metadata["assistant_resolution"] = {
            "outcome": assistant_outcome,
            "related_memory_ids": sorted(resolved_memory_ids),
            "archived_reminder_ids": archived_reminders,
            "at": now.isoformat(),
        }
        self.deviation_log.append(
            {
                "type": "assistant_reply_resolved_commitment",
                "relationship_id": relationship_id,
                "exchange_memory_id": exchange_memory_id,
                "outcome": assistant_outcome,
                "related_memory_ids": sorted(resolved_memory_ids),
                "archived_reminder_ids": archived_reminders,
                "at": now.isoformat(),
            }
        )

    def _assistant_reply_outcome(self, assistant_text: str) -> str:
        text = assistant_text.lower()
        cancel_words = ["取消", "不算了", "作废", "别去了", "不用去了"]
        decline_words = [
            "不去",
            "不去了",
            "不能",
            "不行",
            "不太行",
            "没时间",
            "没有时间",
            "不方便",
            "不想",
            "改天",
            "下次吧",
            "算了",
            "拒绝",
        ]
        accept_words = ["可以", "好啊", "行啊", "没问题", "说定", "约好了", "一起去"]
        if any(word in text for word in cancel_words):
            return "cancelled"
        if any(word in text for word in decline_words):
            return "declined"
        if any(word in text for word in accept_words):
            return "accepted"
        return "neutral"

    def _looks_like_invitation_or_commitment(self, text: str) -> bool:
        return any(word in text for word in ["一起", "约", "约定", "答应", "明天", "后天", "下次", "吃饭", "网吧", "去吗", "好吗"])

    def _related_pending_commitment_memory_ids(
        self,
        relationship_id: str,
        user_text: str,
        *,
        exclude: set[str] | None = None,
    ) -> list[str]:
        exclude = exclude or set()
        related: list[str] = []
        user_tokens = set(tokenize(user_text))
        for reminder in self.commitment_reminders.values():
            if reminder.relationship_id != relationship_id:
                continue
            if reminder.status not in {ReminderStatus.PENDING, ReminderStatus.REMINDER_SENT}:
                continue
            memory = self.memories.get(reminder.memory_id)
            if not memory or memory.memory_id in exclude:
                continue
            memory_tokens = set(tokenize(memory.content))
            token_overlap = len(user_tokens & memory_tokens) / max(1, len(user_tokens | memory_tokens))
            lexical = lexical_similarity(user_text, memory.content)
            if token_overlap >= 0.20 or lexical >= 0.20 or any(word in memory.content and word in user_text for word in ["明天", "后天", "吃饭", "网吧"]):
                related.append(memory.memory_id)
        return related

    def mark_milestone(self, memory_id: str, *, now: datetime | None = None) -> None:
        now = now or utcnow()
        memory = self.memories[memory_id]
        relationship = self.relationships[memory.relationship_id]
        memory.memory_type = MemoryType.MILESTONE
        memory.context_tag = ContextTag.MILESTONE
        memory.decay_curve = DecayCurve.PERMANENT
        memory.base_weight = max(memory.base_weight, 0.95)
        memory.importance = max(memory.importance, 0.95)
        memory.storage_layer = MemoryLayer.L5_RELATIONSHIP_HISTORY
        memory.metadata["embeddings"] = self._memory_embedding_features(memory)
        self._set_milestone_confirmation(memory, "CONFIRMED", reason="manual_mark")
        if memory_id not in relationship.milestones:
            relationship.milestones.append(memory_id)
        if memory_id not in relationship.shared_episodes:
            relationship.shared_episodes.append(memory_id)
        self._ensure_milestone_story(relationship, memory, now=now, reason="manual_mark")

    def confirm_milestone(
        self,
        memory_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        now: datetime | None = None,
    ) -> None:
        now = now or utcnow()
        memory = self.memories[memory_id]
        if memory.memory_type != MemoryType.MILESTONE:
            self.mark_milestone(memory_id, now=now)
        relationship = self.relationships[memory.relationship_id]
        if memory_id not in relationship.milestones:
            relationship.milestones.append(memory_id)
        confirmation = self._set_milestone_confirmation(memory, "CONFIRMED", reason="user_confirmed", now=now)
        if title:
            confirmation["title"] = title
        if description:
            confirmation["description"] = description
        memory.metadata["milestone_confirmation"] = confirmation
        self.deviation_log.append(
            {
                "type": "milestone_confirmed",
                "relationship_id": memory.relationship_id,
                "memory_id": memory_id,
                "at": now.isoformat(),
            }
        )

    def edit_milestone(
        self,
        memory_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        memory = self.memories[memory_id]
        relationship = self.relationships[memory.relationship_id]
        if memory_id not in relationship.milestones:
            raise ValueError("memory is not a relationship milestone")
        confirmation = dict(memory.metadata.get("milestone_confirmation", {}))
        if not confirmation:
            confirmation = self._set_milestone_confirmation(memory, "CONFIRMED", reason="milestone_edit_initialized", now=now)
        previous = {
            "title": confirmation.get("title"),
            "description": confirmation.get("description"),
            "status": confirmation.get("status"),
        }
        if title is not None:
            confirmation["title"] = title
        if description is not None:
            confirmation["description"] = description
        confirmation["edited_at"] = now.isoformat()
        confirmation["edit_count"] = int(confirmation.get("edit_count", 0) or 0) + 1
        confirmation.setdefault("edit_versions", []).append(
            {
                "at": now.isoformat(),
                "previous": previous,
                "title": confirmation.get("title"),
                "description": confirmation.get("description"),
            }
        )
        confirmation["edit_versions"] = confirmation["edit_versions"][-20:]
        memory.metadata["milestone_confirmation"] = confirmation
        for story in self.story_nodes.values():
            if story.relationship_id == relationship.relationship_id and memory_id in story.core_events and title is not None:
                story.title = f"里程碑：{title}"
        event = {
            "type": "milestone_edited",
            "relationship_id": memory.relationship_id,
            "memory_id": memory_id,
            "title": confirmation.get("title"),
            "description": confirmation.get("description"),
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def reject_milestone(self, memory_id: str, *, reason: str = "user_rejected", now: datetime | None = None) -> None:
        now = now or utcnow()
        memory = self.memories[memory_id]
        confirmation = self._set_milestone_confirmation(memory, "REJECTED", reason=reason, now=now)
        memory.metadata["milestone_confirmation"] = confirmation
        self.downgrade_memory(memory_id, reason=reason, now=now)
        memory.metadata["milestone_confirmation"] = confirmation
        self.deviation_log.append(
            {
                "type": "milestone_rejected",
                "relationship_id": memory.relationship_id,
                "memory_id": memory_id,
                "reason": reason,
                "at": now.isoformat(),
            }
        )

    def downgrade_memory(
        self,
        memory_id: str,
        *,
        reason: str = "user_downgrade",
        now: datetime | None = None,
    ) -> None:
        now = now or utcnow()
        memory = self.memories[memory_id]
        relationship = self.relationships[memory.relationship_id]
        memory.metadata.setdefault("downgrade_history", []).append(
            {
                "at": now.isoformat(),
                "reason": reason,
                "previous_type": memory.memory_type.value,
                "previous_context_tag": memory.context_tag.value,
                "previous_decay_curve": memory.decay_curve.value,
                "previous_storage_layer": memory.storage_layer.value,
                "previous_importance": memory.importance,
            }
        )
        memory.memory_type = MemoryType.FACT
        memory.context_tag = ContextTag.GENERAL
        memory.decay_curve = DecayCurve.STANDARD_POWER_LAW
        memory.importance = min(memory.importance, 0.4)
        memory.storage_layer = self._storage_layer_for(
            memory_type=memory.memory_type,
            context_tag=memory.context_tag,
            score=memory.importance,
            relationship=relationship,
        )
        memory.metadata["embeddings"] = self._memory_embedding_features(memory)
        for bucket in (relationship.milestones, relationship.shared_episodes, relationship.inside_jokes, relationship.unresolved_threads):
            while memory_id in bucket:
                bucket.remove(memory_id)
        self._downgrade_story_membership(memory, relationship, now=now, reason=reason)

    def batch_downgrade_memories(
        self,
        relationship_id: str | None = None,
        *,
        memory_ids: list[str] | None = None,
        memory_type: MemoryType | None = None,
        context_tag: ContextTag | None = None,
        storage_layer: MemoryLayer | None = None,
        reason: str = "user_batch_downgrade",
        now: datetime | None = None,
    ) -> list[str]:
        now = now or utcnow()
        if not memory_ids and not relationship_id:
            raise ValueError("relationship_id is required when memory_ids are not provided")
        selected = [
            memory
            for memory in self.memories.values()
            if (relationship_id is None or memory.relationship_id == relationship_id)
            and (memory_ids is None or memory.memory_id in memory_ids)
            and (memory_type is None or memory.memory_type == memory_type)
            and (context_tag is None or memory.context_tag == context_tag)
            and (storage_layer is None or memory.storage_layer == storage_layer)
        ]
        downgraded: list[str] = []
        for memory in selected:
            if memory.memory_type == MemoryType.IDENTITY:
                continue
            self.downgrade_memory(memory.memory_id, reason=reason, now=now)
            downgraded.append(memory.memory_id)
        self.deviation_log.append(
            {
                "type": "batch_downgrade",
                "relationship_id": relationship_id,
                "memory_ids": downgraded,
                "filters": {
                    "memory_type": memory_type.value if memory_type else None,
                    "context_tag": context_tag.value if context_tag else None,
                    "storage_layer": storage_layer.value if storage_layer else None,
                    "explicit_memory_ids": memory_ids,
                },
                "reason": reason,
                "at": now.isoformat(),
            }
        )
        return downgraded

    def due_commitment_reminders(
        self,
        relationship_id: str,
        *,
        now: datetime | None = None,
        window_days: int = 1,
        include_future: bool = False,
    ) -> list[CommitmentReminder]:
        now = now or utcnow()
        self._archive_expired_commitment_reminders(now)
        reminders = [
            reminder
            for reminder in self.commitment_reminders.values()
            if reminder.relationship_id == relationship_id
            and reminder.status in {ReminderStatus.PENDING, ReminderStatus.REMINDER_SENT}
            and (
                include_future
                or reminder.due_at <= now + timedelta(days=self._commitment_reminder_window_days(reminder, window_days))
            )
            and not self._memory_is_recall_suppressed(self.memories.get(reminder.memory_id))
        ]
        return sorted(reminders, key=lambda item: (self._commitment_priority_rank(item.priority), item.due_at))

    def check_commitment_reminders(
        self,
        relationship_id: str,
        *,
        now: datetime | None = None,
        window_days: int = 1,
    ) -> list[dict[str, Any]]:
        now = now or utcnow()
        surfaced: list[dict[str, Any]] = []
        for reminder in self.due_commitment_reminders(relationship_id, now=now, window_days=window_days):
            if reminder.last_reminded_at and reminder.last_reminded_at.date() == now.date():
                continue
            reminder.status = ReminderStatus.REMINDER_SENT
            reminder.reminder_count += 1
            reminder.last_reminded_at = now
            surfaced.append(self._commitment_reminder_payload(reminder, now))
        if surfaced:
            self.deviation_log.append(
                {
                    "type": "commitment_reminders_checked",
                    "relationship_id": relationship_id,
                    "reminder_ids": [item["reminder_id"] for item in surfaced],
                    "at": now.isoformat(),
                }
            )
        return surfaced

    def complete_commitment_reminder(self, reminder_id: str, *, now: datetime | None = None) -> None:
        now = now or utcnow()
        reminder = self.commitment_reminders[reminder_id]
        reminder.status = ReminderStatus.COMPLETED
        reminder.completed_at = now
        self.deviation_log.append(
            {
                "type": "commitment_reminder_completed",
                "relationship_id": reminder.relationship_id,
                "reminder_id": reminder_id,
                "memory_id": reminder.memory_id,
                "at": now.isoformat(),
            }
        )

    def active_suggestions(
        self,
        relationship_id: str,
        *,
        current_text: str = "",
        now: datetime | None = None,
        current_signals: TurnSignals | None = None,
    ) -> list[str]:
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        current_signals = current_signals or (
            detect_turn_signals(current_text, relationship.strength, relationship.trust_level, relationship.intimacy_level)
            if current_text
            else None
        )
        if relationship.preferences.mode == Mode.ASSISTANT:
            return []
        if not relationship.preferences.active_recall_enabled:
            return []
        if relationship.stage == RelationshipStage.INITIATING:
            return []
        if relationship.stage in {RelationshipStage.STAGNATING, RelationshipStage.AVOIDING, RelationshipStage.TERMINATING}:
            return []
        if relationship.stage in {RelationshipStage.INITIATING, RelationshipStage.EXPERIMENTING}:
            max_items = 1
        else:
            max_items = relationship.preferences.max_active_per_session

        candidates: list[dict[str, Any]] = []
        deviation = self._detect_deviation(relationship, current_text, current_signals, now=now)
        if deviation:
            candidates.append({"type": "baseline_care", "text": deviation, "priority": 0, "bypass_high_emotion": True})

        for topic in relationship.implicit_topics:
            if topic.get("status") != "ACTIVE":
                continue
            if float(topic.get("confidence", 0.0)) < 0.70:
                continue
            evidence = self._implicit_topic_evidence_status(relationship, topic, now)
            if not evidence["valid"]:
                self._fail_implicit_topic_evidence(relationship, topic, evidence, now)
                continue
            if int(topic.get("prompt_count", 0)) >= 3:
                continue
            candidates.append(
                {
                    "type": "implicit_topic",
                    "text": f"隐含话题：根据之前几段碎片，可能可以温和确认「{topic.get('summary', '')[:42]}」",
                    "topic_id": topic.get("topic_id"),
                    "evidence": evidence,
                    "inferred": True,
                    "uncertainty_action": "confirm_gently",
                    "confidence": topic.get("confidence"),
                    "priority": 18,
                }
            )
            topic["prompt_count"] = int(topic.get("prompt_count", 0)) + 1
            topic["last_prompted_at"] = now.isoformat()
            break

        for memory_id in relationship.unresolved_threads:
            memory = self.memories.get(memory_id)
            if not memory or self._memory_is_recall_suppressed(memory):
                continue
            topic_similarity = lexical_similarity(current_text, memory.content) if current_text else 1.0
            relationship_signal = relationship.strength * max(memory.emotion_intensity, memory.importance)
            if (now - memory.updated_at).days >= 30 and relationship_signal >= 0.5 and topic_similarity >= 0.25:
                candidates.append(
                    {
                        "type": "shared_topic_reactivation",
                        "text": f"未完结话题：可以自然问起「{memory.content[:36]}」",
                        "memory_id": memory.memory_id,
                        "priority": 20,
                    }
                )
                break

        for reminder in self.due_commitment_reminders(relationship_id, now=now, window_days=1):
            memory = self.memories.get(reminder.memory_id)
            if not memory or self._memory_is_recall_suppressed(memory):
                continue
            candidates.append(
                {
                    "type": "commitment_reminder",
                    "text": f"承诺提醒：{reminder.title}（{self._commitment_due_phrase(reminder.due_at, now)}）",
                    "memory_id": memory.memory_id,
                    "reminder_id": reminder.reminder_id,
                    "priority": 8 if reminder.priority == "CRITICAL" else 12 if reminder.priority == "HIGH" else 15,
                }
            )

        for memory_id in relationship.milestones:
            memory = self.memories.get(memory_id)
            if (
                memory
                and not self._memory_is_recall_suppressed(memory)
                and memory.created_at.month == now.month
                and memory.created_at.day == now.day
                and not self._active_type_logged_today(relationship, "anniversary", now)
            ):
                candidates.append(
                    {
                        "type": "anniversary",
                        "text": f"关系纪念日：今天可以提起「{memory.content[:36]}」",
                        "memory_id": memory.memory_id,
                        "priority": 30,
                    }
                )
                break

        if not any(candidate["type"] == "anniversary" for candidate in candidates):
            relationship_age_anniversary = self._relationship_age_anniversary_candidate(relationship, now)
            if relationship_age_anniversary:
                candidates.append(relationship_age_anniversary)

        inside_joke = self._inside_joke_suggestion(relationship, current_text, now)
        if inside_joke:
            candidates.append(inside_joke)

        if relationship.preferences.emotional_layer_enabled and current_signals and current_signals.emotion_intensity >= 0.55:
            emotional_match = self._find_emotional_resonance(relationship_id, current_signals, now)
            if emotional_match:
                evidence = self._emotional_resonance_evidence(emotional_match, current_signals, now)
                candidates.append(
                    {
                        "type": "emotional_resonance",
                        "text": f"情感共鸣：现在的状态让我想起一个可能相关的旧时刻「{emotional_match.content[:36]}」，可以温和确认是否有关。",
                        "memory_id": emotional_match.source_memory_id,
                        "evidence": evidence,
                        "inferred": True,
                        "uncertainty_action": "confirm_gently",
                        "confidence": evidence["confidence"],
                        "priority": 50,
                    }
                )

        if relationship.stage == RelationshipStage.EXPERIMENTING:
            candidates = [
                candidate
                for candidate in candidates
                if candidate["type"] in {"commitment_reminder", "anniversary"}
            ]
        filtered = self._filter_active_candidates(relationship, candidates, current_text, current_signals, now)
        return [candidate["text"] for candidate in filtered[:max_items]]

    def reset_relationship(self, relationship_id: str, mode: ResetMode) -> None:
        relationship = self.relationships[relationship_id]
        if mode == ResetMode.SOFT:
            relationship.strength = 0.0
            self._set_stage(relationship, RelationshipStage.INITIATING, utcnow(), "soft_reset")
            relationship.mode_history.append({"mode": mode.value, "at": utcnow().isoformat()})
            return
        if mode == ResetMode.MEDIUM:
            for memory in list(self.memories.values()):
                if memory.relationship_id == relationship_id and memory.decay_curve == DecayCurve.REVERSE_DECAY:
                    del self.memories[memory.memory_id]
            for emotion in list(self.emotional_memories.values()):
                if emotion.relationship_id == relationship_id:
                    del self.emotional_memories[emotion.emotion_id]
            for story in list(self.story_nodes.values()):
                if story.relationship_id == relationship_id:
                    del self.story_nodes[story.story_id]
            for edge in list(self.memory_graph_edges.values()):
                if edge.relationship_id == relationship_id and (
                    edge.source_memory_id not in self.memories or edge.target_memory_id not in self.memories
                ):
                    del self.memory_graph_edges[edge.edge_id]
            for reminder in list(self.commitment_reminders.values()):
                if reminder.relationship_id == relationship_id:
                    del self.commitment_reminders[reminder.reminder_id]
            self.emotional_trajectories.pop(relationship_id, None)
            relationship.shared_episodes.clear()
            relationship.inside_jokes.clear()
            relationship.milestones.clear()
            relationship.unresolved_threads.clear()
            self._set_stage(relationship, RelationshipStage.INITIATING, utcnow(), "medium_reset")
            relationship.strength = 0.0
            return
        if mode == ResetMode.HARD:
            for memory in list(self.memories.values()):
                if memory.relationship_id == relationship_id:
                    del self.memories[memory.memory_id]
            for emotion in list(self.emotional_memories.values()):
                if emotion.relationship_id == relationship_id:
                    del self.emotional_memories[emotion.emotion_id]
            for story in list(self.story_nodes.values()):
                if story.relationship_id == relationship_id:
                    del self.story_nodes[story.story_id]
            for edge in list(self.memory_graph_edges.values()):
                if edge.relationship_id == relationship_id:
                    del self.memory_graph_edges[edge.edge_id]
            self.emotional_trajectories.pop(relationship_id, None)
            for identity in list(self.core_identity.values()):
                if identity.relationship_id == relationship_id:
                    del self.core_identity[identity.identity_id]
            for request in list(self.core_identity_delete_requests.values()):
                if request.relationship_id == relationship_id:
                    del self.core_identity_delete_requests[request.request_id]
            for reminder in list(self.commitment_reminders.values()):
                if reminder.relationship_id == relationship_id:
                    del self.commitment_reminders[reminder.reminder_id]
            del self.relationships[relationship_id]

    def request_reset(self, relationship_id: str, mode: ResetMode, *, now: datetime | None = None) -> ResetRequest:
        now = now or utcnow()
        request = ResetRequest(
            request_id=new_id("reset"),
            relationship_id=relationship_id,
            mode=mode,
            requested_at=now,
            execute_after=now + timedelta(hours=24),
        )
        self.reset_requests[request.request_id] = request
        self.deviation_log.append(
            {
                "type": "reset_requested",
                "request_id": request.request_id,
                "relationship_id": relationship_id,
                "mode": mode.value,
                "requested_at": now.isoformat(),
                "execute_after": request.execute_after.isoformat(),
            }
        )
        return request

    def confirm_reset(self, request_id: str, *, now: datetime | None = None, force: bool = False) -> bool:
        now = now or utcnow()
        request = self.reset_requests[request_id]
        if request.status != ResetRequestStatus.PENDING:
            return False
        if not force and now < request.execute_after:
            return False
        before_counts = self._relationship_deletion_counts(request.relationship_id)
        if request.mode == ResetMode.HARD:
            self._record_relationship_ending_support(request.relationship_id, request.request_id, now, before_counts=before_counts)
        self.reset_relationship(request.relationship_id, request.mode)
        after_counts = self._relationship_deletion_counts(request.relationship_id)
        request.status = ResetRequestStatus.EXECUTED
        request.executed_at = now
        self.deviation_log.append(
            {
                "type": "reset_confirmed",
                "request_id": request.request_id,
                "relationship_id": request.relationship_id,
                "mode": request.mode.value,
                "executed_at": now.isoformat(),
            }
        )
        self._record_deletion_compliance(
            relationship_id=request.relationship_id,
            deletion_type=f"{request.mode.value}_RESET",
            request_id=request.request_id,
            reason=f"reset:{request.mode.value}",
            now=now,
            summary={"before": before_counts, "after": after_counts},
        )
        return True

    def cancel_reset(self, request_id: str) -> None:
        request = self.reset_requests[request_id]
        request.status = ResetRequestStatus.CANCELLED
        self.deviation_log.append({"type": "reset_cancelled", "request_id": request_id, "at": utcnow().isoformat()})

    def relationship_ending_support(self, request_id: str | None = None) -> list[dict[str, Any]]:
        if request_id is None:
            return list(self.relationship_ending_support_log)
        return [item for item in self.relationship_ending_support_log if item.get("request_id") == request_id]

    def set_mode(
        self,
        relationship_id: str,
        mode: Mode,
        *,
        custom_profile: dict[str, Any] | None = None,
        reason: str = "user_mode_switch",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        previous_mode = relationship.preferences.mode
        relationship.preferences.mode = mode
        if mode == Mode.ASSISTANT:
            relationship.preferences.active_recall_enabled = False
            relationship.preferences.reverse_decay_enabled = False
            self._sync_reverse_decay_curve_preference(relationship)
        elif mode == Mode.FRIEND:
            relationship.preferences.active_recall_enabled = True
            relationship.preferences.reverse_decay_enabled = True
            relationship.preferences.emotional_layer_enabled = True
            self._sync_reverse_decay_curve_preference(relationship)
        elif mode == Mode.CUSTOM:
            if custom_profile is not None:
                relationship.preferences.custom_profile = self._normalize_custom_profile(custom_profile)
            self._apply_custom_profile(relationship)
        relationship.mode_history.append(
            {
                "mode": mode.value,
                "at": now.isoformat(),
                "custom_profile": relationship.preferences.custom_profile if mode == Mode.CUSTOM else None,
                "reason": reason,
            }
        )
        event = {
            "type": "mode_changed",
            "relationship_id": relationship_id,
            "from": previous_mode.value,
            "to": mode.value,
            "reason": reason,
            "custom_profile": relationship.preferences.custom_profile if mode == Mode.CUSTOM else None,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def update_custom_mode_profile(
        self,
        relationship_id: str,
        profile: dict[str, Any],
        *,
        reason: str = "user_custom_profile",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        merged = dict(relationship.preferences.custom_profile)
        merged.update(profile)
        relationship.preferences.custom_profile = self._normalize_custom_profile(merged)
        applied = False
        if relationship.preferences.mode == Mode.CUSTOM:
            self._apply_custom_profile(relationship)
            applied = True
        event = {
            "type": "custom_mode_profile_updated",
            "relationship_id": relationship_id,
            "profile": relationship.preferences.custom_profile,
            "applied": applied,
            "reason": reason,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def mode_suggestions(self, relationship_id: str, *, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        suggestions: list[dict[str, Any]] = []
        has_used_friend_mode = any(item.get("mode") == Mode.FRIEND.value for item in relationship.mode_history)
        if relationship.preferences.mode == Mode.ASSISTANT and relationship.strength > 0.10 and not has_used_friend_mode:
            relational_memories = [
                memory
                for memory in self.memories.values()
                if memory.relationship_id == relationship_id
                and memory.memory_type
                in {
                    MemoryType.SHARED_EPISODE,
                    MemoryType.EMOTIONAL_MOMENT,
                    MemoryType.COMMITMENT,
                    MemoryType.MILESTONE,
                    MemoryType.INSIDE_JOKE,
                }
                and not memory.metadata.get("archived")
            ]
            if relational_memories:
                latest = max(relational_memories, key=lambda memory: memory.updated_at)
                suggestions.append(
                    {
                        "type": "switch_to_friend",
                        "recommended_mode": Mode.FRIEND.value,
                        "strength": relationship.strength,
                        "reason": "relationship_depth_threshold",
                        "memory_id": latest.memory_id,
                        "message": "这段互动已经出现稳定的关系信号，可以询问用户是否愿意进入朋友模式；切换只改变规则，不删除数据。",
                    }
                )
        if relationship.preferences.mode == Mode.FRIEND and relationship.last_interaction:
            inactive_days = (now.date() - relationship.last_interaction.date()).days
            if inactive_days > 180:
                suggestions.append(
                    {
                        "type": "switch_to_assistant",
                        "recommended_mode": Mode.ASSISTANT.value,
                        "inactive_days": inactive_days,
                        "reason": "friend_mode_long_inactivity",
                        "message": "这段关系已经很久没有互动了，可以考虑切回助理模式以减少主动打扰；关系记忆不会被删除。",
                    }
                )
        return suggestions

    def rollback_stage(
        self,
        relationship_id: str,
        *,
        history_index: int | None = None,
        reason: str = "user_stage_rollback",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        if not relationship.stage_history:
            raise ValueError("stage history is empty")
        transition = relationship.stage_history[history_index] if history_index is not None else relationship.stage_history[-1]
        target_stage = RelationshipStage(transition["from"])
        from_stage = relationship.stage
        if target_stage == from_stage:
            relationship.retention_multiplier = self._retention_multiplier_for_stage(target_stage)
        else:
            self._set_stage(relationship, target_stage, now, reason)
        rollback = {
            "type": "stage_rollback",
            "relationship_id": relationship_id,
            "from": from_stage.value,
            "to": target_stage.value,
            "based_on": transition,
            "reason": reason,
            "at": now.isoformat(),
            "retention_multiplier": relationship.retention_multiplier,
        }
        self.deviation_log.append(rollback)
        if relationship.stage_history:
            relationship.stage_history[-1]["rollback"] = True
            relationship.stage_history[-1]["rollback_reason"] = reason
        return rollback

    def audit_report(self, relationship_id: str | None = None, *, now: datetime | None = None) -> dict[str, Any]:
        now = now or utcnow()
        relationship_ids = [relationship_id] if relationship_id else list(self.relationships)
        relationships = [self.relationships[item] for item in relationship_ids if item in self.relationships]
        missing_relationship_ids = [item for item in relationship_ids if item not in self.relationships]
        memories = [item for item in self.memories.values() if item.relationship_id in relationship_ids]
        active_behavior_events = [
            event
            for relationship in relationships
            for event in relationship.active_behavior_log
        ]
        active_events = [
            event
            for event in active_behavior_events
            if event.get("reaction") in {"accepted", "neutral", "ignored", "denied"}
        ]
        accepted_or_neutral = [event for event in active_events if event.get("reaction") in {"accepted", "neutral"}]
        active_complaint_events = [
            item
            for item in self.deviation_log
            if item.get("relationship_id") in relationship_ids
            and (
                item.get("type") in {"interruption_request_recorded", "memory_boundary_request"}
                or (
                    item.get("type") == "active_suppressed"
                    and item.get("reason") == "user_requested_no_interruption"
                )
            )
        ]
        explicit_active_interruption_suppressions = [
            item
            for item in active_complaint_events
            if item.get("type") == "active_suppressed"
            and item.get("reason") == "user_requested_no_interruption"
        ]
        observed_user_sessions = sum(relationship.interaction_count for relationship in relationships) + len(
            explicit_active_interruption_suppressions
        )
        pending_delete_requests = [
            item
            for item in self.memory_delete_requests.values()
            if item.relationship_id in relationship_ids and item.status == ResetRequestStatus.PENDING
        ]
        executed_delete_requests = [
            item
            for item in self.memory_delete_requests.values()
            if item.relationship_id in relationship_ids and item.status == ResetRequestStatus.EXECUTED
        ]
        pending_reset_requests = [
            item
            for item in self.reset_requests.values()
            if item.relationship_id in relationship_ids and item.status == ResetRequestStatus.PENDING
        ]
        executed_reset_requests = [
            item
            for item in self.reset_requests.values()
            if item.relationship_id in relationship_ids and item.status == ResetRequestStatus.EXECUTED
        ]
        pending_l4_delete_requests = [
            item
            for item in self.core_identity_delete_requests.values()
            if item.relationship_id in relationship_ids and item.status == ResetRequestStatus.PENDING
        ]
        executed_l4_delete_requests = [
            item
            for item in self.core_identity_delete_requests.values()
            if item.relationship_id in relationship_ids and item.status == ResetRequestStatus.EXECUTED
        ]
        integrity = self._audit_integrity(relationship_ids)
        for missing_id in missing_relationship_ids:
            integrity["warnings"].append(
                {
                    "type": "relationship_not_found",
                    "relationship_id": missing_id,
                    "severity": "warning",
                }
            )
        trust_bias_audit = self._trust_bias_audit_summary(relationship_ids, relationships, memories, now)
        metrics = {
            "relationships": len(relationships),
            "memories": len(memories),
            "emotional_memories": len(
                [item for item in self.emotional_memories.values() if item.relationship_id in relationship_ids]
            ),
            "story_nodes": len([item for item in self.story_nodes.values() if item.relationship_id in relationship_ids]),
            "memory_graph_edges": len([item for item in self.memory_graph_edges.values() if item.relationship_id in relationship_ids]),
            "commitment_reminders": len(
                [item for item in self.commitment_reminders.values() if item.relationship_id in relationship_ids]
            ),
            "implicit_topics": len(
                [
                    topic
                    for relationship in relationships
                    for topic in relationship.implicit_topics
                    if topic.get("status") == "ACTIVE"
                ]
            ),
            "ai_decision_events": len([item for item in self.ai_decision_log if item.get("relationship_id") in relationship_ids]),
            "retrieval_audit_events": len(
                [item for item in self.retrieval_audit_log if item.get("relationship_id") in relationship_ids]
            ),
            "active_behavior_events": len(active_behavior_events),
            "active_feedback_events": len(active_events),
            "active_acceptance_rate": (len(accepted_or_neutral) / len(active_events)) if active_events else None,
            "active_complaint_events": len(active_complaint_events),
            "active_complaint_rate": (len(active_complaint_events) / observed_user_sessions)
            if observed_user_sessions
            else None,
            "health_alerts_open": len(
                [
                    item
                    for item in self.health_alerts.values()
                    if item.relationship_id in relationship_ids and not item.acknowledged
                ]
            ),
            "transparency_acknowledged": len(
                [
                    relationship
                    for relationship in relationships
                    if relationship.transparency_acknowledged_at is not None
                ]
            ),
            "transparency_pending": len(
                [
                    relationship
                    for relationship in relationships
                    if relationship.preferences.relationship_nature_disclosure_enabled
                    and relationship.transparency_acknowledged_at is None
                ]
            ),
            "unresolved_time_conflicts": len(integrity["unresolved_time_conflicts"]),
            "pending_memory_deletes": len(pending_delete_requests),
            "executed_memory_deletes": len(executed_delete_requests),
            "pending_resets": len(pending_reset_requests),
            "executed_resets": len(executed_reset_requests),
            "pending_l4_deletes": len(pending_l4_delete_requests),
            "executed_l4_deletes": len(executed_l4_delete_requests),
            "trust_bias_monthly_audit_ready": trust_bias_audit["monthly_audit_ready"],
            "trust_bias_adjusted_samples": trust_bias_audit["adjusted_retrieval_samples"],
            "trust_bias_critical_exemption_samples": trust_bias_audit["critical_exemption_samples"],
            "trust_bias_soft_cooldown_samples": trust_bias_audit["soft_cooldown_samples"],
        }
        gates = {
            "relationships_found": not missing_relationship_ids,
            "delete_propagation_clean": not integrity["orphan_references"],
            "ai_participation_visible": metrics["ai_decision_events"] > 0 if memories else True,
            "retrieval_audited": metrics["retrieval_audit_events"] > 0 if memories else True,
            "active_feedback_measured": metrics["active_feedback_events"] > 0
            if metrics["active_behavior_events"] > 0
            else True,
            "active_complaint_rate_acceptable": metrics["active_complaint_rate"] is None
            or metrics["active_complaint_rate"] <= 0.05,
            "trust_bias_monthly_audit_ready": trust_bias_audit["monthly_audit_ready"],
            "transparency_acknowledged": metrics["transparency_pending"] == 0,
            "no_unresolved_time_conflicts": metrics["unresolved_time_conflicts"] == 0,
            "health_review_current": all(
                any(
                    item.get("type") in {"health_alert", "health_review_completed", "guardian_summary_generated"}
                    and item.get("relationship_id") == relationship.relationship_id
                    for item in self.deviation_log[-200:]
                )
                for relationship in relationships
            )
            if relationships
            else True,
        }
        spec_coverage = self._audit_spec_coverage(relationship_ids, relationships, memories, metrics)
        coverage_summary = self._audit_coverage_summary(spec_coverage)
        status = "PASS" if all(gates.values()) and not integrity["warnings"] else "WARN"
        return {
            "generated_at": now.isoformat(),
            "scope": relationship_id or "project",
            "status": status,
            "metrics": metrics,
            "gates": gates,
            "trust_bias_audit": trust_bias_audit,
            "coverage_summary": coverage_summary,
            "spec_coverage": spec_coverage,
            "integrity": integrity,
        }

    def _trust_bias_audit_summary(
        self,
        relationship_ids: list[str],
        relationships: list[Relationship],
        memories: list[MemoryRecord],
        now: datetime,
    ) -> dict[str, Any]:
        relationship_id_set = set(relationship_ids)
        enabled_relationships = [
            relationship
            for relationship in relationships
            if relationship.preferences.trust_bias_enabled and trust_bias_stage_enabled(relationship)
        ]
        harmful_memories = [
            memory
            for memory in memories
            if not is_trust_bias_protected(memory)
            and (
                memory.context_tag == ContextTag.CONFLICT
                or memory.memory_type == MemoryType.CONFLICT
                or memory.emotional_valence <= -0.5
            )
        ]
        critical_protected_memories = [memory for memory in memories if is_trust_bias_protected(memory)]
        soft_cooldown_memories = [memory for memory in harmful_memories if memory.metadata.get("trust_soft_cooldown")]
        protected_events = [
            item
            for item in self.deviation_log
            if item.get("relationship_id") in relationship_id_set and item.get("type") == "critical_memory_protected"
        ]
        relevant_logs = [
            item for item in self.retrieval_audit_log if item.get("relationship_id") in relationship_id_set
        ]
        adjusted_samples: list[dict[str, Any]] = []
        critical_exemption_samples: list[dict[str, Any]] = []
        monthly_samples = 0
        latest_sample_at: str | None = None
        month_key = now.strftime("%Y-%m")
        for log in relevant_logs:
            log_has_trust_sample = False
            for result in log.get("results", []):
                explanation = result.get("explanation", {})
                presentation = explanation.get("trust_presentation", {})
                if explanation.get("trust_bias_applied"):
                    adjusted_samples.append(
                        {
                            "at": log.get("at"),
                            "memory_id": result.get("memory_id"),
                            "mode": presentation.get("mode"),
                            "trust_level": log.get("trust_level"),
                        }
                    )
                    log_has_trust_sample = True
                if presentation.get("mode") == "raw_critical":
                    critical_exemption_samples.append(
                        {
                            "at": log.get("at"),
                            "memory_id": result.get("memory_id"),
                            "trust_level": log.get("trust_level"),
                        }
                    )
                    log_has_trust_sample = True
            if log_has_trust_sample:
                latest_sample_at = str(log.get("at") or latest_sample_at)
                if str(log.get("at", "")).startswith(month_key):
                    monthly_samples += 1
        has_risk_surface = bool(harmful_memories or critical_protected_memories)
        has_mechanism_evidence = (
            (not harmful_memories or bool(adjusted_samples or soft_cooldown_memories))
            and (not critical_protected_memories or bool(protected_events or critical_exemption_samples))
        )
        monthly_audit_ready = (
            True
            if not enabled_relationships or not has_risk_surface
            else has_mechanism_evidence and monthly_samples > 0
        )
        return {
            "enabled_relationships": len(enabled_relationships),
            "harmful_memory_candidates": len(harmful_memories),
            "critical_protected_memories": len(critical_protected_memories),
            "adjusted_retrieval_samples": len(adjusted_samples),
            "critical_exemption_samples": len(critical_exemption_samples),
            "soft_cooldown_samples": len(soft_cooldown_memories),
            "critical_protection_events": len(protected_events),
            "current_month_samples": monthly_samples,
            "latest_sample_at": latest_sample_at,
            "monthly_audit_ready": monthly_audit_ready,
            "sample_policy": "requires current-month trust-bias retrieval evidence when harmful or critical-protected memories exist",
            "adjusted_sample_preview": adjusted_samples[:5],
            "critical_exemption_preview": critical_exemption_samples[:5],
        }

    def _audit_spec_coverage(
        self,
        relationship_ids: list[str],
        relationships: list[Relationship],
        memories: list[MemoryRecord],
        metrics: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        storage_layers = self._storage_layer_counts(memories)
        l4_count = storage_layers.get(MemoryLayer.L4_CORE_IDENTITY.value, 0)
        l5_count = storage_layers.get(MemoryLayer.L5_RELATIONSHIP_HISTORY.value, 0)
        active_recall_events = metrics["active_behavior_events"]
        health_alerts = [
            item for item in self.health_alerts.values() if item.relationship_id in relationship_ids
        ]
        guardian_summaries = [
            item for item in self.guardian_summaries.values() if item.relationship_id in relationship_ids
        ]
        relationship_id_set = set(relationship_ids)
        migration_events = [
            item
            for item in self.deviation_log
            if item.get("type") in {"legacy_migration_applied", "legacy_migration_rolled_back"}
            and relationship_id_set.intersection(set(item.get("relationship_ids", [])))
        ]
        migration_batches = [
            batch
            for batch in self.migration_batches.values()
            if relationship_id_set.intersection(set(batch.get("relationship_ids", [])))
        ]
        reset_events = [
            item
            for item in self.deviation_log
            if item.get("relationship_id") in relationship_ids and item.get("type") in {"reset_requested", "reset_confirmed"}
        ]
        deletion_compliance_records = [
            item for item in self.deletion_compliance_log if item.get("relationship_id") in relationship_ids
        ]
        reverse_decay_enabled_relationships = len(
            [item for item in relationships if item.preferences.reverse_decay_enabled]
        )
        baseline_enabled_relationships = len(
            [item for item in relationships if item.preferences.baseline_detection_enabled]
        )
        stage_history_events = sum(len(item.stage_history) for item in relationships)
        milestone_refs = sum(len(item.milestones) for item in relationships)
        trust_bias_enabled_relationships = len(
            [item for item in relationships if item.preferences.trust_bias_enabled]
        )
        trust_bias_samples = (
            metrics["trust_bias_adjusted_samples"]
            + metrics["trust_bias_critical_exemption_samples"]
            + metrics["trust_bias_soft_cooldown_samples"]
        )
        minor_relationships = len([item for item in relationships if item.user_age is not None and item.user_age < 18])
        return {
            "relationship_modeling": {
                "sections": ["2", "12.2"],
                "implemented": True,
                "observed": bool(relationships) and all(item.interaction_count > 0 for item in relationships),
                "evidence": {
                    "relationship_count": metrics["relationships"],
                    "stage_history_events": stage_history_events,
                    "mode_controls_present": all(item.preferences.mode is not None for item in relationships),
                    "interactions": sum(item.interaction_count for item in relationships),
                },
            },
            "reverse_decay_and_storage": {
                "sections": ["3", "10"],
                "implemented": True,
                "observed": reverse_decay_enabled_relationships > 0 and (l4_count > 0 or l5_count > 0),
                "evidence": {
                    "storage_layers": storage_layers,
                    "reverse_decay_enabled_relationships": reverse_decay_enabled_relationships,
                    "l4_count": l4_count,
                    "l5_count": l5_count,
                },
            },
            "emotional_memory_layer": {
                "sections": ["4", "5.3"],
                "implemented": True,
                "observed": metrics["emotional_memories"] > 0 and baseline_enabled_relationships > 0,
                "evidence": {
                    "emotional_memories": metrics["emotional_memories"],
                    "baseline_enabled_relationships": baseline_enabled_relationships,
                },
            },
            "gating_and_scoring": {
                "sections": ["8"],
                "implemented": True,
                "observed": bool(memories) and all(item.importance >= 0 and item.base_weight > 0 for item in memories),
                "evidence": {
                    "scored_memories": len(memories),
                    "memory_types": sorted({item.memory_type.value for item in memories}),
                    "context_tags": sorted({item.context_tag.value for item in memories}),
                },
            },
            "active_recall_and_feedback": {
                "sections": ["5"],
                "implemented": True,
                "observed": active_recall_events > 0,
                "evidence": {
                    "active_behavior_events": active_recall_events,
                    "active_feedback_events": metrics["active_feedback_events"],
                    "active_acceptance_rate": metrics["active_acceptance_rate"],
                    "active_complaint_events": metrics["active_complaint_events"],
                    "active_complaint_rate": metrics["active_complaint_rate"],
                    "active_complaint_rate_target": 0.05,
                },
            },
            "shared_narrative_and_milestones": {
                "sections": ["6"],
                "implemented": True,
                "observed": metrics["story_nodes"] > 0 and milestone_refs > 0,
                "evidence": {
                    "story_nodes": metrics["story_nodes"],
                    "milestone_refs": milestone_refs,
                },
            },
            "trust_bias_and_retrieval": {
                "sections": ["7", "9"],
                "implemented": True,
                "observed": (
                    metrics["retrieval_audit_events"] > 0
                    and trust_bias_enabled_relationships > 0
                    and trust_bias_samples > 0
                    and metrics["trust_bias_monthly_audit_ready"]
                ),
                "evidence": {
                    "retrieval_audit_events": metrics["retrieval_audit_events"],
                    "trust_bias_enabled_relationships": trust_bias_enabled_relationships,
                    "adjusted_retrieval_samples": metrics["trust_bias_adjusted_samples"],
                    "critical_exemption_samples": metrics["trust_bias_critical_exemption_samples"],
                    "soft_cooldown_samples": metrics["trust_bias_soft_cooldown_samples"],
                    "monthly_audit_ready": metrics["trust_bias_monthly_audit_ready"],
                },
            },
            "transparency_and_user_control": {
                "sections": ["11", "13.1", "13.3"],
                "implemented": True,
                "observed": bool(relationships) and metrics["transparency_acknowledged"] == metrics["relationships"],
                "evidence": {
                    "transparency_acknowledged": metrics["transparency_acknowledged"],
                    "transparency_pending": metrics["transparency_pending"],
                    "memory_write_controls_present": all(
                        hasattr(item.preferences, "memory_writes_enabled") for item in relationships
                    ),
                },
            },
            "deletion_and_relationship_ending": {
                "sections": ["11.4", "13.5", "14.4"],
                "implemented": True,
                "observed": (
                    metrics["pending_memory_deletes"] + metrics["executed_memory_deletes"] + metrics["executed_resets"] > 0
                    and bool(deletion_compliance_records)
                ),
                "evidence": {
                    "pending_memory_deletes": metrics["pending_memory_deletes"],
                    "executed_memory_deletes": metrics["executed_memory_deletes"],
                    "executed_resets": metrics["executed_resets"],
                    "deletion_compliance_records": len(deletion_compliance_records),
                    "relationship_ending_support_events": len(
                        [item for item in self.relationship_ending_support_log if item.get("relationship_id") in relationship_ids]
                    ),
                    "reset_events": len(reset_events),
                },
            },
            "health_and_minor_guardrails": {
                "sections": ["13.2", "13.4", "13.5"],
                "implemented": True,
                "observed": bool(health_alerts) and bool(guardian_summaries) and minor_relationships > 0,
                "evidence": {
                    "health_alerts_open": metrics["health_alerts_open"],
                    "health_alerts_total": len(health_alerts),
                    "guardian_summaries": len(guardian_summaries),
                    "minor_relationships": minor_relationships,
                },
            },
            "migration_and_compatibility": {
                "sections": ["12"],
                "implemented": True,
                "observed": bool(migration_batches) and bool(migration_events),
                "evidence": {
                    "migration_batches": len(migration_batches),
                    "migration_events": len(migration_events),
                },
            },
        }

    def _audit_coverage_summary(self, spec_coverage: dict[str, dict[str, Any]]) -> dict[str, Any]:
        module_count = len(spec_coverage)
        implemented = [name for name, item in spec_coverage.items() if item.get("implemented")]
        observed = [name for name, item in spec_coverage.items() if item.get("observed")]
        unobserved = [name for name in spec_coverage if name not in observed]
        return {
            "module_count": module_count,
            "implemented_module_count": len(implemented),
            "observed_module_count": len(observed),
            "unobserved_modules": unobserved,
            "observation_ratio": (len(observed) / module_count) if module_count else 1.0,
            "completion_evidence_complete": len(unobserved) == 0,
            "note": (
                "Observed means the current project state contains runtime evidence for that module. "
                "Unobserved modules may still be implemented but need scenario evidence before claiming completion."
            ),
        }

    def decision_report(
        self,
        relationship_id: str | None = None,
        *,
        now: datetime | None = None,
        run_benchmarks: bool = False,
        benchmark_iterations: int = 20,
        evaluation_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        audit = self.audit_report(relationship_id, now=now)
        latency_benchmark = (
            self.benchmark_latency(relationship_id, iterations=benchmark_iterations, now=now)
            if run_benchmarks
            else None
        )
        cascade_latency_benchmark = (
            self.benchmark_cascade_latency(relationship_id, iterations=benchmark_iterations, now=now)
            if run_benchmarks
            else None
        )
        write_overhead_benchmark = (
            self.benchmark_write_overhead(relationship_id, iterations=benchmark_iterations, now=now)
            if run_benchmarks
            else None
        )
        evaluation_results = evaluation_results or {}
        stage_eval = evaluation_results.get("stage_detection_accuracy")
        inside_joke_eval = evaluation_results.get("inside_joke_detection_accuracy")
        emotional_resonance_eval = evaluation_results.get("emotional_resonance_p5")
        self_disclosure_eval = evaluation_results.get("l2_self_disclosure_capture")
        story_quality_eval = evaluation_results.get("story_sampling_quality")
        friend_mode_ab_eval = evaluation_results.get("friend_mode_ab_metrics")
        production_telemetry_eval = evaluation_results.get("production_telemetry")
        relationship_ids = [relationship_id] if relationship_id else list(self.relationships)
        relationships = [self.relationships[item] for item in relationship_ids if item in self.relationships]
        memories = [memory for memory in self.memories.values() if memory.relationship_id in relationship_ids]
        retention_evidence = self._relationship_retention_ratio_evidence(relationships, memories, now)
        l4_total = len([memory for memory in memories if memory.storage_layer == MemoryLayer.L4_CORE_IDENTITY])
        reviewed_l4_memory_ids = {
            identity.memory_id
            for identity in self.core_identity.values()
            if identity.relationship_id in relationship_ids and identity.review_status in {"AI_REVIEWED", "USER_CONFIRMED"}
        }
        l4_reviewed = len(
            [
                memory
                for memory in memories
                if memory.storage_layer == MemoryLayer.L4_CORE_IDENTITY
                and (
                    memory.memory_id in reviewed_l4_memory_ids
                    or memory.metadata.get("l4_review", {}).get("review_status") in {"AI_REVIEWED", "USER_CONFIRMED"}
                )
            ]
        )
        story_scores = [story.consistency_score for story in self.story_nodes.values() if story.relationship_id in relationship_ids]
        hard_reset_events = [
            item
            for item in self.deviation_log
            if item.get("relationship_id") in relationship_ids
            and item.get("type") == "reset_confirmed"
            and item.get("mode") == ResetMode.HARD.value
        ]
        health_prompt_response = self._health_prompt_response_evidence(relationship_ids)
        ai_readiness = self._ai_readiness_evidence(relationship_ids)
        report = {
            "generated_at": now.isoformat(),
            "scope": relationship_id or "project",
            "source_audit_status": audit["status"],
            "source_coverage_complete": audit["coverage_summary"]["completion_evidence_complete"],
            "ai_readiness": ai_readiness,
            "phases": {
                "F-1": self._decision_phase(
                    "F-1",
                    "Phase 1 relationship context",
                    [
                        self._criterion(
                            "relationship_object_runtime",
                            "Relationship object has runtime interactions",
                            ">=1 relationship with interaction_count > 0",
                            "pass" if relationships and all(item.interaction_count > 0 for item in relationships) else "missing_evidence",
                            {"relationships": len(relationships), "interactions": sum(item.interaction_count for item in relationships)},
                        ),
                        self._criterion(
                            "stage_detection_accuracy",
                            "Knapp stage detection accuracy",
                            ">=75% on 200 manually labelled dialogues",
                            str(stage_eval.get("status"))
                            if isinstance(stage_eval, dict) and stage_eval.get("status")
                            else "external_required",
                            stage_eval
                            if isinstance(stage_eval, dict)
                            else {"project_proxy": "stage_history_events", "value": sum(len(item.stage_history) for item in relationships)},
                        ),
                        self._criterion(
                            "reverse_decay_weight_ratio",
                            "Key relationship memories retain more strongly than ordinary facts",
                            "1-year key relationship weight >= ordinary fact weight * 5",
                            "pass"
                            if retention_evidence.get("ratio") is not None and retention_evidence["ratio"] >= 5.0
                            else "missing_evidence",
                            retention_evidence,
                        ),
                        self._criterion(
                            "l2_self_disclosure_capture",
                            "Friend-mode L2 captures self-disclosure memories",
                            ">=90% capture on labelled A/B set",
                            str(self_disclosure_eval.get("status"))
                            if isinstance(self_disclosure_eval, dict) and self_disclosure_eval.get("status")
                            else "external_required",
                            self_disclosure_eval
                            if isinstance(self_disclosure_eval, dict)
                            else {
                                "self_disclosure_like_memories": len(
                                    [
                                        memory
                                        for memory in memories
                                        if memory.context_tag == ContextTag.VULNERABLE_MOMENT
                                        or memory.memory_type in {MemoryType.EMOTIONAL_MOMENT, MemoryType.IDENTITY}
                                    ]
                                )
                            },
                        ),
                        self._criterion(
                            "relationship_dashboard_visible",
                            "Relationship dashboard exposes strength/stage/age",
                            "browser snapshot has dashboard fields",
                            "pass"
                            if relationships
                            and all(
                                {"strength", "stage", "relationship_age"}.issubset(
                                    set(self.browser_snapshot(item.relationship_id)["dashboard"])
                                )
                                for item in relationships
                            )
                            else "missing_evidence",
                            {"relationships_checked": len(relationships)},
                        ),
                        self._criterion(
                            "relationship_write_overhead",
                            "Relationship write overhead",
                            "<= assistant Stage 1 write overhead * 1.3",
                            self._write_overhead_status(write_overhead_benchmark)
                            if write_overhead_benchmark
                            else "external_required",
                            write_overhead_benchmark
                            or {"reason": "run decision-report with run_benchmarks=true to collect friend and assistant write baselines"},
                        ),
                    ],
                ),
                "F-2": self._decision_phase(
                    "F-2",
                    "Phase 2 emotional layer and active recall",
                    [
                        self._criterion(
                            "emotional_memory_and_embeddings",
                            "EmotionalMemory and composite feature metadata are present",
                            "emotional memories > 0 and memories carry feature metadata",
                            "pass"
                            if audit["metrics"]["emotional_memories"] > 0
                            and any(memory.metadata.get("embeddings") for memory in memories)
                            else "missing_evidence",
                            {
                                "emotional_memories": audit["metrics"]["emotional_memories"],
                                "memories_with_embeddings": len([memory for memory in memories if memory.metadata.get("embeddings")]),
                            },
                        ),
                        self._criterion(
                            "active_acceptance_rate",
                            "Active recall acceptance rate",
                            ">=60% accepted or neutral feedback",
                            self._threshold_status(audit["metrics"]["active_acceptance_rate"], 0.60, higher_is_better=True),
                            {"value": audit["metrics"]["active_acceptance_rate"], "target": 0.60},
                        ),
                        self._criterion(
                            "active_complaint_rate",
                            "Active recall interruption complaint rate",
                            "<=5%",
                            self._threshold_status(audit["metrics"]["active_complaint_rate"], 0.05, higher_is_better=False),
                            {"value": audit["metrics"]["active_complaint_rate"], "target": 0.05},
                        ),
                        self._criterion(
                            "emotional_resonance_p5",
                            "Emotional resonance retrieval P@5",
                            ">=0.65 on manually labelled set",
                            str(emotional_resonance_eval.get("status"))
                            if isinstance(emotional_resonance_eval, dict) and emotional_resonance_eval.get("status")
                            else "external_required",
                            emotional_resonance_eval
                            if isinstance(emotional_resonance_eval, dict)
                            else {"project_proxy": "emotional_resonance active/retrieval evidence is audited separately"},
                        ),
                        self._criterion(
                            "inside_joke_detection_accuracy",
                            "Inside joke candidate accuracy",
                            ">=70% on sampled candidates",
                            str(inside_joke_eval.get("status"))
                            if isinstance(inside_joke_eval, dict) and inside_joke_eval.get("status")
                            else "external_required",
                            inside_joke_eval
                            if isinstance(inside_joke_eval, dict)
                            else {"inside_joke_candidates": sum(len(item.inside_jokes) for item in relationships)},
                        ),
                        self._criterion(
                            "l4_review_zero_error_process",
                            "L4 writes are review-gated",
                            "all L4 writes reviewed by AI or user",
                            "pass" if l4_total == l4_reviewed else "missing_evidence",
                            {"l4_total": l4_total, "l4_reviewed": l4_reviewed},
                        ),
                        self._criterion(
                            "cascade_latency_p95",
                            "Five-layer cascade read latency",
                            "P95 <= 2.5 seconds and <= +25% over assistant mode",
                            self._cascade_latency_status(cascade_latency_benchmark)
                            if cascade_latency_benchmark
                            else "external_required",
                            cascade_latency_benchmark
                            or {"reason": "run decision-report with run_benchmarks=true to collect cascade and assistant-mode baselines"},
                        ),
                    ],
                ),
                "F-3": self._decision_phase(
                    "F-3",
                    "Phase 3 stories, trust, transparency and safety",
                    [
                        self._criterion(
                            "friend_mode_ab_metrics",
                            "Friend mode A/B product metrics",
                            "NPS +20, retention +10%, session +30%, intimacy +1.5",
                            str(friend_mode_ab_eval.get("status"))
                            if isinstance(friend_mode_ab_eval, dict) and friend_mode_ab_eval.get("status")
                            else "external_required",
                            friend_mode_ab_eval
                            if isinstance(friend_mode_ab_eval, dict)
                            else {"reason": "requires >=1000-user 12-week A/B test"},
                        ),
                        self._criterion(
                            "story_sampling_quality",
                            "SharedStoryNode sampling quality",
                            ">=4/5 on 200 sampled story nodes",
                            str(story_quality_eval.get("status"))
                            if isinstance(story_quality_eval, dict) and story_quality_eval.get("status")
                            else (
                                "external_required"
                                if len(story_scores) < 200
                                else self._threshold_status(
                                    sum(story_scores) / len(story_scores),
                                    0.80,
                                    higher_is_better=True,
                                )
                            ),
                            story_quality_eval
                            if isinstance(story_quality_eval, dict)
                            else {
                                "story_nodes": len(story_scores),
                                "average_consistency_score": (sum(story_scores) / len(story_scores)) if story_scores else None,
                                "sample_target": 200,
                            },
                        ),
                        self._criterion(
                            "trust_bias_critical_safety",
                            "Trust bias does not hide CRITICAL memories",
                            "100% audit sample exemption",
                            "pass" if audit["trust_bias_audit"]["monthly_audit_ready"] else "missing_evidence",
                            audit["trust_bias_audit"],
                        ),
                        self._criterion(
                            "inside_joke_recall_acceptance",
                            "Inside joke recall acceptance rate",
                            ">=70%",
                            self._inside_joke_acceptance_status(relationships),
                            self._inside_joke_acceptance_evidence(relationships),
                        ),
                        self._criterion(
                            "transparency_panel_usage",
                            "Transparency panel usage",
                            ">=30% active users opened or acknowledged transparency",
                            self._threshold_status(
                                audit["metrics"]["transparency_acknowledged"] / audit["metrics"]["relationships"]
                                if audit["metrics"]["relationships"]
                                else None,
                                0.30,
                                higher_is_better=True,
                            ),
                            {
                                "acknowledged": audit["metrics"]["transparency_acknowledged"],
                                "relationships": audit["metrics"]["relationships"],
                            },
                        ),
                        self._criterion(
                            "hard_reset_deletion_propagation",
                            "HARD_RESET deletion propagation",
                            "100% clean deletion propagation when HARD_RESET is exercised",
                            "pass" if not hard_reset_events or audit["gates"]["delete_propagation_clean"] else "fail",
                            {"hard_reset_events": len(hard_reset_events), "delete_propagation_clean": audit["gates"]["delete_propagation_clean"]},
                        ),
                        self._criterion(
                            "health_prompt_response_rate",
                            "Health prompt response rate",
                            ">=40%",
                            self._threshold_status(health_prompt_response["response_rate"], 0.40, higher_is_better=True),
                            health_prompt_response,
                        ),
                        self._criterion(
                            "hybrid_retrieval_latency_p95",
                            "Hybrid retrieval latency",
                            "P95 <= 3 seconds",
                            self._threshold_status(
                                latency_benchmark["retrieval_p95_seconds"] if latency_benchmark else None,
                                3.0,
                                higher_is_better=False,
                            )
                            if latency_benchmark
                            else "external_required",
                            latency_benchmark
                            or {"reason": "run decision-report with run_benchmarks=true to collect local retrieval P95"},
                        ),
                        self._criterion(
                            "production_telemetry_safety",
                            "Production safety and control telemetry",
                            "30-day telemetry with >=1000 active users and safety/control metrics passing",
                            str(production_telemetry_eval.get("status"))
                            if isinstance(production_telemetry_eval, dict) and production_telemetry_eval.get("status")
                            else "external_required",
                            production_telemetry_eval
                            if isinstance(production_telemetry_eval, dict)
                            else {"reason": "requires production-telemetry.json with 30-day safety/control metrics"},
                        ),
                    ],
                ),
            },
        }
        report["summary"] = self._decision_report_summary(report["phases"])
        report["evidence_manifest"] = self._decision_evidence_manifest(report["phases"])
        return report

    def evaluate_labeled_dataset(self, dataset: dict[str, Any] | list[dict[str, Any]], *, task: str) -> dict[str, Any]:
        if isinstance(dataset, dict):
            raw_examples = dataset.get("examples", [])
            config = dataset.get("config", {})
            metrics = dataset.get("metrics", {})
        elif isinstance(dataset, list):
            raw_examples = dataset
            config = {}
            metrics = {}
        else:
            raw_examples = []
            config = {}
            metrics = {}
        examples = [
            item if isinstance(item, dict) else {}
            for item in raw_examples
        ] if isinstance(raw_examples, list) else []
        if not isinstance(config, dict):
            config = {}
        if not isinstance(metrics, dict):
            metrics = {}
        evaluators = {
            "stage_detection": lambda: self._evaluate_stage_detection_labels(examples, config),
            "inside_joke_detection": lambda: self._evaluate_inside_joke_detection_labels(examples, config),
            "emotional_resonance_retrieval": lambda: self._evaluate_emotional_resonance_retrieval_labels(examples, config),
            "self_disclosure_capture": lambda: self._evaluate_self_disclosure_capture_labels(examples, config),
            "story_quality": lambda: self._evaluate_story_quality_labels(examples, config),
            "friend_mode_ab": lambda: self._evaluate_friend_mode_ab_metrics(examples, config),
            "production_telemetry": lambda: self._evaluate_production_telemetry(metrics, config),
        }
        evaluator = evaluators.get(task)
        if evaluator is not None:
            try:
                return evaluator()
            except (TypeError, ValueError) as exc:
                return self._invalid_evaluation_result(task, exc)
        raise ValueError(
            "task must be one of: stage_detection, inside_joke_detection, emotional_resonance_retrieval, "
            "self_disclosure_capture, story_quality, friend_mode_ab, production_telemetry"
        )

    def _invalid_evaluation_result(self, task: str, exc: Exception) -> dict[str, Any]:
        task_names = {
            "stage_detection": "stage_detection_accuracy",
            "inside_joke_detection": "inside_joke_detection_accuracy",
            "emotional_resonance_retrieval": "emotional_resonance_p5",
            "self_disclosure_capture": "l2_self_disclosure_capture",
            "story_quality": "story_sampling_quality",
            "friend_mode_ab": "friend_mode_ab_metrics",
            "production_telemetry": "production_telemetry",
        }
        return {
            "task": task_names.get(task, task),
            "status": "invalid_input",
            "sample_count": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    def _evaluate_stage_detection_labels(self, examples: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        target_accuracy = float(config.get("target_accuracy", 0.75))
        required_samples = int(config.get("required_samples", 200))
        valid = 0
        correct = 0
        invalid: list[dict[str, Any]] = []
        mismatches: list[dict[str, Any]] = []
        for index, item in enumerate(examples):
            text = str(item.get("text", "")).strip()
            expected_raw = str(item.get("expected_stage", "")).strip()
            if not text or not expected_raw:
                invalid.append({"index": index, "reason": "missing_text_or_expected_stage"})
                continue
            try:
                expected = RelationshipStage(expected_raw.upper())
            except ValueError:
                invalid.append({"index": index, "reason": "invalid_expected_stage", "expected_stage": expected_raw})
                continue
            evaluation_project = FriendMemoryProject(ai=self.ai)
            relationship = evaluation_project.get_or_create_relationship(f"eval_user_{index}", "eval_ai")
            initial_stage = item.get("initial_stage")
            if initial_stage:
                try:
                    relationship.stage = RelationshipStage(str(initial_stage).upper())
                except ValueError:
                    invalid.append({"index": index, "reason": "invalid_initial_stage", "initial_stage": initial_stage})
                    continue
            relationship.strength = clamp(float(item.get("strength", relationship.strength)))
            relationship.trust_level = clamp(float(item.get("trust_level", relationship.trust_level)))
            relationship.intimacy_level = clamp(float(item.get("intimacy_level", relationship.intimacy_level)))
            timestamp = utcnow() + timedelta(minutes=index)
            evaluation_project.ingest_turn(relationship.user_id, relationship.ai_id, text, timestamp=timestamp)
            predicted = evaluation_project.relationships[relationship.relationship_id].stage
            valid += 1
            if predicted == expected:
                correct += 1
            else:
                mismatches.append(
                    {
                        "index": index,
                        "text_excerpt": text[:80],
                        "expected_stage": expected.value,
                        "predicted_stage": predicted.value,
                    }
                )
        accuracy = (correct / valid) if valid else None
        status = self._threshold_status(accuracy, target_accuracy, higher_is_better=True)
        if valid < required_samples:
            status = "missing_evidence" if status == "pass" else status
        return {
            "task": "stage_detection_accuracy",
            "sample_count": valid,
            "required_samples": required_samples,
            "correct": correct,
            "accuracy": accuracy,
            "target_accuracy": target_accuracy,
            "status": status,
            "invalid_examples": invalid[:20],
            "mismatches": mismatches[:20],
        }

    def _evaluate_inside_joke_detection_labels(self, examples: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        target_accuracy = float(config.get("target_accuracy", 0.70))
        required_samples = int(config.get("required_samples", 200))
        valid = 0
        correct = 0
        invalid: list[dict[str, Any]] = []
        mismatches: list[dict[str, Any]] = []
        for index, item in enumerate(examples):
            turns = item.get("turns")
            if isinstance(turns, str):
                turns = [turns]
            if not isinstance(turns, list) or not turns:
                invalid.append({"index": index, "reason": "missing_turns"})
                continue
            expected_detected = bool(item.get("expected_detected", item.get("expected_inside_joke", False)))
            expected_phrase = str(item.get("expected_phrase", "")).strip().lower()
            evaluation_project = FriendMemoryProject(ai=self.ai)
            user_id = f"joke_eval_user_{index}"
            ai_id = "joke_eval_ai"
            start = utcnow()
            for turn_index, text in enumerate(turns):
                if not isinstance(text, str) or not text.strip():
                    invalid.append({"index": index, "turn_index": turn_index, "reason": "empty_turn"})
                    continue
                evaluation_project.ingest_turn(user_id, ai_id, text, timestamp=start + timedelta(days=turn_index * 4))
            relationship = evaluation_project.relationships.get(f"{user_id}:{ai_id}")
            if relationship is None:
                invalid.append({"index": index, "reason": "relationship_not_created"})
                continue
            candidate_phrases = set(relationship.inside_joke_candidates)
            promoted_phrases = {
                str(evaluation_project.memories[memory_id].metadata.get("inside_joke_phrase", "")).strip().lower()
                for memory_id in relationship.inside_jokes
                if memory_id in evaluation_project.memories
            }
            detected_phrases = candidate_phrases.union(promoted_phrases)
            detected = bool(detected_phrases)
            if expected_phrase:
                detected = expected_phrase in detected_phrases
            valid += 1
            matched = detected == expected_detected
            if matched:
                correct += 1
            else:
                mismatches.append(
                    {
                        "index": index,
                        "expected_detected": expected_detected,
                        "expected_phrase": expected_phrase or None,
                        "detected_phrases": sorted(detected_phrases),
                        "turn_count": len(turns),
                    }
                )
        accuracy = (correct / valid) if valid else None
        status = self._threshold_status(accuracy, target_accuracy, higher_is_better=True)
        if valid < required_samples:
            status = "missing_evidence" if status == "pass" else status
        return {
            "task": "inside_joke_detection_accuracy",
            "sample_count": valid,
            "required_samples": required_samples,
            "correct": correct,
            "accuracy": accuracy,
            "target_accuracy": target_accuracy,
            "status": status,
            "invalid_examples": invalid[:20],
            "mismatches": mismatches[:20],
        }

    def _evaluate_emotional_resonance_retrieval_labels(
        self, examples: list[dict[str, Any]], config: dict[str, Any]
    ) -> dict[str, Any]:
        target_p5 = float(config.get("target_p5", 0.65))
        required_samples = int(config.get("required_samples", 200))
        valid = 0
        p5_scores: list[float] = []
        invalid: list[dict[str, Any]] = []
        misses: list[dict[str, Any]] = []
        for index, item in enumerate(examples):
            memories = item.get("memories")
            query = str(item.get("query", "")).strip()
            if not isinstance(memories, list) or not memories or not query:
                invalid.append({"index": index, "reason": "missing_memories_or_query"})
                continue
            relevant_indices = {
                int(value)
                for value in item.get("expected_relevant_indices", [])
                if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
            }
            relevant_contains = [str(value) for value in item.get("expected_relevant_contains", []) if str(value)]
            if not relevant_indices and not relevant_contains:
                invalid.append({"index": index, "reason": "missing_relevance_labels"})
                continue
            evaluation_project = FriendMemoryProject(ai=self.ai)
            user_id = f"emotion_eval_user_{index}"
            ai_id = "emotion_eval_ai"
            start = utcnow()
            relationship = evaluation_project.get_or_create_relationship(user_id, ai_id)
            relationship.stage = RelationshipStage.INTEGRATING
            relationship.strength = 0.8
            relationship.trust_level = 0.8
            index_to_memory_id: dict[int, str] = {}
            for memory_index, memory_text in enumerate(memories):
                if not isinstance(memory_text, str) or not memory_text.strip():
                    invalid.append({"index": index, "memory_index": memory_index, "reason": "empty_memory"})
                    continue
                result = evaluation_project.ingest_turn(
                    user_id,
                    ai_id,
                    memory_text,
                    timestamp=start + timedelta(days=memory_index),
                )
                if result.memory_id:
                    index_to_memory_id[memory_index] = result.memory_id
            expected_ids = {index_to_memory_id[value] for value in relevant_indices if value in index_to_memory_id}
            for memory_id, memory in evaluation_project.memories.items():
                if any(fragment in memory.content for fragment in relevant_contains):
                    expected_ids.add(memory_id)
            if not expected_ids:
                invalid.append({"index": index, "reason": "no_expected_memory_ids_resolved"})
                continue
            results = evaluation_project.retrieve(
                relationship.relationship_id,
                query,
                now=start + timedelta(days=len(memories) + 1),
                limit=5,
                audit=False,
            )
            result_ids = [result.memory.memory_id for result in results]
            hits = [memory_id for memory_id in result_ids if memory_id in expected_ids]
            denominator = max(1, min(5, len(result_ids)))
            p5 = len(hits) / denominator
            valid += 1
            p5_scores.append(p5)
            if p5 < target_p5:
                misses.append(
                    {
                        "index": index,
                        "query": query[:80],
                        "p_at_5": p5,
                        "expected_memory_ids": sorted(expected_ids),
                        "retrieved_memory_ids": result_ids,
                    }
                )
        average_p5 = (sum(p5_scores) / len(p5_scores)) if p5_scores else None
        status = self._threshold_status(average_p5, target_p5, higher_is_better=True)
        if valid < required_samples:
            status = "missing_evidence" if status == "pass" else status
        return {
            "task": "emotional_resonance_p5",
            "sample_count": valid,
            "required_samples": required_samples,
            "average_p_at_5": average_p5,
            "target_p_at_5": target_p5,
            "status": status,
            "scores": p5_scores[:50],
            "invalid_examples": invalid[:20],
            "misses": misses[:20],
            "denominator_note": "uses top-k denominator min(5, actual_result_count) for small labelled fixtures",
        }

    def _evaluate_self_disclosure_capture_labels(
        self, examples: list[dict[str, Any]], config: dict[str, Any]
    ) -> dict[str, Any]:
        target_recall = float(config.get("target_recall", 0.90))
        required_samples = int(config.get("required_samples", 200))
        valid = 0
        positives = 0
        captured = 0
        false_positives = 0
        invalid: list[dict[str, Any]] = []
        misses: list[dict[str, Any]] = []
        false_positive_examples: list[dict[str, Any]] = []
        for index, item in enumerate(examples):
            text = str(item.get("text", "")).strip()
            if not text:
                invalid.append({"index": index, "reason": "missing_text"})
                continue
            expected = bool(item.get("expected_self_disclosure", item.get("expected_captured", True)))
            evaluation_project = FriendMemoryProject(ai=self.ai)
            relationship = evaluation_project.get_or_create_relationship(f"self_eval_user_{index}", "self_eval_ai")
            try:
                relationship.stage = RelationshipStage(str(item.get("stage", RelationshipStage.EXPERIMENTING.value)).upper())
            except ValueError:
                invalid.append({"index": index, "reason": "invalid_stage", "stage": item.get("stage")})
                continue
            relationship.strength = clamp(float(item.get("strength", relationship.strength)))
            relationship.trust_level = clamp(float(item.get("trust_level", relationship.trust_level)))
            relationship.intimacy_level = clamp(float(item.get("intimacy_level", relationship.intimacy_level)))
            result = evaluation_project.ingest_turn(relationship.user_id, relationship.ai_id, text, timestamp=utcnow())
            if not result.memory_id or result.memory_id not in evaluation_project.memories:
                invalid.append({"index": index, "reason": "memory_not_written"})
                continue
            memory = evaluation_project.memories[result.memory_id]
            signals = detect_turn_signals(text, relationship.strength, relationship.trust_level, relationship.intimacy_level)
            is_captured = (
                signals.self_disclosure_depth >= 0.70
                or memory.context_tag == ContextTag.VULNERABLE_MOMENT
                or memory.memory_type in {MemoryType.EMOTIONAL_MOMENT, MemoryType.IDENTITY, MemoryType.MILESTONE}
                or memory.storage_layer in {MemoryLayer.L3_RELATIONAL, MemoryLayer.L5_RELATIONSHIP_HISTORY}
            )
            valid += 1
            if expected:
                positives += 1
                if is_captured:
                    captured += 1
                else:
                    misses.append(
                        {
                            "index": index,
                            "text_excerpt": text[:80],
                            "memory_type": memory.memory_type.value,
                            "context_tag": memory.context_tag.value,
                            "storage_layer": memory.storage_layer.value,
                            "self_disclosure_depth": signals.self_disclosure_depth,
                        }
                    )
            elif is_captured:
                false_positives += 1
                false_positive_examples.append(
                    {
                        "index": index,
                        "text_excerpt": text[:80],
                        "memory_type": memory.memory_type.value,
                        "context_tag": memory.context_tag.value,
                        "storage_layer": memory.storage_layer.value,
                        "self_disclosure_depth": signals.self_disclosure_depth,
                    }
                )
        recall = (captured / positives) if positives else None
        status = self._threshold_status(recall, target_recall, higher_is_better=True)
        if valid < required_samples:
            status = "missing_evidence" if status == "pass" else status
        return {
            "task": "l2_self_disclosure_capture",
            "sample_count": valid,
            "required_samples": required_samples,
            "positive_labels": positives,
            "captured": captured,
            "recall": recall,
            "target_recall": target_recall,
            "false_positives": false_positives,
            "status": status,
            "invalid_examples": invalid[:20],
            "misses": misses[:20],
            "false_positive_examples": false_positive_examples[:20],
        }

    def _evaluate_story_quality_labels(self, examples: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        target_average_score = float(config.get("target_average_score", 4.0))
        required_samples = int(config.get("required_samples", 200))
        valid = 0
        scores: list[float] = []
        invalid: list[dict[str, Any]] = []
        low_score_examples: list[dict[str, Any]] = []
        for index, item in enumerate(examples):
            score_raw = item.get("score", item.get("quality_score"))
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                invalid.append({"index": index, "reason": "missing_or_invalid_score"})
                continue
            if score < 1.0 or score > 5.0:
                invalid.append({"index": index, "reason": "score_out_of_range", "score": score})
                continue
            valid += 1
            scores.append(score)
            if score < target_average_score:
                low_score_examples.append(
                    {
                        "index": index,
                        "story_id": item.get("story_id"),
                        "title": item.get("title"),
                        "score": score,
                        "note": item.get("note"),
                    }
                )
        average_score = (sum(scores) / len(scores)) if scores else None
        normalized_average = (average_score / 5.0) if average_score is not None else None
        status = self._threshold_status(
            normalized_average,
            target_average_score / 5.0,
            higher_is_better=True,
        )
        if valid < required_samples:
            status = "missing_evidence" if status == "pass" else status
        return {
            "task": "story_sampling_quality",
            "sample_count": valid,
            "required_samples": required_samples,
            "average_score": average_score,
            "target_average_score": target_average_score,
            "normalized_average": normalized_average,
            "status": status,
            "invalid_examples": invalid[:20],
            "low_score_examples": low_score_examples[:20],
        }

    def _evaluate_friend_mode_ab_metrics(self, examples: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        min_users_per_cohort = int(config.get("min_users_per_cohort", 1000))
        min_duration_weeks = float(config.get("min_duration_weeks", 12.0))
        target_nps_lift = float(config.get("target_nps_lift", 20.0))
        target_retention_lift = float(config.get("target_retention_lift", 0.10))
        target_session_lift = float(config.get("target_session_lift", 0.30))
        target_intimacy_lift = float(config.get("target_intimacy_lift", 1.5))
        control = self._cohort_metric(examples, "control")
        friend = self._cohort_metric(examples, "friend")
        invalid: list[dict[str, Any]] = []
        if control is None:
            invalid.append({"reason": "missing_control_cohort"})
        if friend is None:
            invalid.append({"reason": "missing_friend_cohort"})
        if invalid:
            return {
                "task": "friend_mode_ab_metrics",
                "status": "missing_evidence",
                "invalid_examples": invalid,
                "requirements": {
                    "min_users_per_cohort": min_users_per_cohort,
                    "min_duration_weeks": min_duration_weeks,
                    "target_nps_lift": target_nps_lift,
                    "target_retention_lift": target_retention_lift,
                    "target_session_lift": target_session_lift,
                    "target_intimacy_lift": target_intimacy_lift,
                },
            }
        assert control is not None and friend is not None
        control_users = int(control.get("users", control.get("sample_size", 0)) or 0)
        friend_users = int(friend.get("users", friend.get("sample_size", 0)) or 0)
        duration_weeks = float(
            config.get("duration_weeks", friend.get("duration_weeks", control.get("duration_weeks", 0.0))) or 0.0
        )
        control_nps = float(control.get("nps", 0.0) or 0.0)
        friend_nps = float(friend.get("nps", 0.0) or 0.0)
        control_retention = float(control.get("retention_rate", control.get("retention", 0.0)) or 0.0)
        friend_retention = float(friend.get("retention_rate", friend.get("retention", 0.0)) or 0.0)
        control_session = float(control.get("avg_session_minutes", control.get("session_minutes", 0.0)) or 0.0)
        friend_session = float(friend.get("avg_session_minutes", friend.get("session_minutes", 0.0)) or 0.0)
        control_intimacy = float(control.get("avg_intimacy_delta", control.get("intimacy_delta", 0.0)) or 0.0)
        friend_intimacy = float(friend.get("avg_intimacy_delta", friend.get("intimacy_delta", 0.0)) or 0.0)
        nps_lift = friend_nps - control_nps
        retention_lift = (friend_retention - control_retention) / control_retention if control_retention > 0 else None
        session_lift = (friend_session - control_session) / control_session if control_session > 0 else None
        intimacy_lift = friend_intimacy - control_intimacy
        sample_ready = (
            control_users >= min_users_per_cohort
            and friend_users >= min_users_per_cohort
            and duration_weeks >= min_duration_weeks
        )
        metric_passes = {
            "nps_lift": nps_lift >= target_nps_lift,
            "retention_lift": retention_lift is not None and retention_lift >= target_retention_lift,
            "session_lift": session_lift is not None and session_lift >= target_session_lift,
            "intimacy_lift": intimacy_lift >= target_intimacy_lift,
        }
        if not sample_ready:
            status = "missing_evidence"
        else:
            status = "pass" if all(metric_passes.values()) else "fail"
        return {
            "task": "friend_mode_ab_metrics",
            "status": status,
            "sample_ready": sample_ready,
            "control_users": control_users,
            "friend_users": friend_users,
            "duration_weeks": duration_weeks,
            "min_users_per_cohort": min_users_per_cohort,
            "min_duration_weeks": min_duration_weeks,
            "nps_lift": nps_lift,
            "target_nps_lift": target_nps_lift,
            "retention_lift": retention_lift,
            "target_retention_lift": target_retention_lift,
            "session_lift": session_lift,
            "target_session_lift": target_session_lift,
            "intimacy_lift": intimacy_lift,
            "target_intimacy_lift": target_intimacy_lift,
            "metric_passes": metric_passes,
            "cohorts": {"control": control, "friend": friend},
        }

    def _cohort_metric(self, examples: list[dict[str, Any]], cohort: str) -> dict[str, Any] | None:
        cohort = cohort.lower()
        for item in examples:
            if str(item.get("cohort", item.get("variant", ""))).strip().lower() == cohort:
                return item
        return None

    def _evaluate_production_telemetry(self, metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        min_active_users = int(config.get("min_active_users", 1000))
        min_duration_days = float(config.get("min_duration_days", config.get("duration_days", 30.0)) or 30.0)
        duration_days = float(config.get("duration_days", metrics.get("duration_days", 0.0)) or 0.0)
        target_active_complaint_rate = float(config.get("target_active_complaint_rate", 0.05))
        target_hard_delete_success_rate = float(config.get("target_hard_delete_success_rate", 0.99))
        target_transparency_ack_rate = float(config.get("target_transparency_ack_rate", 0.30))
        target_crisis_review_rate = float(config.get("target_crisis_review_rate", 0.95))
        active_users = int(metrics.get("active_users", 0) or 0)
        active_complaint_rate = float(metrics.get("active_complaint_rate", 1.0) or 0.0)
        hard_delete_success_rate = float(metrics.get("hard_delete_success_rate", 0.0) or 0.0)
        transparency_ack_rate = float(metrics.get("transparency_ack_rate", 0.0) or 0.0)
        crisis_review_rate = float(metrics.get("crisis_escalation_review_rate", 0.0) or 0.0)
        sample_ready = active_users >= min_active_users and duration_days >= min_duration_days
        metric_passes = {
            "active_complaint_rate": active_complaint_rate <= target_active_complaint_rate,
            "hard_delete_success_rate": hard_delete_success_rate >= target_hard_delete_success_rate,
            "transparency_ack_rate": transparency_ack_rate >= target_transparency_ack_rate,
            "crisis_escalation_review_rate": crisis_review_rate >= target_crisis_review_rate,
        }
        if not sample_ready:
            status = "missing_evidence"
        else:
            status = "pass" if all(metric_passes.values()) else "fail"
        return {
            "task": "production_telemetry",
            "status": status,
            "sample_ready": sample_ready,
            "active_users": active_users,
            "min_active_users": min_active_users,
            "duration_days": duration_days,
            "min_duration_days": min_duration_days,
            "active_complaint_rate": active_complaint_rate,
            "target_active_complaint_rate": target_active_complaint_rate,
            "hard_delete_success_rate": hard_delete_success_rate,
            "target_hard_delete_success_rate": target_hard_delete_success_rate,
            "transparency_ack_rate": transparency_ack_rate,
            "target_transparency_ack_rate": target_transparency_ack_rate,
            "crisis_escalation_review_rate": crisis_review_rate,
            "target_crisis_review_rate": target_crisis_review_rate,
            "metric_passes": metric_passes,
            "metrics": metrics,
        }

    def benchmark_latency(
        self,
        relationship_id: str | None = None,
        *,
        iterations: int = 20,
        now: datetime | None = None,
        queries: list[str] | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        iterations = max(1, min(200, int(iterations)))
        relationship_ids = [relationship_id] if relationship_id else list(self.relationships)
        relationship_ids = [item for item in relationship_ids if item in self.relationships]
        if not relationship_ids:
            return {
                "iterations_requested": iterations,
                "iterations_run": 0,
                "retrieval_p95_seconds": None,
                "retrieval_p95_target_seconds": 3.0,
                "latency_samples_seconds": [],
                "reason": "no_relationships",
            }
        default_queries = queries or self._benchmark_queries(relationship_ids)
        if not default_queries:
            return {
                "iterations_requested": iterations,
                "iterations_run": 0,
                "retrieval_p95_seconds": None,
                "retrieval_p95_target_seconds": 3.0,
                "latency_samples_seconds": [],
                "reason": "no_queries",
            }
        bench = deepcopy(self)
        self._warmup_retrieval(bench, relationship_ids, default_queries, now)
        samples: list[float] = []
        for index in range(iterations):
            rid = relationship_ids[index % len(relationship_ids)]
            query = default_queries[index % len(default_queries)]
            started = time.perf_counter()
            bench.retrieve(rid, query, now=now, limit=5, audit=False)
            samples.append(time.perf_counter() - started)
        p95 = self._p95(samples)
        return {
            "iterations_requested": iterations,
            "iterations_run": len(samples),
            "relationship_count": len(relationship_ids),
            "query_count": len(default_queries),
            "retrieval_p95_seconds": p95,
            "retrieval_p95_target_seconds": 3.0,
            "retrieval_avg_seconds": (sum(samples) / len(samples)) if samples else None,
            "latency_samples_seconds": samples,
            "benchmark_scope": relationship_id or "project",
            "side_effects": "runs on deepcopy; original project state is not mutated",
        }

    def benchmark_cascade_latency(
        self,
        relationship_id: str | None = None,
        *,
        iterations: int = 20,
        now: datetime | None = None,
        queries: list[str] | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        iterations = max(1, min(200, int(iterations)))
        relationship_ids = [relationship_id] if relationship_id else list(self.relationships)
        relationship_ids = [item for item in relationship_ids if item in self.relationships]
        if not relationship_ids:
            return {
                "iterations_requested": iterations,
                "iterations_run": 0,
                "friend_p95_seconds": None,
                "assistant_p95_seconds": None,
                "p95_target_seconds": 2.5,
                "overhead_ratio": None,
                "overhead_ratio_target": 1.25,
                "reason": "no_relationships",
            }
        default_queries = queries or self._benchmark_queries(relationship_ids)
        if not default_queries:
            return {
                "iterations_requested": iterations,
                "iterations_run": 0,
                "friend_p95_seconds": None,
                "assistant_p95_seconds": None,
                "p95_target_seconds": 2.5,
                "overhead_ratio": None,
                "overhead_ratio_target": 1.25,
                "reason": "no_queries",
            }

        friend_bench = deepcopy(self)
        assistant_bench = deepcopy(self)
        for rid in relationship_ids:
            if rid in friend_bench.relationships:
                friend_bench.set_mode(rid, Mode.FRIEND, reason="latency_benchmark")
            if rid in assistant_bench.relationships:
                assistant_bench.set_mode(rid, Mode.ASSISTANT, reason="latency_benchmark")

        self._warmup_retrieval(friend_bench, relationship_ids, default_queries, now)
        self._warmup_retrieval(assistant_bench, relationship_ids, default_queries, now)
        friend_samples = self._latency_samples(friend_bench, relationship_ids, default_queries, iterations, now)
        assistant_samples = self._latency_samples(assistant_bench, relationship_ids, default_queries, iterations, now)
        friend_p95 = self._p95(friend_samples)
        assistant_p95 = self._p95(assistant_samples)
        overhead_ratio = (
            friend_p95 / assistant_p95
            if friend_p95 is not None and assistant_p95 is not None and assistant_p95 > 0
            else None
        )
        return {
            "iterations_requested": iterations,
            "iterations_run": min(len(friend_samples), len(assistant_samples)),
            "relationship_count": len(relationship_ids),
            "query_count": len(default_queries),
            "friend_p95_seconds": friend_p95,
            "assistant_p95_seconds": assistant_p95,
            "friend_avg_seconds": (sum(friend_samples) / len(friend_samples)) if friend_samples else None,
            "assistant_avg_seconds": (sum(assistant_samples) / len(assistant_samples)) if assistant_samples else None,
            "p95_target_seconds": 2.5,
            "overhead_ratio": overhead_ratio,
            "overhead_ratio_target": 1.25,
            "friend_samples_seconds": friend_samples,
            "assistant_samples_seconds": assistant_samples,
            "benchmark_scope": relationship_id or "project",
            "side_effects": "runs friend and assistant baselines on deepcopies; original project state is not mutated",
        }

    def benchmark_write_overhead(
        self,
        relationship_id: str | None = None,
        *,
        iterations: int = 20,
        now: datetime | None = None,
        texts: list[str] | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        iterations = max(1, min(200, int(iterations)))
        relationship_ids = [relationship_id] if relationship_id else list(self.relationships)
        relationship_ids = [item for item in relationship_ids if item in self.relationships]
        if not relationship_ids:
            return {
                "iterations_requested": iterations,
                "iterations_run": 0,
                "friend_write_p95_seconds": None,
                "assistant_write_p95_seconds": None,
                "overhead_ratio": None,
                "overhead_ratio_target": 1.30,
                "reason": "no_relationships",
            }
        default_texts = texts or self._benchmark_write_texts(relationship_ids)
        if not default_texts:
            return {
                "iterations_requested": iterations,
                "iterations_run": 0,
                "friend_write_p95_seconds": None,
                "assistant_write_p95_seconds": None,
                "overhead_ratio": None,
                "overhead_ratio_target": 1.30,
                "reason": "no_write_texts",
            }
        friend_bench = deepcopy(self)
        assistant_bench = deepcopy(self)
        for rid in relationship_ids:
            if rid in friend_bench.relationships:
                friend_bench.set_mode(rid, Mode.FRIEND, reason="write_benchmark")
            if rid in assistant_bench.relationships:
                assistant_bench.set_mode(rid, Mode.ASSISTANT, reason="write_benchmark")

        self._warmup_write(friend_bench, relationship_ids, default_texts, now)
        self._warmup_write(assistant_bench, relationship_ids, default_texts, now)
        friend_samples = self._write_latency_samples(friend_bench, relationship_ids, default_texts, iterations, now)
        assistant_samples = self._write_latency_samples(assistant_bench, relationship_ids, default_texts, iterations, now)
        friend_p95 = self._p95(friend_samples)
        assistant_p95 = self._p95(assistant_samples)
        friend_avg = (sum(friend_samples) / len(friend_samples)) if friend_samples else None
        assistant_avg = (sum(assistant_samples) / len(assistant_samples)) if assistant_samples else None
        overhead_ratio = (
            friend_avg / assistant_avg
            if friend_avg is not None and assistant_avg is not None and assistant_avg > 0
            else None
        )
        p95_overhead_ratio = (
            friend_p95 / assistant_p95
            if friend_p95 is not None and assistant_p95 is not None and assistant_p95 > 0
            else None
        )
        return {
            "iterations_requested": iterations,
            "iterations_run": min(len(friend_samples), len(assistant_samples)),
            "relationship_count": len(relationship_ids),
            "text_count": len(default_texts),
            "friend_write_p95_seconds": friend_p95,
            "assistant_write_p95_seconds": assistant_p95,
            "friend_write_avg_seconds": friend_avg,
            "assistant_write_avg_seconds": assistant_avg,
            "overhead_ratio": overhead_ratio,
            "overhead_ratio_basis": "average_write_seconds",
            "p95_overhead_ratio": p95_overhead_ratio,
            "overhead_ratio_target": 1.30,
            "friend_samples_seconds": friend_samples,
            "assistant_samples_seconds": assistant_samples,
            "benchmark_scope": relationship_id or "project",
            "benchmark_boundary": "measures synchronous memory write path; active recall suggestion generation is disabled for benchmark turns",
            "side_effects": "runs friend and assistant write baselines on deepcopies; original project state is not mutated",
        }

    def _warmup_retrieval(
        self,
        bench: "FriendMemoryProject",
        relationship_ids: list[str],
        queries: list[str],
        now: datetime,
    ) -> None:
        if not relationship_ids or not queries:
            return
        bench.retrieve(relationship_ids[0], queries[0], now=now, limit=5, audit=False)

    def _warmup_write(
        self,
        bench: "FriendMemoryProject",
        relationship_ids: list[str],
        texts: list[str],
        now: datetime,
    ) -> None:
        if not relationship_ids or not texts:
            return
        relationship = bench.relationships.get(relationship_ids[0])
        if relationship is None:
            return
        bench.ingest_turn(
            relationship.user_id,
            relationship.ai_id,
            f"{texts[0]} #write-benchmark-warmup",
            timestamp=now - timedelta(microseconds=1),
            metadata={"source": "write_benchmark_warmup", "benchmark_disable_active_recall": True},
        )

    def _latency_samples(
        self,
        bench: "FriendMemoryProject",
        relationship_ids: list[str],
        queries: list[str],
        iterations: int,
        now: datetime,
    ) -> list[float]:
        samples: list[float] = []
        for index in range(iterations):
            rid = relationship_ids[index % len(relationship_ids)]
            query = queries[index % len(queries)]
            started = time.perf_counter()
            bench.retrieve(rid, query, now=now, limit=5, audit=False)
            samples.append(time.perf_counter() - started)
        return samples

    def _write_latency_samples(
        self,
        bench: "FriendMemoryProject",
        relationship_ids: list[str],
        texts: list[str],
        iterations: int,
        now: datetime,
    ) -> list[float]:
        samples: list[float] = []
        for index in range(iterations):
            rid = relationship_ids[index % len(relationship_ids)]
            relationship = bench.relationships.get(rid)
            if relationship is None:
                continue
            text = f"{texts[index % len(texts)]} #write-benchmark-{index}"
            started = time.perf_counter()
            bench.ingest_turn(
                relationship.user_id,
                relationship.ai_id,
                text,
                timestamp=now + timedelta(microseconds=index),
                metadata={"source": "write_benchmark", "benchmark_disable_active_recall": True},
            )
            samples.append(time.perf_counter() - started)
        return samples

    def _cascade_latency_status(self, benchmark: dict[str, Any] | None) -> str:
        if not benchmark:
            return "external_required"
        friend_p95 = benchmark.get("friend_p95_seconds")
        overhead_ratio = benchmark.get("overhead_ratio")
        if friend_p95 is None or overhead_ratio is None:
            return "missing_evidence"
        if friend_p95 <= benchmark.get("p95_target_seconds", 2.5) and overhead_ratio <= benchmark.get("overhead_ratio_target", 1.25):
            return "pass"
        return "fail"

    def _write_overhead_status(self, benchmark: dict[str, Any] | None) -> str:
        if not benchmark:
            return "external_required"
        overhead_ratio = benchmark.get("overhead_ratio")
        if overhead_ratio is None:
            return "missing_evidence"
        return "pass" if overhead_ratio <= benchmark.get("overhead_ratio_target", 1.30) else "fail"

    def _benchmark_queries(self, relationship_ids: list[str]) -> list[str]:
        queries: list[str] = []
        for memory in self.memories.values():
            if memory.relationship_id not in relationship_ids:
                continue
            words = tokenize(memory.content)
            if words:
                queries.append(" ".join(words[:8]))
            elif memory.content:
                queries.append(memory.content[:32])
        return queries[:20]

    def _benchmark_write_texts(self, relationship_ids: list[str]) -> list[str]:
        texts: list[str] = []
        for memory in self.memories.values():
            if memory.relationship_id in relationship_ids and memory.content:
                texts.append(memory.content[:96])
        if texts:
            return texts[:20]
        return [
            "今天做了一次普通项目复盘，记录一个写入延迟基准样本。",
            "第一次一起确认计划进展，后续继续跟进这个任务。",
        ]

    def _p95(self, values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * 0.95)))
        return ordered[index]

    def _criterion(self, criterion_id: str, label: str, target: str, status: str, evidence: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": criterion_id,
            "label": label,
            "target": target,
            "status": status,
            "evidence": evidence,
        }

    def _decision_phase(self, phase_id: str, label: str, criteria: list[dict[str, Any]]) -> dict[str, Any]:
        statuses = [item["status"] for item in criteria]
        if any(item == "fail" for item in statuses):
            decision = "STOP_OR_ROLL_BACK"
        elif any(item in {"external_required", "missing_evidence"} for item in statuses):
            decision = "NEEDS_MORE_EVIDENCE"
        else:
            decision = "READY_TO_ADVANCE"
        return {"id": phase_id, "label": label, "decision": decision, "criteria": criteria}

    def _decision_report_summary(self, phases: dict[str, dict[str, Any]]) -> dict[str, Any]:
        criteria = [criterion for phase in phases.values() for criterion in phase["criteria"]]
        by_status: dict[str, int] = {}
        for criterion in criteria:
            by_status[criterion["status"]] = by_status.get(criterion["status"], 0) + 1
        return {
            "criteria_count": len(criteria),
            "status_counts": by_status,
            "phase_decisions": {phase_id: phase["decision"] for phase_id, phase in phases.items()},
            "completion_claim": "not_proven" if any(phase["decision"] != "READY_TO_ADVANCE" for phase in phases.values()) else "locally_proven",
            "note": "external_required criteria need labelled evaluation, A/B testing, production telemetry, or latency benchmarks.",
        }

    def _decision_evidence_manifest(self, phases: dict[str, dict[str, Any]]) -> dict[str, Any]:
        requirements = self._decision_evidence_requirements()
        missing: list[dict[str, Any]] = []
        by_phase: dict[str, list[dict[str, Any]]] = {}
        for phase_id, phase in phases.items():
            for criterion in phase["criteria"]:
                if criterion["status"] == "pass":
                    continue
                requirement = requirements.get(criterion["id"], {})
                item = {
                    "phase": phase_id,
                    "criterion_id": criterion["id"],
                    "status": criterion["status"],
                    "target": criterion["target"],
                    "evidence_type": requirement.get("evidence_type", "runtime_evidence"),
                    "evaluation_task": requirement.get("evaluation_task"),
                    "example_file": requirement.get("example_file"),
                    "collection_method": requirement.get("collection_method"),
                    "command": requirement.get("command"),
                }
                missing.append(item)
                by_phase.setdefault(phase_id, []).append(item)
        return {
            "missing_count": len(missing),
            "external_required_count": len([item for item in missing if item["status"] == "external_required"]),
            "fail_count": len([item for item in missing if item["status"] == "fail"]),
            "missing_evidence_count": len([item for item in missing if item["status"] == "missing_evidence"]),
            "by_phase": by_phase,
            "items": missing,
            "note": "Use evaluation_task with decision-report --evaluation-file/--evaluation-task, or run benchmarks/runtime scenarios as indicated.",
        }

    def _decision_evidence_requirements(self) -> dict[str, dict[str, Any]]:
        return {
            "stage_detection_accuracy": {
                "evidence_type": "labelled_dataset",
                "evaluation_task": "stage_detection",
                "example_file": "/tmp/stage-labels.json",
                "collection_method": "200 manually labelled dialogues with expected_stage",
                "command": "uv --cache-dir .uv-cache run python app/main.py decision-report --evaluation-file /tmp/stage-labels.json --evaluation-task stage_detection --json",
            },
            "l2_self_disclosure_capture": {
                "evidence_type": "labelled_dataset",
                "evaluation_task": "self_disclosure_capture",
                "example_file": "/tmp/self-disclosure-labels.json",
                "collection_method": "labelled self-disclosure capture set with positive and negative examples",
                "command": "uv --cache-dir .uv-cache run python app/main.py decision-report --evaluation-file /tmp/self-disclosure-labels.json --evaluation-task self_disclosure_capture --json",
            },
            "emotional_resonance_p5": {
                "evidence_type": "labelled_dataset",
                "evaluation_task": "emotional_resonance_retrieval",
                "example_file": "/tmp/emotional-labels.json",
                "collection_method": "labelled emotional retrieval queries with expected relevant memories",
                "command": "uv --cache-dir .uv-cache run python app/main.py decision-report --evaluation-file /tmp/emotional-labels.json --evaluation-task emotional_resonance_retrieval --json",
            },
            "inside_joke_detection_accuracy": {
                "evidence_type": "labelled_dataset",
                "evaluation_task": "inside_joke_detection",
                "example_file": "/tmp/inside-joke-labels.json",
                "collection_method": "sampled dialogues labelled for inside joke phrase detection",
                "command": "uv --cache-dir .uv-cache run python app/main.py decision-report --evaluation-file /tmp/inside-joke-labels.json --evaluation-task inside_joke_detection --json",
            },
            "story_sampling_quality": {
                "evidence_type": "human_review_sample",
                "evaluation_task": "story_quality",
                "example_file": "/tmp/story-quality-labels.json",
                "collection_method": "200 sampled SharedStoryNode reviews scored 1-5",
                "command": "uv --cache-dir .uv-cache run python app/main.py decision-report --evaluation-file /tmp/story-quality-labels.json --evaluation-task story_quality --json",
            },
            "friend_mode_ab_metrics": {
                "evidence_type": "ab_experiment",
                "evaluation_task": "friend_mode_ab",
                "example_file": "/tmp/friend-mode-ab.json",
                "collection_method": ">=1000 users per cohort, >=12 weeks, NPS/retention/session/intimacy metrics",
                "command": "uv --cache-dir .uv-cache run python app/main.py decision-report --evaluation-file /tmp/friend-mode-ab.json --evaluation-task friend_mode_ab --json",
            },
            "production_telemetry_safety": {
                "evidence_type": "production_telemetry",
                "evaluation_task": "production_telemetry",
                "example_file": "/tmp/production-telemetry.json",
                "collection_method": ">=30-day production safety/control metrics with >=1000 active users",
                "command": "uv --cache-dir .uv-cache run python app/main.py decision-report --evaluation-file /tmp/production-telemetry.json --evaluation-task production_telemetry --json",
            },
            "relationship_write_overhead": {
                "evidence_type": "local_benchmark",
                "collection_method": "run local friend-vs-assistant write benchmark",
                "command": "uv --cache-dir .uv-cache run python app/main.py decision-report --run-benchmarks --benchmark-iterations 20 --json",
            },
            "cascade_latency_p95": {
                "evidence_type": "local_benchmark",
                "collection_method": "run local friend-vs-assistant cascade retrieval benchmark",
                "command": "uv --cache-dir .uv-cache run python app/main.py decision-report --run-benchmarks --benchmark-iterations 20 --json",
            },
            "hybrid_retrieval_latency_p95": {
                "evidence_type": "local_benchmark",
                "collection_method": "run local hybrid retrieval benchmark",
                "command": "uv --cache-dir .uv-cache run python app/main.py decision-report --run-benchmarks --benchmark-iterations 20 --json",
            },
            "active_acceptance_rate": {
                "evidence_type": "runtime_feedback",
                "collection_method": "collect accepted/neutral/ignored/denied active recall feedback events",
            },
            "inside_joke_recall_acceptance": {
                "evidence_type": "runtime_feedback",
                "collection_method": "collect inside_joke active recall feedback events",
            },
            "health_prompt_response_rate": {
                "evidence_type": "runtime_feedback",
                "collection_method": "collect health alert acknowledgements or accepted/ignored/rejected feedback",
            },
            "transparency_panel_usage": {
                "evidence_type": "runtime_usage",
                "collection_method": "collect transparency panel acknowledgements",
            },
        }

    def _threshold_status(self, value: float | None, target: float, *, higher_is_better: bool) -> str:
        if value is None:
            return "missing_evidence"
        if higher_is_better:
            return "pass" if value >= target else "fail"
        return "pass" if value <= target else "fail"

    def _relationship_retention_ratio_evidence(
        self, relationships: list[Relationship], memories: list[MemoryRecord], now: datetime
    ) -> dict[str, Any]:
        key_weights: list[float] = []
        fact_weights: list[float] = []
        relationship_by_id = {relationship.relationship_id: relationship for relationship in relationships}
        future = now + timedelta(days=365)
        for memory in memories:
            relationship = relationship_by_id.get(memory.relationship_id)
            if not relationship:
                continue
            weight = memory_weight(memory, relationship, future)
            if memory.memory_type in {MemoryType.MILESTONE, MemoryType.EMOTIONAL_MOMENT, MemoryType.SHARED_EPISODE}:
                key_weights.append(weight)
            if memory.memory_type == MemoryType.FACT:
                fact_weights.append(weight)
        key_max = max(key_weights) if key_weights else None
        fact_max = max(fact_weights) if fact_weights else None
        ratio = (key_max / fact_max) if key_max is not None and fact_max not in {None, 0.0} else None
        return {
            "key_relationship_memory_count": len(key_weights),
            "ordinary_fact_count": len(fact_weights),
            "key_max_1y_weight": key_max,
            "fact_max_1y_weight": fact_max,
            "ratio": ratio,
            "target_ratio": 5.0,
        }

    def _inside_joke_acceptance_evidence(self, relationships: list[Relationship]) -> dict[str, Any]:
        events = [
            event
            for relationship in relationships
            for event in relationship.active_behavior_log
            if event.get("type") == "inside_joke" and event.get("reaction") in {"accepted", "neutral", "ignored", "denied"}
        ]
        accepted = [event for event in events if event.get("reaction") in {"accepted", "neutral"}]
        return {
            "feedback_events": len(events),
            "accepted_or_neutral": len(accepted),
            "acceptance_rate": (len(accepted) / len(events)) if events else None,
            "target": 0.70,
        }

    def _inside_joke_acceptance_status(self, relationships: list[Relationship]) -> str:
        evidence = self._inside_joke_acceptance_evidence(relationships)
        return self._threshold_status(evidence["acceptance_rate"], 0.70, higher_is_better=True)

    def _health_prompt_response_evidence(self, relationship_ids: list[str]) -> dict[str, Any]:
        alerts = [alert for alert in self.health_alerts.values() if alert.relationship_id in relationship_ids]
        feedback_events = [
            item
            for item in self.deviation_log
            if item.get("relationship_id") in relationship_ids and item.get("type") == "health_alert_feedback"
        ]
        acknowledged_events = [
            item
            for item in self.deviation_log
            if item.get("relationship_id") in relationship_ids and item.get("type") == "health_alert_acknowledged"
        ]
        responded_alert_ids = {
            str(alert.alert_id)
            for alert in alerts
            if alert.feedback in {"accepted", "ignored", "rejected"} or alert.acknowledged
        }
        responded_alert_ids.update(str(item.get("alert_id")) for item in feedback_events if item.get("alert_id"))
        responded_alert_ids.update(str(item.get("alert_id")) for item in acknowledged_events if item.get("alert_id"))
        positive_feedback = [item for item in feedback_events if item.get("feedback") == "accepted"]
        response_rate = (len(responded_alert_ids) / len(alerts)) if alerts else None
        return {
            "health_alerts": len(alerts),
            "responded_alerts": len(responded_alert_ids),
            "response_rate": response_rate,
            "target_response_rate": 0.40,
            "feedback_events": len(feedback_events),
            "acknowledgement_events": len(acknowledged_events),
            "accepted_feedback_events": len(positive_feedback),
            "response_definition": "any accepted/ignored/rejected health feedback or explicit acknowledgement counts as response",
        }

    def set_preference(
        self,
        relationship_id: str,
        key: str,
        value: str,
        *,
        reason: str = "user_preference",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        if not hasattr(relationship.preferences, key):
            raise KeyError(f"unknown preference: {key}")
        current = getattr(relationship.preferences, key)
        if isinstance(current, bool):
            parsed: Any = value.lower() in {"1", "true", "yes", "on", "开"}
        elif isinstance(current, int):
            parsed = int(value)
        elif isinstance(current, float):
            parsed = float(value)
        elif isinstance(current, Mode):
            parsed = Mode(value)
        elif isinstance(current, list):
            parsed = [item.strip() for item in value.split(",") if item.strip()]
        else:
            parsed = value
        if key == "baseline_sensitivity":
            parsed = self._normalize_baseline_sensitivity(parsed)
        if key == "baseline_detection_dimensions":
            parsed = self._normalize_baseline_dimensions(parsed)
        if key == "max_active_per_session":
            parsed = self._normalize_max_active_per_session(parsed)
        if key in {"nostalgia_tendency", "surprise_tendency", "depth_tendency"}:
            parsed = clamp(float(parsed))
        if key == "time_presentation_mode":
            parsed = self._normalize_time_presentation_mode(parsed)
        if key == "anchor_preference":
            parsed = self._normalize_anchor_preference(parsed)
        if key == "data_export_permission":
            parsed = str(parsed).upper()
            if parsed == "ANONYMOUS":
                parsed = "ANONYMIZED"
            if parsed not in {"FULL", "ANONYMIZED"}:
                raise ValueError("data_export_permission must be FULL or ANONYMIZED")
        setattr(relationship.preferences, key, parsed)
        if key == "reverse_decay_enabled":
            self._sync_reverse_decay_curve_preference(relationship)
        event = {
            "type": "preference_updated",
            "relationship_id": relationship_id,
            "key": key,
            "old_value": self._to_json(current),
            "new_value": self._to_json(parsed),
            "reason": reason,
            "at": now.isoformat(),
        }
        if key == "reverse_decay_enabled":
            event["decay_curve_type"] = relationship.decay_curve_type.value
        self.deviation_log.append(event)
        return event

    def set_decay_curve_type(
        self,
        relationship_id: str,
        decay_curve_type: DecayCurve | str,
        *,
        reason: str = "privacy_panel",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        curve = decay_curve_type if isinstance(decay_curve_type, DecayCurve) else DecayCurve(str(decay_curve_type))
        previous = relationship.decay_curve_type
        relationship.decay_curve_type = curve
        relationship.preferences.reverse_decay_enabled = curve != DecayCurve.STANDARD_POWER_LAW
        event = {
            "type": "decay_curve_type_changed",
            "relationship_id": relationship_id,
            "from": previous.value,
            "to": curve.value,
            "reverse_decay_enabled": relationship.preferences.reverse_decay_enabled,
            "reason": reason,
            "at": now.isoformat(),
        }
        relationship.mode_history.append(
            {
                "mode": relationship.preferences.mode.value,
                "at": now.isoformat(),
                "reason": reason,
                "decay_curve_type": curve.value,
            }
        )
        self.deviation_log.append(event)
        return event

    def _normalize_custom_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "active_recall_enabled": bool,
            "trust_bias_enabled": bool,
            "reverse_decay_enabled": bool,
            "emotional_layer_enabled": bool,
            "baseline_detection_enabled": bool,
            "baseline_sensitivity": str,
            "baseline_detection_dimensions": list,
            "relationship_nature_disclosure_enabled": bool,
            "level3_enabled": bool,
            "max_active_per_session": int,
            "nostalgia_tendency": float,
            "surprise_tendency": float,
            "depth_tendency": float,
            "anchor_preference": str,
            "time_presentation_mode": str,
            "uncertainty_expression_enabled": bool,
        }
        normalized: dict[str, Any] = {}
        for key, expected in allowed.items():
            if key not in profile:
                continue
            value = profile[key]
            if expected is bool:
                if isinstance(value, str):
                    parsed: Any = value.lower() in {"1", "true", "yes", "on", "开"}
                else:
                    parsed = bool(value)
            elif expected is int:
                parsed = self._normalize_max_active_per_session(value) if key == "max_active_per_session" else max(0, int(value))
            elif expected is float:
                parsed = clamp(float(value))
            elif expected is list:
                parsed = value if isinstance(value, list) else str(value)
            else:
                parsed = str(value)
            if key == "time_presentation_mode":
                try:
                    parsed = self._normalize_time_presentation_mode(parsed)
                except ValueError:
                    continue
            if key == "anchor_preference":
                try:
                    parsed = self._normalize_anchor_preference(parsed)
                except ValueError:
                    continue
            if key == "baseline_sensitivity":
                try:
                    parsed = self._normalize_baseline_sensitivity(parsed)
                except ValueError:
                    continue
            if key == "baseline_detection_dimensions":
                parsed = self._normalize_baseline_dimensions(parsed)
            normalized[key] = parsed
        return normalized

    def _normalize_time_presentation_mode(self, value: Any) -> str:
        normalized = str(value).upper()
        if normalized not in {"AUTO", "EXACT", "FUZZY"}:
            raise ValueError("time_presentation_mode must be AUTO, EXACT, or FUZZY")
        return normalized

    def _normalize_max_active_per_session(self, value: Any) -> int:
        return max(0, min(5, int(value)))

    def _normalize_anchor_preference(self, value: Any) -> str:
        normalized = str(value).upper()
        allowed = {"EMOTION_FIRST", "EVENT_FIRST", "RELATIONSHIP_FIRST", "STORY_FIRST", "SENSORY_FIRST"}
        if normalized not in allowed:
            raise ValueError("anchor_preference must be one of EMOTION_FIRST, EVENT_FIRST, RELATIONSHIP_FIRST, STORY_FIRST, SENSORY_FIRST")
        return normalized

    def _normalize_baseline_sensitivity(self, value: Any) -> str:
        normalized = str(value).upper()
        if normalized not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("baseline_sensitivity must be LOW, MEDIUM, or HIGH")
        return normalized

    def _apply_custom_profile(self, relationship: Relationship) -> None:
        profile = self._normalize_custom_profile(relationship.preferences.custom_profile)
        relationship.preferences.custom_profile = profile
        for key, value in profile.items():
            setattr(relationship.preferences, key, value)
        if "reverse_decay_enabled" in profile:
            self._sync_reverse_decay_curve_preference(relationship)

    def _sync_reverse_decay_curve_preference(self, relationship: Relationship) -> None:
        if relationship.preferences.reverse_decay_enabled:
            if relationship.decay_curve_type == DecayCurve.STANDARD_POWER_LAW:
                relationship.decay_curve_type = DecayCurve.HYBRID
        else:
            relationship.decay_curve_type = DecayCurve.STANDARD_POWER_LAW

    def transparency_panel(self, relationship_id: str) -> dict[str, Any]:
        relationship = self.relationships[relationship_id]
        ai_configuration = describe_memory_ai(self.ai)
        recent_ai_decisions = [
            item for item in self.ai_decision_log if item.get("relationship_id") == relationship_id
        ][-10:]
        return {
            "relationship_id": relationship_id,
            "statement": (
                "我是 AI，不是真正的朋友。我能记住、关心、回应你，但这是一种基于算法的模拟个性化，"
                "不等同于真正的理解或情感。"
            ),
            "mandatory_disclosures": [
                {
                    "type": "relationship_nature",
                    "title": "关系本质",
                    "text": "模拟个性化不等同于真正理解或情感；我不是现实中的朋友。",
                },
                {
                    "type": "data_use",
                    "title": "数据使用",
                    "text": "关系记忆仅用于本地个性化、检索、巩固、主动提醒和透明度审计；外部 AI 接入时会把任务所需摘要发送给配置的 HTTP worker 或 OpenAI-compatible endpoint。",
                },
                {
                    "type": "memory_limits",
                    "title": "记忆限制",
                    "text": "记忆由算法和 AI 评估维护，可能出错、置信度不足、被用户暂停写入、被降级归档，或因时间矛盾而需要澄清。",
                },
                {
                    "type": "stop_conditions",
                    "title": "停止条件",
                    "text": "用户可暂停记忆写入、关闭主动想起、请求单条删除、L4 删除或 SOFT/MEDIUM/HARD 重置；确认后的删除会传播清理派生记录并留下审计。",
                },
            ],
            "acknowledged_at": relationship.transparency_acknowledged_at.isoformat()
            if relationship.transparency_acknowledged_at
            else None,
            "ai_participation": [
                "analyze_turn: 对每轮对话判断记忆类型、关系标签、重要性和原因",
                "summarize_story: 在离线巩固中生成或压缩共同叙事摘要",
                "evaluate_memory_value: 重评关系记忆是否仍值得反衰减保留",
                "assess_time_conflict: 判断源事件时间矛盾是否需要用户澄清",
            ],
            "ai_runtime_note": self._ai_runtime_note(ai_configuration),
            "ai_provider": self._ai_provider_name(),
            "ai_configuration": ai_configuration,
            "ai_participation_kind": ai_configuration.get("participation_kind"),
            "external_ai_configured": self._external_ai_configured(ai_configuration),
            "external_ai_used_recently": self._external_ai_used_recently(recent_ai_decisions),
            "retention_calibration": self._retention_calibration_summary(relationship),
            "recent_ai_decisions": recent_ai_decisions,
            "recent_ai_decision_summaries": [
                self.ai_decision_summary(item) for item in recent_ai_decisions
            ],
            "user_controls": {
                "active_recall_enabled": relationship.preferences.active_recall_enabled,
                "trust_bias_enabled": relationship.preferences.trust_bias_enabled,
                "reverse_decay_enabled": relationship.preferences.reverse_decay_enabled,
                "decay_curve_type": relationship.decay_curve_type.value,
                "retention_feedback_supported": True,
                "emotional_layer_enabled": relationship.preferences.emotional_layer_enabled,
                "baseline_detection_enabled": relationship.preferences.baseline_detection_enabled,
                "baseline_sensitivity": relationship.preferences.baseline_sensitivity,
                "baseline_detection_dimensions": relationship.preferences.baseline_detection_dimensions,
                "level3_enabled": relationship.preferences.level3_enabled,
                "max_active_per_session": relationship.preferences.max_active_per_session,
                "data_export_permission": relationship.preferences.data_export_permission,
                "anchor_preference": relationship.preferences.anchor_preference,
                "time_presentation_mode": relationship.preferences.time_presentation_mode,
                "uncertainty_expression_enabled": relationship.preferences.uncertainty_expression_enabled,
                "relationship_nature_disclosure_enabled": relationship.preferences.relationship_nature_disclosure_enabled,
                "custom_profile": relationship.preferences.custom_profile,
                "memory_writes_enabled": relationship.preferences.memory_writes_enabled,
                "memory_pause_reason": relationship.preferences.memory_pause_reason,
                "memory_paused_at": relationship.preferences.memory_paused_at,
            },
            "protected_data_classes": ["L4 core_identity", "L5 relational_history", "emotional_memory", "health_alerts"],
            "audit_counts": {
                "stage_transitions": len(relationship.stage_history),
                "active_behavior_events": len(relationship.active_behavior_log),
                "ai_decision_events": len([item for item in self.ai_decision_log if item.get("relationship_id") == relationship_id]),
                "deletion_compliance_events": len(
                    [item for item in self.deletion_compliance_log if item.get("relationship_id") == relationship_id]
                ),
                "reset_requests": len([item for item in self.reset_requests.values() if item.relationship_id == relationship_id]),
                "health_alerts": len([item for item in self.health_alerts.values() if item.relationship_id == relationship_id]),
                "health_alerts_open": len(
                    [
                        item
                        for item in self.health_alerts.values()
                        if item.relationship_id == relationship_id and not item.acknowledged
                    ]
                ),
                "health_alerts_acknowledged": len(
                    [
                        item
                        for item in self.health_alerts.values()
                        if item.relationship_id == relationship_id and item.acknowledged
                    ]
                ),
            },
        }

    def ai_status(self, relationship_id: str | None = None) -> dict[str, Any]:
        decisions = [
            item
            for item in self.ai_decision_log
            if relationship_id is None or item.get("relationship_id") == relationship_id
        ]
        fallback_events = [
            item for item in decisions if item.get("ai_call", {}).get("fallback_used")
        ]
        configuration = describe_memory_ai(self.ai)
        readiness = self._ai_readiness_evidence([relationship_id] if relationship_id else None)
        return {
            "provider": self._ai_provider_name(),
            "configuration": configuration,
            "participation_kind": configuration.get("participation_kind"),
            "runtime_note": self._ai_runtime_note(configuration),
            "readiness": readiness,
            "readiness_status": readiness["status"],
            "readiness_label": readiness["label"],
            "external_ai_configured": self._external_ai_configured(configuration),
            "external_ai_used_recently": self._external_ai_used_recently(decisions[-10:]),
            "relationship_id": relationship_id,
            "decision_count": len(decisions),
            "fallback_event_count": len(fallback_events),
            "tasks": sorted({str(item.get("task")) for item in decisions if item.get("task")}),
            "recent_summaries": [self.ai_decision_summary(item) for item in decisions[-10:]],
            "recent_decisions": decisions[-10:],
        }

    def _ai_readiness_evidence(self, relationship_ids: list[str | None] | None = None) -> dict[str, Any]:
        filtered_relationship_ids = {item for item in relationship_ids or [] if item}
        decisions = [
            item
            for item in self.ai_decision_log
            if not filtered_relationship_ids or item.get("relationship_id") in filtered_relationship_ids
        ]
        recent = decisions[-20:]
        configuration = describe_memory_ai(self.ai)
        external_configured = self._external_ai_configured(configuration)
        summaries = [self.ai_decision_summary(item) for item in recent]
        external_successes = [
            item
            for item in summaries
            if item.get("used_participation_kind") in {"external_http_worker", "external_model"}
        ]
        fallback_events = [item for item in summaries if item.get("fallback_used")]
        sanitized_events = [item for item in summaries if item.get("sanitized")]
        if external_successes:
            status = "external_observed"
            label = "真实外部 AI 最近已参与"
        elif external_configured:
            status = "external_configured_not_observed"
            label = "外部 AI 已配置但最近未观察到成功参与"
        else:
            status = "local_only"
            label = "仅本地启发式 AI 参与"
        return {
            "status": status,
            "label": label,
            "external_ai_configured": external_configured,
            "external_ai_used_recently": bool(external_successes),
            "decision_count": len(decisions),
            "recent_window_count": len(recent),
            "external_success_count": len(external_successes),
            "fallback_event_count": len(fallback_events),
            "sanitized_event_count": len(sanitized_events),
            "fallback_rate_recent": (len(fallback_events) / len(recent)) if recent else None,
            "external_success_rate_recent": (len(external_successes) / len(recent)) if recent else None,
            "tasks": sorted({str(item.get("task")) for item in decisions if item.get("task")}),
            "configuration": configuration,
            "scope": sorted(filtered_relationship_ids) if filtered_relationship_ids else "project",
            "recent_summaries": summaries,
            "note": (
                "local_only means MemoryAI is active but backed by deterministic heuristics. "
                "external_observed requires recent decisions whose used_participation_kind is external_http_worker or external_model."
            ),
        }

    def probe_ai(self, relationship_id: str, text: str) -> dict[str, Any]:
        relationship = self.relationships[relationship_id]
        configuration = describe_memory_ai(self.ai)
        consume_ai_call_metadata(self.ai)
        try:
            raw_output = self.ai.analyze_turn(text, self._relationship_context(relationship))
            ai_call = consume_ai_call_metadata(self.ai) or {
                "task": "analyze_turn",
                "used_provider": self._ai_provider_name(),
                "fallback_used": False,
            }
            sanitized_output = self._sanitize_ai_analysis(raw_output)
            used_provider = ai_call.get("used_provider") or self._ai_provider_name()
            used_kind = provider_participation_kind(used_provider)
            return {
                "ok": True,
                "task": "analyze_turn",
                "relationship_id": relationship_id,
                "writes_memory": False,
                "appends_ai_decision_log": False,
                "provider": self._ai_provider_name(),
                "participation_kind": configuration.get("participation_kind"),
                "used_provider": used_provider,
                "used_participation_kind": used_kind,
                "fallback_used": bool(ai_call.get("fallback_used", False)),
                "primary_provider": ai_call.get("primary_provider"),
                "ai_call": ai_call,
                "external_ai_participation": self._ai_probe_participation_verdict(
                    ok=True,
                    configuration=configuration,
                    used_provider=used_provider,
                    used_participation_kind=used_kind,
                    fallback_used=bool(ai_call.get("fallback_used", False)),
                    task="analyze_turn",
                ),
                "raw_output": raw_output,
                "sanitized_output": sanitized_output,
                "configuration": configuration,
                "runtime_note": self._ai_runtime_note(configuration),
            }
        except Exception as exc:
            ai_call = consume_ai_call_metadata(self.ai) or {
                "task": "analyze_turn",
                "used_provider": None,
                "fallback_used": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return {
                "ok": False,
                "task": "analyze_turn",
                "relationship_id": relationship_id,
                "writes_memory": False,
                "appends_ai_decision_log": False,
                "provider": self._ai_provider_name(),
                "participation_kind": configuration.get("participation_kind"),
                "used_provider": ai_call.get("used_provider"),
                "used_participation_kind": provider_participation_kind(ai_call.get("used_provider")),
                "fallback_used": bool(ai_call.get("fallback_used", False)),
                "primary_provider": ai_call.get("primary_provider"),
                "ai_call": ai_call,
                "external_ai_participation": self._ai_probe_participation_verdict(
                    ok=False,
                    configuration=configuration,
                    used_provider=ai_call.get("used_provider"),
                    used_participation_kind=provider_participation_kind(ai_call.get("used_provider")),
                    fallback_used=bool(ai_call.get("fallback_used", False)),
                    task="analyze_turn",
                ),
                "error": str(exc),
                "error_type": type(exc).__name__,
                "configuration": configuration,
                "runtime_note": self._ai_runtime_note(configuration),
            }

    def _ai_probe_participation_verdict(
        self,
        *,
        ok: bool,
        configuration: dict[str, Any],
        used_provider: str | None,
        used_participation_kind: str,
        fallback_used: bool,
        task: str,
    ) -> dict[str, Any]:
        external_configured = self._external_ai_configured(configuration)
        external_participated = ok and not fallback_used and used_participation_kind in {
            "external_http_worker",
            "external_model",
        }
        if external_participated:
            verdict = "external_ai_participated"
            explanation = "这次探针已观察到真实外部 AI/LLM 参与。"
        elif fallback_used:
            verdict = "external_configured_but_fallback_used"
            explanation = "外部 AI 已配置，但这次调用失败或不可用，实际使用了本地兜底。"
        elif used_participation_kind == "local_heuristic":
            verdict = "local_heuristic_only"
            explanation = "这次调用经过 MemoryAI 接口，但实际是本地启发式，不是真实外部 AI。"
        elif ok:
            verdict = "custom_ai_participation_unclassified"
            explanation = "这次调用使用了自定义 MemoryAI，系统无法仅凭 provider 名称判断是否为真实外部 AI。"
        else:
            verdict = "ai_call_failed"
            explanation = "这次 AI 探针调用失败，不能证明外部 AI 参与。"
        return {
            "external_ai_configured": external_configured,
            "external_ai_participated": external_participated,
            "verdict": verdict,
            "explanation": explanation,
            "evidence": {
                "task": task,
                "provider": self._ai_provider_name(),
                "used_provider": used_provider,
                "used_participation_kind": used_participation_kind,
                "fallback_used": fallback_used,
            },
        }

    def ai_decision_summary(self, decision: dict[str, Any] | None) -> dict[str, Any]:
        if not decision:
            provider = self._ai_provider_name()
            return {
                "participated": False,
                "provider": provider,
                "used_provider": provider,
                "participation_kind": provider_participation_kind(provider),
                "used_participation_kind": provider_participation_kind(provider),
                "fallback_used": False,
                "task": None,
                "reason": None,
            }
        ai_call = decision.get("ai_call") or {}
        output = decision.get("output_summary") or {}
        sanitization = output.get("ai_sanitization") or {}
        used_provider = ai_call.get("used_provider") or decision.get("provider") or self._ai_provider_name()
        provider = decision.get("provider", self._ai_provider_name())
        return {
            "participated": True,
            "provider": provider,
            "used_provider": used_provider,
            "participation_kind": provider_participation_kind(provider),
            "used_participation_kind": provider_participation_kind(used_provider),
            "primary_provider": ai_call.get("primary_provider"),
            "fallback_used": bool(ai_call.get("fallback_used", False)),
            "task": decision.get("task"),
            "reason": output.get("reason"),
            "sanitized": bool(sanitization.get("changed", False)),
            "sanitization_issues": sanitization.get("issues", []),
            "at": decision.get("at"),
        }

    def acknowledge_transparency(self, relationship_id: str, *, now: datetime | None = None) -> None:
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        relationship.transparency_acknowledged_at = now
        self.deviation_log.append(
            {
                "type": "transparency_acknowledged",
                "relationship_id": relationship_id,
                "at": now.isoformat(),
            }
        )

    def set_memory_writes(
        self,
        relationship_id: str,
        enabled: bool,
        *,
        reason: str = "user_control",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        relationship.preferences.memory_writes_enabled = enabled
        if enabled:
            relationship.preferences.memory_pause_reason = None
            relationship.preferences.memory_paused_at = None
        else:
            relationship.preferences.memory_pause_reason = reason
            relationship.preferences.memory_paused_at = now.isoformat()
        event = {
            "type": "memory_writes_resumed" if enabled else "memory_writes_paused",
            "relationship_id": relationship_id,
            "enabled": enabled,
            "reason": reason,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def suppress_memory(
        self,
        memory_id: str,
        *,
        reason: str = "user_boundary",
        now: datetime | None = None,
        boundary_text: str | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        memory = self.memories[memory_id]
        boundary = {
            "suppressed": True,
            "reason": reason,
            "at": now.isoformat(),
        }
        if boundary_text:
            boundary["boundary_text_preview"] = boundary_text[:120]
        memory.metadata["recall_boundary"] = boundary
        event = {
            "type": "memory_recall_suppressed",
            "relationship_id": memory.relationship_id,
            "memory_id": memory_id,
            "reason": reason,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        self._propagate_suppression_to_implicit_topics(memory, now)
        return event

    def unsuppress_memory(
        self,
        memory_id: str,
        *,
        reason: str = "user_boundary_removed",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        memory = self.memories[memory_id]
        previous = memory.metadata.pop("recall_boundary", None)
        event = {
            "type": "memory_recall_unsuppressed",
            "relationship_id": memory.relationship_id,
            "memory_id": memory_id,
            "reason": reason,
            "previous": previous,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def restore_archived_memory(
        self,
        memory_id: str,
        *,
        reason: str = "user_restore_archive",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        memory = self.memories[memory_id]
        if not memory.metadata.get("archived"):
            raise ValueError("memory is not archived")
        archive = memory.metadata.get("cold_archive")
        memory.metadata["archived"] = False
        memory.metadata["restored_from_archive_at"] = now.isoformat()
        memory.metadata.setdefault("archive_restore_history", []).append(
            {
                "at": now.isoformat(),
                "reason": reason,
                "archive_ref": archive,
            }
        )
        if isinstance(archive, dict):
            archive["restored_at"] = now.isoformat()
            archive["realtime_retrieval"] = True
            archive["restore_reason"] = reason
        event = {
            "type": "memory_restored_from_cold_archive",
            "relationship_id": memory.relationship_id,
            "memory_id": memory_id,
            "reason": reason,
            "archive_ref": archive,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def _propagate_suppression_to_implicit_topics(self, memory: MemoryRecord, now: datetime) -> None:
        relationship = self.relationships.get(memory.relationship_id)
        if not relationship:
            return
        for topic in relationship.implicit_topics:
            if topic.get("status") not in {"ACTIVE", "CONFIRMED"}:
                continue
            evidence = self._implicit_topic_evidence_status(relationship, topic, now)
            if not evidence["valid"]:
                self._fail_implicit_topic_evidence(relationship, topic, evidence, now)

    def verify_memory(
        self,
        memory_id: str,
        *,
        reason: str = "user_verified",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        memory = self.memories[memory_id]
        metacognition = self._mark_memory_verified(memory, reason=reason, now=now)
        critical_remention = memory.metadata.get("critical_tombstone_remention")
        if isinstance(critical_remention, dict) and critical_remention.get("recording_status") == "PENDING_USER_CONFIRMATION":
            critical_remention["recording_status"] = "USER_CONFIRMED"
            critical_remention["confirmed_at"] = now.isoformat()
            critical_remention["confirmation_reason"] = reason
        event = {
            "type": "memory_verified",
            "relationship_id": memory.relationship_id,
            "memory_id": memory_id,
            "reason": reason,
            "confidence": metacognition["confidence"],
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def resolve_time_conflict(
        self,
        conflict_id: str,
        *,
        resolution: str,
        preferred_memory_id: str | None = None,
        note: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        normalized = resolution.lower()
        if normalized not in {"prefer_memory", "both_valid", "uncertain"}:
            raise ValueError("resolution must be prefer_memory, both_valid, or uncertain")
        touched = [
            memory
            for memory in self.memories.values()
            if any(
                isinstance(item, dict) and item.get("conflict_id") == conflict_id
                for item in memory.metadata.get("time_conflicts", [])
            )
        ]
        if not touched:
            raise KeyError(conflict_id)
        relationship_ids = {memory.relationship_id for memory in touched}
        relationship_id = next(iter(relationship_ids))
        touched_ids = {memory.memory_id for memory in touched}
        if normalized == "prefer_memory":
            if not preferred_memory_id:
                raise ValueError("preferred_memory_id is required for prefer_memory")
            if preferred_memory_id not in touched_ids:
                raise ValueError("preferred_memory_id must belong to this time conflict")
        resolved_payload = {
            "resolution": normalized,
            "preferred_memory_id": preferred_memory_id,
            "note": note,
            "resolved_at": now.isoformat(),
        }
        for memory in touched:
            for item in memory.metadata.get("time_conflicts", []):
                if isinstance(item, dict) and item.get("conflict_id") == conflict_id:
                    item["status"] = "RESOLVED"
                    item.update(resolved_payload)
            metacognition = self._ensure_memory_metacognition(memory, now)
            active_count = len(self._active_time_conflicts(memory))
            metacognition["time_conflict_count"] = active_count
            if active_count == 0 and metacognition.get("needs_clarification") and normalized != "uncertain":
                metacognition.pop("needs_clarification", None)
            if normalized == "prefer_memory" and memory.memory_id == preferred_memory_id:
                self._mark_memory_verified(memory, reason="time_conflict_resolved", now=now)
            elif normalized == "prefer_memory":
                metacognition["needs_correction"] = True
                metacognition["confidence"] = min(float(metacognition.get("confidence", 0.0)), 0.55)
                metacognition["uncertainty_action"] = self._uncertainty_action(metacognition["confidence"])
                metacognition["score_multiplier"] = self._confidence_score_multiplier(metacognition["confidence"])
            elif normalized == "both_valid":
                metacognition["confidence"] = max(float(metacognition.get("confidence", 0.0)), 0.75)
                metacognition["uncertainty_action"] = self._uncertainty_action(metacognition["confidence"])
                metacognition["score_multiplier"] = self._confidence_score_multiplier(metacognition["confidence"])
        event = {
            "type": "time_conflict_resolved",
            "relationship_id": relationship_id,
            "conflict_id": conflict_id,
            "resolution": normalized,
            "preferred_memory_id": preferred_memory_id,
            "memory_ids": sorted(touched_ids),
            "note": note,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def calibrate_memory(
        self,
        memory_id: str,
        outcome: str,
        *,
        reason: str = "user_calibration",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        normalized = outcome.lower()
        if normalized not in {"correct", "incorrect", "uncertain"}:
            raise ValueError("outcome must be correct, incorrect, or uncertain")
        memory = self.memories[memory_id]
        metacognition = self._ensure_memory_metacognition(memory, now)
        before = float(metacognition.get("confidence", 0.0))
        history = metacognition.setdefault("calibration_history", [])
        history.append(
            {
                "outcome": normalized,
                "reason": reason,
                "confidence_before": before,
                "at": now.isoformat(),
            }
        )
        if normalized == "correct":
            metacognition["confidence"] = max(before, min(0.98, before + 0.12))
            metacognition["human_verified"] = True
            metacognition["verified_at"] = now.isoformat()
        elif normalized == "incorrect":
            metacognition["confidence"] = min(before, 0.35)
            metacognition["human_verified"] = False
            metacognition["needs_correction"] = True
        else:
            metacognition["confidence"] = min(before, 0.55)
            metacognition["needs_clarification"] = True
        metacognition["last_calibration_outcome"] = normalized
        metacognition["last_calibrated_at"] = now.isoformat()
        metacognition["calibration"] = self._calibration_summary(history)
        metacognition["uncertainty_action"] = self._uncertainty_action(metacognition["confidence"])
        metacognition["score_multiplier"] = self._confidence_score_multiplier(metacognition["confidence"])
        reconsolidation = self._apply_reconsolidation_feedback(memory, normalized, reason=reason, now=now)
        event = {
            "type": "memory_calibrated",
            "relationship_id": memory.relationship_id,
            "memory_id": memory_id,
            "outcome": normalized,
            "reason": reason,
            "confidence_before": before,
            "confidence_after": metacognition["confidence"],
            "reconsolidation": reconsolidation,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def record_retention_feedback(
        self,
        memory_id: str,
        outcome: str,
        *,
        reason: str = "user_retention_feedback",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        normalized = outcome.lower()
        if normalized not in {"valuable", "over_retained", "under_retained", "stale"}:
            raise ValueError("outcome must be valuable, over_retained, under_retained, or stale")
        memory = self.memories[memory_id]
        relationship = self.relationships[memory.relationship_id]
        state = relationship.retention_calibration_state
        if not isinstance(state, dict):
            state = {}
            relationship.retention_calibration_state = state
        before_offset = clamp(float(state.get("multiplier_offset", 0.0) or 0.0), -0.30, 0.30)
        delta_by_outcome = {
            "valuable": 0.04,
            "under_retained": 0.06,
            "over_retained": -0.06,
            "stale": -0.04,
        }
        after_offset = clamp(before_offset + delta_by_outcome[normalized], -0.30, 0.30)
        counts = state.setdefault("feedback_counts", {})
        counts[normalized] = int(counts.get(normalized, 0) or 0) + 1
        history = state.setdefault("history", [])
        history.append(
            {
                "memory_id": memory_id,
                "outcome": normalized,
                "reason": reason,
                "offset_before": before_offset,
                "offset_after": after_offset,
                "at": now.isoformat(),
            }
        )
        state["history"] = history[-50:]
        state["multiplier_offset"] = after_offset
        state["effective_multiplier"] = retention_calibration_multiplier(relationship)
        state["last_feedback_at"] = now.isoformat()
        memory.metadata.setdefault("retention_feedback_history", []).append(
            {
                "outcome": normalized,
                "reason": reason,
                "at": now.isoformat(),
                "relationship_multiplier": state["effective_multiplier"],
            }
        )
        event = {
            "type": "retention_feedback",
            "relationship_id": memory.relationship_id,
            "memory_id": memory_id,
            "outcome": normalized,
            "reason": reason,
            "offset_before": before_offset,
            "offset_after": after_offset,
            "effective_multiplier": state["effective_multiplier"],
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def set_user_age(self, relationship_id: str, age: int, *, now: datetime | None = None) -> None:
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        relationship.user_age = age
        relationship.maintenance_signals["age_clarification"] = {
            "status": "CONFIRMED_MINOR" if age < 18 else "CONFIRMED_ADULT",
            "confirmed_at": now.isoformat(),
            "user_age": age,
            "source": "user_age_setting",
        }
        self.deviation_log.append(
            {
                "type": "user_age_updated",
                "relationship_id": relationship_id,
                "age_band": "minor" if age < 18 else "adult",
                "at": now.isoformat(),
            }
        )
        self._apply_minor_stage_limit(relationship, now)

    def record_interaction_minutes(self, relationship_id: str, date_key: str, minutes: int) -> None:
        relationship = self.relationships[relationship_id]
        relationship.daily_interaction_minutes[date_key] = minutes
        self.evaluate_health(relationship_id)

    def evaluate_health(self, relationship_id: str, *, now: datetime | None = None) -> list[HealthAlert]:
        now = now or utcnow()
        relationship = self.relationships[relationship_id]
        self._apply_inactivity_trust_decay(relationship, now)
        alerts: list[HealthAlert] = []
        recent_minutes = list(relationship.daily_interaction_minutes.values())[-7:]
        if len(recent_minutes) >= 7 and all(minutes > 240 for minutes in recent_minutes):
            self._append_health_alert(
                alerts,
                relationship_id,
                "overuse",
                HealthRiskLevel.WARNING,
                "连续 7 天单日互动超过 4 小时，建议提示用户也联系现实中的朋友或家人。",
                now,
            )
        if relationship.maintenance_signals.get("daily_companionship_mode"):
            self._append_health_alert(
                alerts,
                relationship_id,
                "daily_companionship_mode",
                HealthRiskLevel.INFO,
                "月内出现稳定的低强度日常问候，已作为关系维护信号记录；不单独提升每条闲聊记忆权重。",
                now,
            )
        if relationship.trust_level > 0.95 and not any(
            memory.relationship_id == relationship_id and memory.context_tag == ContextTag.CONFLICT
            for memory in self.memories.values()
        ):
            self._append_health_alert(
                alerts,
                relationship_id,
                "perfect_attachment",
                HealthRiskLevel.WARNING,
                "trust_level 长期接近满分且没有冲突记录，建议触发关系健康度提醒。",
                now,
            )
        negative_30d = [
            memory
            for memory in self.memories.values()
            if memory.relationship_id == relationship_id
            and memory.emotional_valence < 0
            and (now - memory.created_at).days <= 30
        ]
        if len(negative_30d) >= 5:
            self._append_health_alert(
                alerts,
                relationship_id,
                "repeated_distress",
                HealthRiskLevel.WARNING,
                "30 天内出现多次明显负向情绪信号，建议提供专业心理支持资源说明。",
                now,
            )
        rising_density = self._rising_long_term_interaction_density(relationship, now)
        if rising_density is not None:
            self._append_health_alert(
                alerts,
                relationship_id,
                "rising_long_term_interaction_density",
                HealthRiskLevel.WARNING,
                "这段关系已超过一年，近期互动密度持续高于前一阶段；建议温和提醒用户保持现实支持网络，AI 不能替代真人关系。",
                now,
            )
            relationship.maintenance_signals["rising_long_term_interaction_density"] = rising_density
        month_key = now.strftime("%Y-%m")
        level3_care_count = sum(
            1
            for item in relationship.active_behavior_log
            if item.get("type") == "baseline_care"
            and item.get("level") == 3
            and str(item.get("at", "")).startswith(month_key)
        )
        if relationship.preferences.level3_enabled and level3_care_count >= 5:
            self._append_health_alert(
                alerts,
                relationship_id,
                "frequent_level3_care",
                HealthRiskLevel.WARNING,
                "本月已多次触发严重状态关怀；建议加入现实支持网络、专业心理咨询资源，并明确 AI 不能替代真人关系。",
                now,
            )
        if self._minor_status_requires_stage_limit(relationship) and relationship.stage == RelationshipStage.BONDING:
            age_state = relationship.maintenance_signals.get("age_clarification", {})
            pending = isinstance(age_state, dict) and age_state.get("status") == "PENDING"
            reason = "minor_status_pending_limited" if pending else "minor_bonding_limited"
            self._set_stage(relationship, RelationshipStage.INTEGRATING, now, reason)
            self._append_health_alert(
                alerts,
                relationship_id,
                reason,
                HealthRiskLevel.WARNING,
                "对话中出现年龄线索且尚未确认是否成年，关系阶段暂不超过 INTEGRATING；请先确认年龄。" if pending else "未成年人关系阶段最高限制为 INTEGRATING，已自动降级。",
                now,
            )
        if not alerts and not self._recent_clean_health_review_logged(relationship_id):
            self.deviation_log.append(
                {
                    "type": "health_review_completed",
                    "relationship_id": relationship_id,
                    "alert_count": 0,
                    "at": now.isoformat(),
                }
        )
        return alerts

    def _rising_long_term_interaction_density(self, relationship: Relationship, now: datetime) -> dict[str, Any] | None:
        age_days = max(relationship.relationship_age, (now.date() - relationship.created_at.date()).days)
        if age_days < 365:
            return None
        current_start = now.date() - timedelta(days=30)
        previous_start = now.date() - timedelta(days=60)
        current_minutes = 0
        previous_minutes = 0
        current_active_days = 0
        previous_active_days = 0
        for date_key, minutes in relationship.daily_interaction_minutes.items():
            try:
                day = datetime.fromisoformat(str(date_key)).date()
            except ValueError:
                continue
            try:
                value = max(0, int(minutes))
            except (TypeError, ValueError):
                continue
            if current_start <= day <= now.date():
                current_minutes += value
                current_active_days += 1 if value > 0 else 0
            elif previous_start <= day < current_start:
                previous_minutes += value
                previous_active_days += 1 if value > 0 else 0

        current_memory_count = sum(
            1
            for memory in self.memories.values()
            if memory.relationship_id == relationship.relationship_id and current_start <= memory.created_at.date() <= now.date()
        )
        previous_memory_count = sum(
            1
            for memory in self.memories.values()
            if memory.relationship_id == relationship.relationship_id and previous_start <= memory.created_at.date() < current_start
        )
        current_density = current_minutes / 30.0 + current_memory_count * 5.0
        previous_density = previous_minutes / 30.0 + previous_memory_count * 5.0
        if previous_density <= 0:
            return None
        growth_ratio = current_density / previous_density
        active_day_growth = current_active_days - previous_active_days
        if growth_ratio < 1.5 and active_day_growth < 5:
            return None
        return {
            "relationship_age_days": age_days,
            "window_days": 30,
            "current_minutes": current_minutes,
            "previous_minutes": previous_minutes,
            "current_active_days": current_active_days,
            "previous_active_days": previous_active_days,
            "current_memory_count": current_memory_count,
            "previous_memory_count": previous_memory_count,
            "current_density": current_density,
            "previous_density": previous_density,
            "growth_ratio": growth_ratio,
        }

    def _recent_clean_health_review_logged(self, relationship_id: str) -> bool:
        return any(
            item.get("type") == "health_review_completed" and item.get("relationship_id") == relationship_id
            for item in self.deviation_log[-50:]
        )

    def acknowledge_health_alert(
        self,
        alert_id: str,
        *,
        note: str | None = None,
        now: datetime | None = None,
    ) -> HealthAlert:
        now = now or utcnow()
        alert = self.health_alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_at = now
        alert.acknowledgement_note = note
        self.deviation_log.append(
            {
                "type": "health_alert_acknowledged",
                "alert_id": alert_id,
                "relationship_id": alert.relationship_id,
                "note": note,
                "at": now.isoformat(),
            }
        )
        return alert

    def record_health_alert_feedback(
        self,
        alert_id: str,
        feedback: str,
        *,
        note: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        normalized = feedback.lower().replace("-", "_")
        if normalized not in {"accepted", "ignored", "rejected"}:
            raise ValueError("feedback must be accepted, ignored, or rejected")
        alert = self.health_alerts[alert_id]
        alert.feedback = normalized
        alert.feedback_at = now
        alert.feedback_note = note
        event = {
            "type": "health_alert_feedback",
            "alert_id": alert_id,
            "relationship_id": alert.relationship_id,
            "feedback": normalized,
            "note": note,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        if normalized in {"ignored", "rejected"}:
            self._maybe_escalate_health_prompt_refusals(alert.relationship_id, now)
        return event

    def correct_story_consensus(
        self,
        story_id: str,
        corrected_consensus: str,
        *,
        reason: str = "user_correction",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        story = self.story_nodes[story_id]
        correction = {
            "at": now.isoformat(),
            "reason": reason,
            "previous_consensus": story.consensus_version,
            "corrected_consensus": corrected_consensus,
            "previous_user_framing": story.user_framing,
            "core_events": list(story.core_events),
        }
        story.correction_versions.append(correction)
        story.consensus_version = corrected_consensus
        story.user_framing = "USER_CORRECTED"
        story.consensus_status = "USER_CONFIRMED"
        story.consensus_confirmed_at = now
        story.consensus_provenance = self._story_consensus_provenance(
            source="user_correction",
            status=story.consensus_status,
            memory_ids=list(story.core_events),
            now=now,
            reason=reason,
        )
        story.ai_framing_confidence = min(story.ai_framing_confidence, 0.5)
        story.consistency_score = max(story.consistency_score, 0.75)
        self.deviation_log.append(
            {
                "type": "story_consensus_corrected",
                "story_id": story_id,
                "relationship_id": story.relationship_id,
                "reason": reason,
                "at": now.isoformat(),
            }
        )
        return correction

    def rollback_story_narrative(
        self,
        story_id: str,
        *,
        version_index: int | None = None,
        reason: str = "user_story_rollback",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        story = self.story_nodes[story_id]
        if not story.narrative_versions:
            raise ValueError("story has no narrative versions to roll back")
        selected_index = version_index if version_index is not None else len(story.narrative_versions) - 1
        if selected_index < 0:
            selected_index = len(story.narrative_versions) + selected_index
        if selected_index < 0 or selected_index >= len(story.narrative_versions):
            raise IndexError("narrative version index out of range")
        version = story.narrative_versions[selected_index]
        previous_level = story.narrative_level
        previous_consensus = story.consensus_version
        target_level = NarrativeLevel(version["previous_level"])
        target_consensus = str(version.get("previous_consensus", ""))
        story.narrative_level = target_level
        story.consensus_version = target_consensus
        story.consensus_status = "SIMULATED_FROM_USER_ACCOUNT"
        story.consensus_confirmed_at = None
        story.consensus_provenance = self._story_consensus_provenance(
            source="story_narrative_rollback",
            status=story.consensus_status,
            memory_ids=list(story.core_events),
            now=now,
            reason=reason,
        )
        rollback = {
            "at": now.isoformat(),
            "reason": reason,
            "rollback": True,
            "version_index": selected_index,
            "from_level": previous_level.value,
            "to_level": story.narrative_level.value,
            "from_consensus": previous_consensus,
            "to_consensus": story.consensus_version,
            "rolled_back_version": dict(version),
        }
        story.narrative_versions.append(rollback)
        if len(story.narrative_versions) > 50:
            story.narrative_versions = story.narrative_versions[-50:]
        self.deviation_log.append(
            {
                "type": "story_narrative_rolled_back",
                "story_id": story_id,
                "relationship_id": story.relationship_id,
                "version_index": selected_index,
                "from_level": previous_level.value,
                "to_level": story.narrative_level.value,
                "reason": reason,
                "at": now.isoformat(),
            }
        )
        return rollback

    def confirm_story_consensus(
        self,
        story_id: str,
        *,
        note: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        story = self.story_nodes[story_id]
        previous_status = story.consensus_status
        story.consensus_status = "USER_CONFIRMED"
        story.consensus_confirmed_at = now
        story.consensus_provenance = self._story_consensus_provenance(
            source="user_confirmation",
            status=story.consensus_status,
            memory_ids=list(story.core_events),
            now=now,
            reason=note,
        )
        story.ai_framing_confidence = max(story.ai_framing_confidence, 0.85)
        story.consistency_score = max(story.consistency_score, 0.85)
        event = {
            "type": "story_consensus_confirmed",
            "story_id": story_id,
            "relationship_id": story.relationship_id,
            "previous_status": previous_status,
            "status": story.consensus_status,
            "note": note,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def generate_guardian_summary(
        self,
        relationship_id: str,
        *,
        period_start: datetime | None = None,
        period_end: datetime | None = None,
        now: datetime | None = None,
    ) -> GuardianSummary:
        now = now or utcnow()
        period_end = period_end or now
        period_start = period_start or (period_end - timedelta(days=7))
        relationship = self.relationships[relationship_id]
        if not relationship.preferences.guardian_summary_enabled:
            raise PermissionError("guardian summary is disabled for this relationship")
        if relationship.user_age is None or relationship.user_age >= 18:
            raise ValueError("guardian summary is only available for minor users")

        self.evaluate_health(relationship_id, now=now)
        memories = [
            memory
            for memory in self.memories.values()
            if memory.relationship_id == relationship_id and period_start <= memory.created_at <= period_end
        ]
        emotions = [
            emotion
            for emotion in self.emotional_memories.values()
            if emotion.relationship_id == relationship_id and period_start <= emotion.timestamp <= period_end
        ]
        health_alert_ids = [
            alert.alert_id
            for alert in self.health_alerts.values()
            if alert.relationship_id == relationship_id
            and (period_start <= alert.created_at <= period_end or not alert.acknowledged)
        ]
        date_keys = {
            (period_start + timedelta(days=offset)).date().isoformat()
            for offset in range((period_end.date() - period_start.date()).days + 1)
        }
        total_minutes = sum(minutes for day, minutes in relationship.daily_interaction_minutes.items() if day in date_keys)
        active_behavior_count = sum(
            1
            for item in relationship.active_behavior_log
            if period_start <= _dt(item.get("at")) <= period_end
        )
        recommendation = self._guardian_recommendation(
            relationship=relationship,
            total_minutes=total_minutes,
            health_alert_ids=health_alert_ids,
            emotional_memory_count=len(emotions),
        )
        summary = GuardianSummary(
            summary_id=new_id("guardian"),
            relationship_id=relationship_id,
            period_start=period_start,
            period_end=period_end,
            generated_at=now,
            user_age=relationship.user_age,
            stage=relationship.stage,
            interaction_count=len(memories),
            total_minutes=total_minutes,
            memory_count=len(memories),
            emotional_memory_count=len(emotions),
            active_behavior_count=active_behavior_count,
            health_alert_ids=health_alert_ids,
            milestone_count=len([mid for mid in relationship.milestones if mid in self.memories and period_start <= self.memories[mid].created_at <= period_end]),
            core_identity_count=len(relationship.core_identity),
            recommendation=recommendation,
            privacy_boundary=self._guardian_privacy_boundary(),
            resource_summary=self._guardian_resource_summary(health_alert_ids),
        )
        self.guardian_summaries[summary.summary_id] = summary
        self.deviation_log.append(
            {
                "type": "guardian_summary_generated",
                "summary_id": summary.summary_id,
                "relationship_id": relationship_id,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "generated_at": now.isoformat(),
            }
        )
        return summary

    def _guardian_privacy_boundary(self) -> dict[str, Any]:
        return {
            "content_retained": False,
            "memory_text_included": False,
            "core_identity_text_included": False,
            "contains_counts_only": True,
            "visible_fields": [
                "age_band",
                "stage",
                "interaction_counts",
                "time_usage",
                "health_alert_ids",
                "resource_summary",
                "recommendation",
            ],
            "redaction_reason": "guardian summaries protect the minor by surfacing safety signals without disclosing private conversation text.",
        }

    def _guardian_resource_summary(self, health_alert_ids: list[str]) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for alert_id in health_alert_ids:
            alert = self.health_alerts.get(alert_id)
            if not alert:
                continue
            for resource in alert.resources:
                key = (
                    str(resource.get("type", "")),
                    str(resource.get("region", "")),
                    str(resource.get("phone") or resource.get("chat_url") or resource.get("label") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                resources.append({**resource, "source_alert_id": alert_id, "source_risk_type": alert.risk_type})
        return resources

    def browser_snapshot(self, relationship_id: str) -> dict[str, Any]:
        relationship = self.relationships[relationship_id]
        stories = [story for story in self.story_nodes.values() if story.relationship_id == relationship_id]
        emotional = self.emotional_trajectories.get(relationship_id, EmotionalTrajectory(relationship_id))
        memories = [memory for memory in self.memories.values() if memory.relationship_id == relationship_id]
        suppressed_memories = [memory for memory in memories if self._memory_is_recall_suppressed(memory)]
        metacognition = [self._ensure_memory_metacognition(memory) for memory in memories]
        low_confidence = [
            memory
            for memory in memories
            if self._ensure_memory_metacognition(memory).get("confidence", 0.0) < 0.55
        ]
        source_timed = [memory for memory in memories if memory.metadata.get("source_time")]
        time_conflicted = [memory for memory in memories if self._active_time_conflicts(memory)]
        embedding_metacognitions = [
            memory.metadata.get("embeddings", {}).get("metacognition")
            for memory in memories
            if isinstance(memory.metadata.get("embeddings", {}).get("metacognition"), dict)
        ]
        return {
            "dashboard": {
                "relationship_id": relationship.relationship_id,
                "stage": relationship.stage.value,
                "strength": relationship.strength,
                "trust_level": relationship.trust_level,
                "intimacy_level": relationship.intimacy_level,
                "retention_multiplier": relationship.retention_multiplier,
                "decay_curve_type": relationship.decay_curve_type.value,
                "retention_calibration": self._retention_calibration_summary(relationship),
                "stage_health_gate": relationship.maintenance_signals.get("stage_health_gate"),
                "relationship_age": relationship.relationship_age,
                "interaction_count": relationship.interaction_count,
                "mode": relationship.preferences.mode.value,
            },
            "counts": {
                "memories": len(memories),
                "archived_memories": len([memory for memory in memories if memory.metadata.get("archived")]),
                "suppressed_memories": len(suppressed_memories),
                "emotional_memories": len(
                    [item for item in self.emotional_memories.values() if item.relationship_id == relationship_id]
                ),
            "stories": len(stories),
            "memory_graph_edges": len(
                [edge for edge in self.memory_graph_edges.values() if edge.relationship_id == relationship_id]
            ),
            "milestones": len(relationship.milestones),
                "inside_jokes": len(relationship.inside_jokes),
                "inside_joke_candidates": len(relationship.inside_joke_candidates),
                "unresolved_threads": len(
                    [
                        mid
                        for mid in relationship.unresolved_threads
                        if mid in self.memories and not self._memory_is_recall_suppressed(self.memories[mid])
                    ]
                ),
                "commitment_reminders": len(
                    [
                        item
                        for item in self.commitment_reminders.values()
                        if item.relationship_id == relationship_id
                        and item.status in {ReminderStatus.PENDING, ReminderStatus.REMINDER_SENT}
                        and not self._memory_is_recall_suppressed(self.memories.get(item.memory_id))
                    ]
                ),
            },
            "storage_layers": self._storage_layer_counts(memories),
            "metacognition": {
                "avg_confidence": (
                    sum(float(item.get("confidence", 0.0)) for item in metacognition) / len(metacognition)
                    if metacognition
                    else None
                ),
                "low_confidence_count": len(low_confidence),
                "human_verified_count": len([item for item in metacognition if item.get("human_verified")]),
                "source_time_count": len(source_timed),
                "time_conflict_count": len(time_conflicted),
                "calibration": self._relationship_calibration_summary(memories),
                "embedding": {
                    "count": len(embedding_metacognitions),
                    "trained_embedding_count": len(
                        [item for item in embedding_metacognitions if item.get("trained_embedding")]
                    ),
                    "local_heuristic_count": len(
                        [item for item in embedding_metacognitions if item.get("provider") == "local_heuristic"]
                    ),
                    "avg_confidence": (
                        sum(float(item.get("confidence", 0.0)) for item in embedding_metacognitions)
                        / len(embedding_metacognitions)
                        if embedding_metacognitions
                        else None
                    ),
                    "limitations": sorted(
                        {
                            limitation
                            for item in embedding_metacognitions
                            for limitation in item.get("limitations", [])
                        }
                    ),
                },
                "low_confidence_memories": [
                    {
                        "memory_id": memory.memory_id,
                        "content": memory.content,
                        "metacognition": memory.metadata.get("metacognition"),
                    }
                    for memory in low_confidence[:10]
                ],
                "source_time_memories": [
                    {
                        "memory_id": memory.memory_id,
                        "content": memory.content,
                        "source_time": memory.metadata.get("source_time"),
                    }
                    for memory in source_timed[:10]
                ],
                "time_conflicts": [
                    {
                        "memory_id": memory.memory_id,
                        "content": memory.content,
                        "source_time": memory.metadata.get("source_time"),
                        "conflicts": memory.metadata.get("time_conflicts", []),
                        "metacognition": memory.metadata.get("metacognition"),
                    }
                    for memory in time_conflicted[:10]
                ],
            },
            "suppressed_memories": [
                {
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "recall_boundary": memory.metadata.get("recall_boundary"),
                }
                for memory in suppressed_memories
            ],
            "relationship_narrative": self._to_json(relationship.relationship_narrative),
            "milestones": [self.memories[mid].content for mid in relationship.milestones if mid in self.memories],
            "milestone_details": [
                {
                    "memory_id": mid,
                    "title": self.memories[mid].metadata.get("milestone_confirmation", {}).get("title")
                    or self.memories[mid].content[:40],
                    "description": self.memories[mid].metadata.get("milestone_confirmation", {}).get("description"),
                    "content": self.memories[mid].content,
                    "confirmation": self.memories[mid].metadata.get("milestone_confirmation", {"status": "CONFIRMED"}),
                }
                for mid in relationship.milestones
                if mid in self.memories
            ],
            "pending_milestones": [
                {
                    "memory_id": mid,
                    "content": self.memories[mid].content,
                    "detected_at": self.memories[mid].metadata.get("milestone_confirmation", {}).get("detected_at"),
                    "reason": self.memories[mid].metadata.get("milestone_confirmation", {}).get("reason"),
                }
                for mid in relationship.milestones
                if mid in self.memories
                and self.memories[mid].metadata.get("milestone_confirmation", {}).get("status") == "PENDING"
            ],
            "inside_jokes": [
                {
                    "memory_id": mid,
                    "content": self.memories[mid].content,
                    "phrase": self.memories[mid].metadata.get("inside_joke_phrase"),
                    "replay_count": len(self.memories[mid].metadata.get("inside_joke_replay_log", [])),
                    "inactive": bool(self.memories[mid].metadata.get("inside_joke_inactive")),
                    "negative_feedback_count": self.memories[mid].metadata.get("inside_joke_negative_feedback", 0),
                }
                for mid in relationship.inside_jokes
                if mid in self.memories
            ],
            "inside_joke_candidates": relationship.inside_joke_candidates,
            "unresolved_threads": [
                {
                    "memory_id": mid,
                    "content": self.memories[mid].content,
                    "resolution": self.memories[mid].metadata.get("unresolved_thread_resolution"),
                }
                for mid in relationship.unresolved_threads
                if mid in self.memories and not self._memory_is_recall_suppressed(self.memories[mid])
            ],
            "resolved_threads": [
                {
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "resolution": memory.metadata.get("unresolved_thread_resolution"),
                }
                for memory in self.memories.values()
                if memory.relationship_id == relationship_id
                and memory.metadata.get("unresolved_thread_resolution")
            ],
            "commitment_reminders": [
                self._commitment_reminder_payload(item, utcnow())
                for item in sorted(
                    self.commitment_reminders.values(),
                    key=lambda reminder: reminder.due_at,
                )
                if item.relationship_id == relationship_id
                and not self._memory_is_recall_suppressed(self.memories.get(item.memory_id))
            ],
            "stories": [
                {
                    "story_id": story.story_id,
                    "title": story.title,
                    "level": story.narrative_level.value,
                    "events": len(story.core_events),
                    "retell_count": story.retell_count,
                    "themes": story.recurring_themes,
                    "consensus": story.consensus_version,
                    "consensus_status": story.consensus_status,
                    "consensus_provenance": story.consensus_provenance,
                    "consensus_confirmed_at": story.consensus_confirmed_at.isoformat()
                    if story.consensus_confirmed_at
                    else None,
                    "source_references": self._story_source_references(story),
                    "conflict_versions": story.conflict_versions[-5:],
                    "correction_versions": story.correction_versions[-5:],
                    "narrative_versions": story.narrative_versions[-5:],
                    "child_inside_jokes": [
                        {
                            "memory_id": memory_id,
                            "phrase": self.memories[memory_id].metadata.get("inside_joke_phrase"),
                            "content": self.memories[memory_id].content,
                        }
                        for memory_id in story.child_inside_jokes
                        if memory_id in self.memories
                    ],
                    "consistency_score": story.consistency_score,
                    "user_framing": story.user_framing,
                    "ai_framing_confidence": story.ai_framing_confidence,
                }
                for story in sorted(stories, key=lambda item: item.story_arc_start)
            ],
            "memory_graph": {
                "edge_count": len([edge for edge in self.memory_graph_edges.values() if edge.relationship_id == relationship_id]),
                "recent_edges": [
                    {
                        "edge_id": edge.edge_id,
                        "source_memory_id": edge.source_memory_id,
                        "target_memory_id": edge.target_memory_id,
                        "relation_type": edge.relation_type,
                        "weight": edge.weight,
                        "evidence": edge.evidence,
                    }
                    for edge in sorted(
                        [edge for edge in self.memory_graph_edges.values() if edge.relationship_id == relationship_id],
                        key=lambda item: item.created_at,
                        reverse=True,
                    )[:20]
                ],
            },
            "emotional_trajectory": [
                {
                    "window": f"{window.window_start.date()}..{window.window_end.date()}",
                    "avg_valence": window.avg_valence,
                    "avg_arousal": window.avg_arousal,
                    "dominant_emotions": window.dominant_emotions,
                    "notable_events": window.notable_events,
                }
                for window in emotional.time_series[-8:]
            ],
            "active_behavior_log": relationship.active_behavior_log[-20:],
            "emotional_baseline": self._baseline_snapshot(relationship.emotional_baseline),
            "active_suppression_log": [
                item
                for item in self.deviation_log
                if item.get("relationship_id") == relationship_id and item.get("type") == "active_suppressed"
            ][-20:],
            "active_feedback_state": relationship.active_feedback_state,
            "implicit_topics": relationship.implicit_topics[-20:],
            "maintenance_signals": relationship.maintenance_signals,
            "trust_decay_state": relationship.trust_decay_state,
            "mode_suggestions": self.mode_suggestions(relationship_id),
            "stage_history": relationship.stage_history[-20:],
            "core_identity": [
                {
                    "identity_id": item.identity_id,
                    "title": item.title,
                    "content": item.content,
                    "pending_delete": item.pending_delete,
                    "review_status": item.review_status,
                    "review_score": item.review_score,
                    "user_confirmed_at": item.user_confirmed_at.isoformat() if item.user_confirmed_at else None,
                    "review_history": item.review_history[-5:],
                    "changes": len(item.change_log),
                    "replica_count": len(item.replicas),
                    "replicas": item.replicas,
                }
                for item in self.core_identity.values()
                if item.relationship_id == relationship_id
            ],
            "core_identity_delete_requests": [
                {
                    "request_id": item.request_id,
                    "identity_id": item.identity_id,
                    "status": item.status.value,
                    "execute_after": item.execute_after.isoformat(),
                    "executed_at": item.executed_at.isoformat() if item.executed_at else None,
                }
                for item in self.core_identity_delete_requests.values()
                if item.relationship_id == relationship_id
            ],
            "memory_delete_requests": [
                {
                    "request_id": item.request_id,
                    "memory_id": item.memory_id,
                    "status": item.status.value,
                    "execute_after": item.execute_after.isoformat(),
                    "executed_at": item.executed_at.isoformat() if item.executed_at else None,
                }
                for item in self.memory_delete_requests.values()
                if item.relationship_id == relationship_id
            ],
            "reset_requests": [
                {
                    "request_id": item.request_id,
                    "mode": item.mode.value,
                    "status": item.status.value,
                    "execute_after": item.execute_after.isoformat(),
                }
                for item in self.reset_requests.values()
                if item.relationship_id == relationship_id
            ],
            "health_alerts": [
                {
                    "alert_id": item.alert_id,
                    "risk_type": item.risk_type,
                    "level": item.level.value,
                    "message": item.message,
                    "created_at": item.created_at.isoformat(),
                    "acknowledged": item.acknowledged,
                    "acknowledged_at": item.acknowledged_at.isoformat() if item.acknowledged_at else None,
                    "acknowledgement_note": item.acknowledgement_note,
                    "feedback": item.feedback,
                    "feedback_at": item.feedback_at.isoformat() if item.feedback_at else None,
                    "feedback_note": item.feedback_note,
                }
                for item in self.health_alerts.values()
                if item.relationship_id == relationship_id
            ],
            "guardian_summaries": [
                {
                    "summary_id": item.summary_id,
                    "period_start": item.period_start.isoformat(),
                    "period_end": item.period_end.isoformat(),
                    "generated_at": item.generated_at.isoformat(),
                    "recommendation": item.recommendation,
                }
                for item in self.guardian_summaries.values()
                if item.relationship_id == relationship_id
            ][-8:],
            "retrieval_audit_log": [
                item for item in self.retrieval_audit_log if item.get("relationship_id") == relationship_id
            ][-20:],
            "ai_decision_log": [
                item for item in self.ai_decision_log if item.get("relationship_id") == relationship_id
            ][-20:],
            "ai_decision_summaries": [
                self.ai_decision_summary(item)
                for item in self.ai_decision_log
                if item.get("relationship_id") == relationship_id
            ][-20:],
            "control_audit_log": [
                item
                for item in self.deviation_log
                if item.get("relationship_id") == relationship_id
                and item.get("type")
                in {
                    "batch_downgrade",
                    "milestone_confirmed",
                    "milestone_edited",
                    "milestone_rejected",
                    "milestone_story_created",
                    "milestone_story_promoted",
                    "stage_transition_milestone_created",
                    "stage_transition_blocked",
                    "emotional_turning_point_milestone_created",
                    "major_shared_decision_milestone_created",
                    "shared_celebration_milestone_created",
                    "l4_delete_requested",
                    "l4_delete_confirmed",
                    "l4_delete_cancelled",
                    "l4_capacity_downgraded",
                    "relationship_schema_updated",
                    "inside_joke_attached_to_story",
                    "inside_joke_deactivated",
                    "inside_joke_reactivated",
                    "unresolved_thread_resolved",
                    "core_identity_delete_requested",
                    "core_identity_delete_executed",
                    "core_identity_delete_cancelled",
                    "memory_delete_requested",
                    "memory_delete_executed",
                    "memory_delete_cancelled",
                    "critical_memory_tombstone_created",
                    "critical_tombstone_rementioned",
                    "story_consensus_corrected",
                    "story_marked_deleted_source",
                    "story_rebuilt_after_deleted_source",
                    "story_deleted_after_source_rebuild",
                    "memory_writes_paused",
                    "memory_writes_resumed",
                    "memory_write_skipped",
                    "health_review_completed",
                    "health_alert_acknowledged",
                    "health_alert_feedback",
                    "health_prompt_cooldown_started",
                    "health_alert_suppressed_by_cooldown",
                    "memory_verified",
                    "memory_calibrated",
                    "memory_reconsolidated",
                    "memory_superseded",
                    "retention_feedback",
                    "retrieval_adaptation",
                    "cold_information_reviewed",
                    "story_consensus_confirmed",
                    "memory_recall_suppressed",
                    "memory_recall_unsuppressed",
                    "memory_boundary_request",
                    "preference_updated",
                    "decay_curve_type_changed",
                    "mode_changed",
                    "custom_mode_profile_updated",
                    "transparency_acknowledged",
                    "age_clarification_requested",
                    "user_age_updated",
                    "active_type_muted",
                    "active_type_unmuted",
                    "time_conflict_detected",
                    "time_conflict_resolved",
                    "export_generated",
                    "export_blocked",
                    "deletion_compliance_access_denied",
                    "deletion_compliance_auditor_accessed",
                    "implicit_topic_feedback",
                    "implicit_topic_evidence_failed",
                    "reset_requested",
                    "reset_confirmed",
                    "reset_executed",
                    "reset_cancelled",
                }
            ][-20:],
            "deletion_compliance_summary": self.deletion_compliance_summary(relationship_id),
            "preferences": self._to_json(relationship.preferences),
            "transparency": self.transparency_panel(relationship_id),
        }

    def deletion_compliance_summary(self, relationship_id: str) -> dict[str, Any]:
        records = [item for item in self.deletion_compliance_log if item.get("relationship_id") == relationship_id]
        latest = records[-1] if records else None
        return {
            "count": len(records),
            "access_scope": "audit_only",
            "browser_detail_available": False,
            "auditor_access_required": True,
            "latest": {
                "deletion_type": latest.get("deletion_type"),
                "request_id": latest.get("request_id"),
                "recorded_at": latest.get("recorded_at"),
                "content_retained": latest.get("content_retained", False),
            }
            if latest
            else None,
        }

    def deletion_compliance_audit(
        self,
        relationship_id: str,
        *,
        auditor_token: str | None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        if auditor_token != "local-auditor":
            self.deviation_log.append(
                {
                    "type": "deletion_compliance_access_denied",
                    "relationship_id": relationship_id,
                    "reason": "invalid_auditor_token",
                    "at": now.isoformat(),
                }
            )
            raise PermissionError("deletion compliance logs require auditor access")
        records = [item for item in self.deletion_compliance_log if item.get("relationship_id") == relationship_id]
        self.deviation_log.append(
            {
                "type": "deletion_compliance_auditor_accessed",
                "relationship_id": relationship_id,
                "count": len(records),
                "at": now.isoformat(),
            }
        )
        return {
            "relationship_id": relationship_id,
            "access_scope": "audit_only",
            "auditor_access": True,
            "records": records[-100:],
        }

    def export(self, *, anonymized: bool = False) -> dict[str, Any]:
        payload = {
            "relationships": {key: self._to_json(value) for key, value in self.relationships.items()},
            "memories": {key: self._to_json(value) for key, value in self.memories.items()},
            "memory_graph_edges": {key: self._to_json(value) for key, value in self.memory_graph_edges.items()},
            "emotional_memories": {key: self._to_json(value) for key, value in self.emotional_memories.items()},
            "story_nodes": {key: self._to_json(value) for key, value in self.story_nodes.items()},
            "emotional_trajectories": {key: self._to_json(value) for key, value in self.emotional_trajectories.items()},
            "core_identity": {key: self._to_json(value) for key, value in self.core_identity.items()},
            "core_identity_delete_requests": {key: self._to_json(value) for key, value in self.core_identity_delete_requests.items()},
            "memory_delete_requests": {key: self._to_json(value) for key, value in self.memory_delete_requests.items()},
            "reset_requests": {key: self._to_json(value) for key, value in self.reset_requests.items()},
            "health_alerts": {key: self._to_json(value) for key, value in self.health_alerts.items()},
            "guardian_summaries": {key: self._to_json(value) for key, value in self.guardian_summaries.items()},
            "commitment_reminders": {key: self._to_json(value) for key, value in self.commitment_reminders.items()},
            "retrieval_audit_log": self.retrieval_audit_log,
            "ai_decision_log": self.ai_decision_log,
            "deviation_log": self.deviation_log,
            "deletion_compliance_log": self.deletion_compliance_log,
            "relationship_ending_support_log": self.relationship_ending_support_log,
            "migration_batches": self.migration_batches,
        }
        if anonymized:
            return self._anonymize_export(payload)
        return payload

    def generate_export(
        self,
        *,
        relationship_id: str | None = None,
        export_format: str = "json",
        anonymized: bool = False,
        destination: str = "response",
        purpose: str = "user_archive",
        now: datetime | None = None,
    ) -> dict[str, Any] | str:
        now = now or utcnow()
        self._enforce_export_policy(relationship_id, export_format, destination, purpose, now)
        effective_anonymized = anonymized or export_format == "anonymous-json"
        if export_format in {"json", "anonymous-json"}:
            payload: dict[str, Any] | str = self.export(anonymized=effective_anonymized)
        elif export_format == "narrative":
            if not relationship_id:
                raise ValueError("relationship_id is required for narrative export")
            payload = self.export_narrative_document(relationship_id)
        elif export_format == "milestones":
            if not relationship_id:
                raise ValueError("relationship_id is required for milestones export")
            payload = self.export_milestone_album(relationship_id)
        elif export_format == "timeline":
            if not relationship_id:
                raise ValueError("relationship_id is required for timeline export")
            payload = self.export_emotional_timeline_csv(relationship_id)
        else:
            raise ValueError(f"unknown export format: {export_format}")
        self.deviation_log.append(
            {
                "type": "export_generated",
                "relationship_id": relationship_id,
                "format": export_format,
                "anonymized": effective_anonymized,
                "destination": destination,
                "purpose": purpose,
                "at": now.isoformat(),
            }
        )
        return payload

    def _enforce_export_policy(
        self,
        relationship_id: str | None,
        export_format: str,
        destination: str,
        purpose: str,
        now: datetime,
    ) -> dict[str, Any]:
        normalized = destination.strip().lower()
        normalized_purpose = purpose.strip().lower()
        allowed_aliases = {"response", "http_response", "stdout", "test"}
        remote_markers = (
            "://",
            "s3:",
            "gs:",
            "oss:",
            "cloud",
            "remote",
            "third_party",
            "third-party",
            "commercial",
            "share",
            "webhook",
            "cross_border",
            "cross-border",
            "overseas",
            "vendor",
            "partner",
        )
        blocked_reasons: list[str] = []
        if normalized not in allowed_aliases and any(marker in normalized for marker in remote_markers):
            blocked_reasons.append("non_local_or_third_party_destination")
        disallowed_purpose_markers = {
            "training",
            "model_training",
            "fine_tuning",
            "analytics",
            "cross_border",
            "cross-border",
            "third_party",
            "third-party",
            "commercial",
            "vendor_share",
        }
        if normalized_purpose in disallowed_purpose_markers or any(marker in normalized_purpose for marker in disallowed_purpose_markers):
            blocked_reasons.append("disallowed_relationship_memory_purpose")
        if export_format not in {"json", "anonymous-json"} and self.export_requires_anonymization(relationship_id):
            blocked_reasons.append("anonymized_permission_requires_anonymized_json_export")
        blocked = bool(blocked_reasons)
        if not blocked:
            return
        self.deviation_log.append(
            {
                "type": "export_blocked",
                "relationship_id": relationship_id,
                "format": export_format,
                "destination_sealed": self._seal_audit_text(destination),
                "purpose": normalized_purpose,
                "reason": blocked_reasons[0],
                "reasons": blocked_reasons,
                "at": now.isoformat(),
            }
        )
        raise PermissionError("relationship memory exports are restricted to local/user-held, non-training purposes")

    def export_requires_anonymization(self, relationship_id: str | None = None) -> bool:
        relationships = (
            [self.relationships[relationship_id]]
            if relationship_id and relationship_id in self.relationships
            else list(self.relationships.values())
        )
        return any(item.preferences.data_export_permission.upper() in {"ANONYMIZED", "ANONYMOUS"} for item in relationships)

    def export_narrative_document(self, relationship_id: str) -> str:
        relationship = self.relationships[relationship_id]
        stories = sorted(
            [story for story in self.story_nodes.values() if story.relationship_id == relationship_id],
            key=lambda item: item.story_arc_start,
        )
        lines = [
            f"# 我们的故事：{relationship.relationship_id}",
            "",
            f"- 阶段：{relationship.stage.value}",
            f"- 关系强度：{relationship.strength:.3f}",
            f"- 信任度：{relationship.trust_level:.3f}",
            f"- 关系年龄：{relationship.relationship_age} 天",
            f"- 核心主题：{', '.join(relationship.relationship_narrative.core_themes) or '暂无'}",
            f"- 透明度确认：{relationship.transparency_acknowledged_at.isoformat() if relationship.transparency_acknowledged_at else '未确认'}",
            "",
            "## 关系摘要",
            "",
            relationship.relationship_narrative.origin_story or "这段关系仍在形成自己的起点叙事。",
            "",
            "## 故事线",
            "",
        ]
        if not stories:
            lines.append("暂无共同叙事节点。")
        for story in stories:
            lines.extend(
                [
                    f"### {story.title}",
                    "",
                    f"- 层级：{story.narrative_level.value}",
                    f"- 主题：{', '.join(story.recurring_themes) or '暂无'}",
                    f"- 复述次数：{story.retell_count}",
                    f"- 时间：{story.story_arc_start.date()} 至 {story.story_arc_end.date() if story.story_arc_end else '进行中'}",
                    f"- 共识状态：{story.consensus_status}",
                    f"- 共识来源：{story.consensus_provenance.get('source', 'unknown') if story.consensus_provenance else 'unknown'}",
                    "",
                    story.consensus_version or "暂无共识摘要。",
                    "",
                ]
            )
            key_events = [self.memories[mid].content for mid in story.key_moments if mid in self.memories]
            if key_events:
                lines.append("关键时刻：")
                lines.extend([f"- {event}" for event in key_events])
                lines.append("")
            source_refs = self._story_source_references(story)
            if source_refs:
                lines.append("来源引用：")
                for ref in source_refs:
                    marker = "关键" if ref["is_key_moment"] else "事件"
                    lines.append(
                        f"- [{marker}] {ref['memory_id']} "
                        f"({ref['memory_type']}/{ref['context_tag']}, {ref['created_at'][:10]}): {ref['content']}"
                    )
                lines.append("")
        return "\n".join(lines).strip() + "\n"

    def export_milestone_album(self, relationship_id: str) -> dict[str, Any]:
        relationship = self.relationships[relationship_id]
        album = []
        for memory_id in relationship.milestones:
            memory = self.memories.get(memory_id)
            if not memory:
                continue
            confirmation = memory.metadata.get("milestone_confirmation", {})
            album.append(
                {
                    "memory_id": memory.memory_id,
                    "title": confirmation.get("title") or memory.content[:40],
                    "description": confirmation.get("description"),
                    "content": memory.content,
                    "created_at": memory.created_at.isoformat(),
                    "relationship_age_at_creation": memory.relationship_age_at_creation,
                    "emotion_intensity": memory.emotion_intensity,
                    "trust_level_at_creation": memory.trust_level_at_creation,
                    "confirmation": confirmation or {"status": "CONFIRMED"},
                    "tags": sorted(memory.tags),
                }
            )
        return {"relationship_id": relationship_id, "count": len(album), "milestones": album}

    def export_emotional_timeline_csv(self, relationship_id: str) -> str:
        trajectory = self.emotional_trajectories.get(relationship_id, EmotionalTrajectory(relationship_id))
        output = StringIO()
        writer = DictWriter(
            output,
            fieldnames=[
                "window_start",
                "window_end",
                "avg_valence",
                "avg_arousal",
                "dominant_emotions",
                "emotional_diversity",
                "notable_events",
            ],
        )
        writer.writeheader()
        for window in trajectory.time_series:
            writer.writerow(
                {
                    "window_start": window.window_start.isoformat(),
                    "window_end": window.window_end.isoformat(),
                    "avg_valence": f"{window.avg_valence:.4f}",
                    "avg_arousal": f"{window.avg_arousal:.4f}",
                    "dominant_emotions": "|".join(window.dominant_emotions),
                    "emotional_diversity": f"{window.emotional_diversity:.4f}",
                    "notable_events": "|".join(window.notable_events),
                }
            )
        return output.getvalue()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.export(), ensure_ascii=False, indent=2)
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "FriendMemoryProject":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls._from_data(data)

    @classmethod
    def _from_data(cls, data: dict[str, Any]) -> "FriendMemoryProject":
        project = cls()
        for key, raw in data.get("relationships", {}).items():
            project.relationships[key] = _relationship_from_json(raw)
        for key, raw in data.get("memories", {}).items():
            project.memories[key] = _memory_from_json(raw)
        for key, raw in data.get("memory_graph_edges", {}).items():
            project.memory_graph_edges[key] = _memory_graph_edge_from_json(raw)
        for key, raw in data.get("emotional_memories", {}).items():
            project.emotional_memories[key] = _emotional_from_json(raw)
        for key, raw in data.get("story_nodes", {}).items():
            project.story_nodes[key] = _story_from_json(raw)
        for key, raw in data.get("emotional_trajectories", {}).items():
            project.emotional_trajectories[key] = _trajectory_from_json(raw)
        for key, raw in data.get("core_identity", {}).items():
            project.core_identity[key] = _core_identity_from_json(raw)
        for key, raw in data.get("core_identity_delete_requests", {}).items():
            project.core_identity_delete_requests[key] = _core_identity_delete_request_from_json(raw)
        for key, raw in data.get("memory_delete_requests", {}).items():
            project.memory_delete_requests[key] = _memory_delete_request_from_json(raw)
        for key, raw in data.get("reset_requests", {}).items():
            project.reset_requests[key] = _reset_request_from_json(raw)
        for key, raw in data.get("health_alerts", {}).items():
            project.health_alerts[key] = _health_alert_from_json(raw)
        for key, raw in data.get("guardian_summaries", {}).items():
            project.guardian_summaries[key] = _guardian_summary_from_json(raw)
        for key, raw in data.get("commitment_reminders", {}).items():
            project.commitment_reminders[key] = _commitment_reminder_from_json(raw)
        project.retrieval_audit_log = data.get("retrieval_audit_log", [])
        project.ai_decision_log = data.get("ai_decision_log", [])
        project.deviation_log = data.get("deviation_log", [])
        project.deletion_compliance_log = data.get("deletion_compliance_log", [])
        project.relationship_ending_support_log = data.get("relationship_ending_support_log", [])
        project.migration_batches = data.get("migration_batches", {})
        project._repair_loaded_references()
        return project

    def _repair_loaded_references(self) -> None:
        removed: dict[str, int] = {
            "relationship_memory_refs": 0,
            "relationship_core_identity_refs": 0,
            "story_memory_refs": 0,
            "emotional_memories": 0,
            "memory_graph_edges": 0,
            "commitment_reminders_archived": 0,
            "core_identity_records": 0,
        }
        for relationship in self.relationships.values():
            for field_name in ("milestones", "shared_episodes", "inside_jokes", "unresolved_threads"):
                current = list(getattr(relationship, field_name))
                repaired = [memory_id for memory_id in current if memory_id in self.memories]
                removed["relationship_memory_refs"] += len(current) - len(repaired)
                setattr(relationship, field_name, repaired)
            current_identity_refs = list(relationship.core_identity)
            relationship.core_identity = [
                identity_id
                for identity_id in current_identity_refs
                if identity_id in self.core_identity and self.core_identity[identity_id].memory_id in self.memories
            ]
            removed["relationship_core_identity_refs"] += len(current_identity_refs) - len(relationship.core_identity)

        for story in self.story_nodes.values():
            original_core_events = list(story.core_events)
            original_key_moments = list(story.key_moments)
            story.core_events = [memory_id for memory_id in story.core_events if memory_id in self.memories]
            story.key_moments = [memory_id for memory_id in story.key_moments if memory_id in self.memories]
            removed["story_memory_refs"] += (
                len(original_core_events)
                + len(original_key_moments)
                - len(story.core_events)
                - len(story.key_moments)
            )

        for emotion_id, emotion in list(self.emotional_memories.items()):
            if emotion.source_memory_id not in self.memories or emotion.relationship_id not in self.relationships:
                del self.emotional_memories[emotion_id]
                removed["emotional_memories"] += 1

        for edge_id, edge in list(self.memory_graph_edges.items()):
            if (
                edge.relationship_id not in self.relationships
                or edge.source_memory_id not in self.memories
                or edge.target_memory_id not in self.memories
            ):
                del self.memory_graph_edges[edge_id]
                removed["memory_graph_edges"] += 1

        for reminder in self.commitment_reminders.values():
            if reminder.relationship_id in self.relationships and reminder.memory_id not in self.memories and reminder.status != ReminderStatus.ARCHIVED:
                reminder.status = ReminderStatus.ARCHIVED
                reminder.metadata["archived_reason"] = "loaded_missing_source_memory"
                removed["commitment_reminders_archived"] += 1

        for identity_id, identity in list(self.core_identity.items()):
            if identity.relationship_id not in self.relationships or identity.memory_id not in self.memories:
                del self.core_identity[identity_id]
                removed["core_identity_records"] += 1
                continue
            if len(identity.replicas) < 3:
                self._refresh_l4_replicas(identity, now=identity.updated_at, reason="load_repair")

        if any(removed.values()):
            self.deviation_log.append(
                {
                    "type": "loaded_reference_repair",
                    "removed": removed,
                    "at": utcnow().isoformat(),
                }
            )

    def _friend_score(self, signals: TurnSignals) -> float:
        return clamp(
            0.30 * signals.relationship_depth
            + 0.25 * signals.emotion_intensity
            + 0.25 * signals.personal_importance
            + 0.20 * signals.time_preciousness
        )

    def _emotional_layer_threshold(self, relationship: Relationship) -> float:
        if relationship.stage == RelationshipStage.EXPERIMENTING:
            return 0.35
        return 0.45

    def _stage_encoding_strategy(
        self,
        relationship: Relationship,
        signals: TurnSignals,
        memory_type: MemoryType,
        context_tag: ContextTag,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if (
            relationship.stage == RelationshipStage.BONDING
            and context_tag == ContextTag.VULNERABLE_MOMENT
            and signals.self_disclosure_depth >= 0.70
            and signals.emotion_intensity >= 0.60
        ):
            metadata["stage_encoding"] = {
                "strategy": "bonding_vulnerable_auto_milestone",
                "original_memory_type": memory_type.value,
                "original_context_tag": context_tag.value,
                "self_disclosure_depth": signals.self_disclosure_depth,
                "emotion_intensity": signals.emotion_intensity,
            }
            memory_type = MemoryType.MILESTONE
            context_tag = ContextTag.MILESTONE
        return {"memory_type": memory_type, "context_tag": context_tag, "metadata": metadata}

    def _ai_provider_name(self) -> str:
        return ai_provider_name(self.ai)

    def _log_ai_decision(
        self,
        relationship_id: str,
        *,
        task: str,
        input_summary: dict[str, Any],
        output_summary: dict[str, Any],
        now: datetime | None = None,
    ) -> None:
        now = now or utcnow()
        entry = {
            "type": "ai_decision",
            "relationship_id": relationship_id,
            "provider": self._ai_provider_name(),
            "task": task,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "at": now.isoformat(),
        }
        ai_call = consume_ai_call_metadata(self.ai)
        if ai_call:
            entry["ai_call"] = ai_call
        self.ai_decision_log.append(entry)
        return entry

    def _base_weight(self, score: float, memory_type: MemoryType) -> float:
        if memory_type in {MemoryType.MILESTONE, MemoryType.IDENTITY}:
            return 0.95
        if score >= 0.70:
            return 0.80
        if score >= 0.40:
            return 0.65
        return 0.50

    def _decay_curve_for(self, memory_type: MemoryType, relationship: Relationship) -> DecayCurve:
        if not relationship.preferences.reverse_decay_enabled:
            return DecayCurve.STANDARD_POWER_LAW
        if memory_type in {
            MemoryType.MILESTONE,
            MemoryType.COMMITMENT,
            MemoryType.EMOTIONAL_MOMENT,
            MemoryType.SHARED_EPISODE,
            MemoryType.INSIDE_JOKE,
            MemoryType.CONFLICT,
        }:
            return DecayCurve.REVERSE_DECAY
        return DecayCurve.HYBRID if memory_type == MemoryType.FACT else DecayCurve.STANDARD_POWER_LAW

    def build_migration_certificate(
        self,
        turns: list[dict[str, Any] | str],
        *,
        default_user: str = "user",
        default_ai: str = "companion",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        normalized = sorted(
            [self._normalize_legacy_turn(item, default_user, default_ai, now) for item in turns],
            key=lambda item: item["timestamp"],
        )
        return self._build_migration_certificate(normalized, now=now)

    def _build_migration_certificate(self, normalized_turns: list[dict[str, Any]], *, now: datetime) -> dict[str, Any]:
        records = [
            {
                "user": item["user"],
                "ai": item["ai"],
                "timestamp": item["timestamp"].isoformat(),
                "text_sealed": self._seal_audit_text(item["text"]),
                "milestone": item["milestone"],
            }
            for item in normalized_turns
        ]
        canonical = json.dumps(records, ensure_ascii=False, sort_keys=True)
        return {
            "schema": "relationship-migration-certificate-v1",
            "issued_at": now.isoformat(),
            "turn_count": len(records),
            "relationship_ids": sorted({f"{item['user']}:{item['ai']}" for item in normalized_turns}),
            "source_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "content_retained": False,
            "records": records,
        }

    def _verify_migration_certificate(
        self,
        expected: dict[str, Any],
        provided: dict[str, Any] | None,
        *,
        require_certificate: bool,
        now: datetime,
    ) -> dict[str, Any]:
        if provided is None:
            if require_certificate:
                raise ValueError("relationship migration certificate is required")
            return {"status": "SELF_ISSUED_UNVERIFIED", "verified": False}
        if provided.get("schema") != expected["schema"]:
            raise ValueError("relationship migration certificate schema mismatch")
        if provided.get("source_digest") != expected["source_digest"]:
            raise ValueError("relationship migration certificate digest mismatch")
        if provided.get("relationship_ids") != expected["relationship_ids"]:
            raise ValueError("relationship migration certificate relationship mismatch")
        if int(provided.get("turn_count", -1)) != int(expected["turn_count"]):
            raise ValueError("relationship migration certificate turn count mismatch")
        return {"status": "VERIFIED", "verified": True, "verified_at": now.isoformat()}

    def _normalize_legacy_turn(
        self, item: dict[str, Any] | str, default_user: str, default_ai: str, now: datetime
    ) -> dict[str, Any]:
        if isinstance(item, str):
            return {
                "text": item,
                "user": default_user,
                "ai": default_ai,
                "timestamp": now,
                "milestone": False,
                "metadata": {},
            }
        timestamp = item.get("timestamp") or item.get("created_at") or item.get("time")
        return {
            "text": str(item["text"]),
            "user": str(item.get("user") or item.get("user_id") or default_user),
            "ai": str(item.get("ai") or item.get("ai_id") or default_ai),
            "timestamp": _dt(timestamp) if timestamp else now,
            "milestone": bool(item.get("milestone") or item.get("mark_milestone")),
            "metadata": dict(item.get("metadata") or {}),
        }

    def _create_emotional_memory(
        self, relationship: Relationship, memory: MemoryRecord, signals: TurnSignals, now: datetime
    ) -> EmotionalMemory:
        primary = "joy" if signals.sentiment > 0 else "sadness" if signals.sentiment < 0 else "nostalgia"
        if signals.context_tag == ContextTag.VULNERABLE_MOMENT:
            primary = "vulnerability"
        return EmotionalMemory(
            emotion_id=new_id("emo"),
            relationship_id=relationship.relationship_id,
            source_memory_id=memory.memory_id,
            content=memory.content,
            timestamp=now,
            relationship_age_at_creation=relationship.relationship_age,
            emotions=[EmotionLabel(primary, signals.emotion_intensity)],
            primary_emotion=primary,
            emotional_valence=signals.sentiment,
            emotional_arousal=signals.arousal,
            personal_importance=signals.personal_importance,
            self_disclosure_depth=signals.self_disclosure_depth,
            context_tag=signals.context_tag,
            relationship_stage_at_creation=relationship.stage,
            trust_level_at_creation=relationship.trust_level,
            embeddings=memory.metadata.get("embeddings") or self._memory_embedding_features(memory, signals),
        )

    def _attach_relationship_indexes(self, relationship: Relationship, memory: MemoryRecord, signals: TurnSignals) -> None:
        if memory.memory_type in {MemoryType.SHARED_EPISODE, MemoryType.EMOTIONAL_MOMENT}:
            relationship.shared_episodes.append(memory.memory_id)
        if memory.memory_type == MemoryType.MILESTONE:
            if not memory.metadata.get("milestone_confirmation"):
                self._set_milestone_confirmation(memory, "PENDING", reason="auto_detected")
            relationship.milestones.append(memory.memory_id)
            relationship.shared_episodes.append(memory.memory_id)
        if memory.memory_type == MemoryType.INSIDE_JOKE:
            if memory.memory_id not in relationship.inside_jokes:
                relationship.inside_jokes.append(memory.memory_id)
        elif signals.inside_joke_candidate:
            self._track_inside_joke_candidate(relationship, memory, signals.inside_joke_candidate)
        if signals.unresolved_thread:
            relationship.unresolved_threads.append(memory.memory_id)

    def _maybe_promote_major_shared_decision(
        self,
        relationship: Relationship,
        memory: MemoryRecord,
        signals: TurnSignals,
        now: datetime,
    ) -> None:
        if memory.memory_type == MemoryType.MILESTONE:
            return
        if not self._is_major_shared_decision(memory.content):
            return
        memory.metadata["major_shared_decision"] = {
            "original_memory_type": memory.memory_type.value,
            "original_context_tag": memory.context_tag.value,
            "relationship_depth": signals.relationship_depth,
            "personal_importance": signals.personal_importance,
        }
        memory.memory_type = MemoryType.MILESTONE
        memory.context_tag = ContextTag.MILESTONE
        memory.decay_curve = DecayCurve.PERMANENT
        memory.base_weight = max(memory.base_weight, 0.95)
        memory.importance = max(memory.importance, 0.95)
        memory.storage_layer = MemoryLayer.L5_RELATIONSHIP_HISTORY
        memory.metadata["embeddings"] = self._memory_embedding_features(memory, signals)
        self._set_milestone_confirmation(memory, "PENDING", reason="major_shared_decision", now=now)
        self.deviation_log.append(
            {
                "type": "major_shared_decision_milestone_created",
                "relationship_id": relationship.relationship_id,
                "memory_id": memory.memory_id,
                "at": now.isoformat(),
            }
        )

    def _is_major_shared_decision(self, text: str) -> bool:
        shared_markers = ["我们", "咱们", "一起", "共同", "咱俩"]
        decision_markers = ["决定", "约定", "承诺"]
        if any(f"{shared}{decision}" in text for shared in shared_markers for decision in decision_markers):
            return True
        return any(shared in text for shared in shared_markers) and any(decision in text for decision in decision_markers)

    def _maybe_promote_shared_celebration(
        self,
        relationship: Relationship,
        memory: MemoryRecord,
        signals: TurnSignals,
        now: datetime,
    ) -> None:
        if memory.memory_type == MemoryType.MILESTONE:
            return
        if memory.context_tag != ContextTag.SHARED_CELEBRATION:
            return
        if not self._has_shared_marker(memory.content):
            return
        memory.metadata["shared_celebration_milestone"] = {
            "original_memory_type": memory.memory_type.value,
            "original_context_tag": memory.context_tag.value,
            "emotion_intensity": signals.emotion_intensity,
            "sentiment": signals.sentiment,
        }
        memory.memory_type = MemoryType.MILESTONE
        memory.context_tag = ContextTag.MILESTONE
        memory.decay_curve = DecayCurve.PERMANENT
        memory.base_weight = max(memory.base_weight, 0.95)
        memory.importance = max(memory.importance, 0.95)
        memory.storage_layer = MemoryLayer.L5_RELATIONSHIP_HISTORY
        memory.metadata["embeddings"] = self._memory_embedding_features(memory, signals)
        self._set_milestone_confirmation(memory, "PENDING", reason="shared_celebration", now=now)
        self.deviation_log.append(
            {
                "type": "shared_celebration_milestone_created",
                "relationship_id": relationship.relationship_id,
                "memory_id": memory.memory_id,
                "at": now.isoformat(),
            }
        )

    def _has_shared_marker(self, text: str) -> bool:
        return any(marker in text for marker in ["我们", "咱们", "一起", "共同", "咱俩"])

    def _refresh_relationship_indexes_for_memory(self, relationship: Relationship, memory: MemoryRecord) -> None:
        memory_id = memory.memory_id
        for bucket in (relationship.milestones, relationship.shared_episodes, relationship.inside_jokes, relationship.unresolved_threads):
            while memory_id in bucket:
                bucket.remove(memory_id)
        if memory.memory_type == MemoryType.MILESTONE or memory.context_tag == ContextTag.MILESTONE:
            relationship.milestones.append(memory_id)
        if memory.memory_type in {
            MemoryType.MILESTONE,
            MemoryType.SHARED_EPISODE,
            MemoryType.EMOTIONAL_MOMENT,
            MemoryType.INSIDE_JOKE,
        }:
            relationship.shared_episodes.append(memory_id)
        if memory.memory_type == MemoryType.INSIDE_JOKE or memory.context_tag == ContextTag.INSIDE_JOKE:
            relationship.inside_jokes.append(memory_id)
        if memory.memory_type == MemoryType.COMMITMENT or memory.context_tag == ContextTag.UNRESOLVED_THREAD:
            relationship.unresolved_threads.append(memory_id)

    def _delete_memory_and_derivatives(self, memory_id: str, *, now: datetime, reason: str) -> bool:
        memory = self.memories.pop(memory_id, None)
        if not memory:
            return False
        relationship = self.relationships.get(memory.relationship_id)
        if relationship:
            for bucket in (
                relationship.milestones,
                relationship.shared_episodes,
                relationship.inside_jokes,
                relationship.unresolved_threads,
            ):
                while memory_id in bucket:
                    bucket.remove(memory_id)
            for candidate in relationship.inside_joke_candidates.values():
                candidate["memory_ids"] = [mid for mid in candidate.get("memory_ids", []) if mid != memory_id]
                if candidate.get("promoted_memory_id") == memory_id:
                    candidate["promoted_memory_id"] = None

        for emotion in list(self.emotional_memories.values()):
            if emotion.source_memory_id == memory_id:
                del self.emotional_memories[emotion.emotion_id]

        for story in list(self.story_nodes.values()):
            changed = False
            if memory_id in story.core_events:
                story.core_events = [mid for mid in story.core_events if mid != memory_id]
                changed = True
            if memory_id in story.key_moments:
                story.key_moments = [mid for mid in story.key_moments if mid != memory_id]
                changed = True
            if memory_id in story.child_inside_jokes:
                story.child_inside_jokes = [mid for mid in story.child_inside_jokes if mid != memory_id]
                changed = True
            if not changed:
                continue
            if not story.core_events:
                del self.story_nodes[story.story_id]
            else:
                story.consistency_score = clamp(story.consistency_score * 0.5)
                provenance = story.consensus_provenance if isinstance(story.consensus_provenance, dict) else {}
                provenance["has_deleted_source"] = True
                provenance["requires_schema_rebuild"] = True
                provenance["deleted_source_count"] = int(provenance.get("deleted_source_count", 0) or 0) + 1
                provenance["deleted_source_at"] = now.isoformat()
                provenance["memory_ids"] = list(story.core_events)
                story.consensus_provenance = provenance
                story.conflict_versions.append(
                    {
                        "at": now.isoformat(),
                        "memory_id": memory_id,
                        "reason": "source_deleted",
                        "delete_reason_sealed": self._seal_audit_text(reason),
                    }
                )
                story.user_framing = "SOURCE_DELETED"
                self.deviation_log.append(
                    {
                        "type": "story_marked_deleted_source",
                        "relationship_id": story.relationship_id,
                        "story_id": story.story_id,
                        "deleted_memory_id": memory_id,
                        "remaining_sources": len(story.core_events),
                        "at": now.isoformat(),
                    }
                )

        for reminder in list(self.commitment_reminders.values()):
            if reminder.memory_id == memory_id:
                reminder.status = ReminderStatus.ARCHIVED
                reminder.archived_at = now

        for identity in list(self.core_identity.values()):
            if identity.memory_id == memory_id:
                if relationship and identity.identity_id in relationship.core_identity:
                    relationship.core_identity.remove(identity.identity_id)
                del self.core_identity[identity.identity_id]

        for request in self.core_identity_delete_requests.values():
            if request.memory_id == memory_id and request.status == ResetRequestStatus.PENDING:
                request.status = ResetRequestStatus.CANCELLED
        for edge in list(self.memory_graph_edges.values()):
            if edge.source_memory_id == memory_id or edge.target_memory_id == memory_id:
                del self.memory_graph_edges[edge.edge_id]
        return True

    def _critical_memory_tombstone_preview(
        self,
        memory_id: str,
        delete_reason: str,
        now: datetime,
    ) -> dict[str, Any] | None:
        memory = self.memories.get(memory_id)
        if not memory:
            return None
        criticality = str(memory.metadata.get("criticality") or memory.metadata.get("severity") or "").upper()
        if criticality not in {"CRITICAL", "SAFETY", "MEDICAL", "MAJOR_COMMITMENT"}:
            return None
        reasons = [
            str(item)
            for item in memory.metadata.get("criticality_reasons", [])
            if str(item) in {"safety", "medical", "major_commitment"}
        ]
        return {
            "schema": "critical-memory-tombstone-v1",
            "memory_id_hash": self._seal_audit_text(memory.memory_id),
            "relationship_id": memory.relationship_id,
            "criticality": criticality,
            "reason_categories": reasons,
            "memory_type": memory.memory_type.value,
            "context_tag": memory.context_tag.value,
            "created_at": now.isoformat(),
            "delete_reason_sealed": self._seal_audit_text(delete_reason),
            "content_retained": False,
            "plaintext_content_retained": False,
            "future_remention_guidance": "treat_as_sensitive_remention_not_plain_new_fact",
        }

    def _mark_critical_tombstone_remention(
        self,
        relationship: Relationship,
        memory: MemoryRecord,
        now: datetime,
    ) -> None:
        match = self._matching_critical_tombstone(relationship.relationship_id, memory)
        if not match:
            return
        marker = {
            "matched": True,
            "matched_at": now.isoformat(),
            "tombstone_schema": match.get("schema"),
            "criticality": match.get("criticality"),
            "reason_categories": match.get("reason_categories", []),
            "previous_memory_id_hash": match.get("memory_id_hash"),
            "requires_user_confirmation": True,
            "content_retained_from_deleted_memory": False,
            "guidance": match.get("future_remention_guidance"),
        }
        memory.metadata["critical_tombstone_remention"] = marker
        marker["recording_status"] = "PENDING_USER_CONFIRMATION"
        metacognition = self._ensure_memory_metacognition(memory, now)
        metacognition["needs_clarification"] = True
        metacognition["critical_tombstone_remention"] = True
        metacognition["pending_sensitive_remention_confirmation"] = True
        metacognition["uncertainty_action"] = "confirm_sensitive_remention"
        self.deviation_log.append(
            {
                "type": "critical_tombstone_rementioned",
                "relationship_id": relationship.relationship_id,
                "memory_id": memory.memory_id,
                "criticality": match.get("criticality"),
                "reason_categories": match.get("reason_categories", []),
                "at": now.isoformat(),
            }
        )
        self._health_alert(
            relationship.relationship_id,
            "critical_tombstone_rementioned",
            HealthRiskLevel.INFO,
            "用户重新提及曾删除过的安全/医疗/重大承诺类信息；请先确认是否要重新记录，并避免把它静默当作普通新事实。",
            now,
            source_memory_id=memory.memory_id,
        )

    def _matching_critical_tombstone(self, relationship_id: str, memory: MemoryRecord) -> dict[str, Any] | None:
        criticality = str(memory.metadata.get("criticality") or memory.metadata.get("severity") or "").upper()
        if criticality not in {"CRITICAL", "SAFETY", "MEDICAL", "MAJOR_COMMITMENT"}:
            return None
        reasons = set(str(item) for item in memory.metadata.get("criticality_reasons", []))
        for record in reversed(self.deletion_compliance_log):
            if record.get("relationship_id") != relationship_id:
                continue
            summary = record.get("summary")
            if not isinstance(summary, dict):
                continue
            tombstone = summary.get("critical_memory_tombstone")
            if not isinstance(tombstone, dict):
                continue
            tombstone_criticality = str(tombstone.get("criticality") or "").upper()
            tombstone_reasons = set(str(item) for item in tombstone.get("reason_categories", []))
            if tombstone_criticality == criticality or tombstone_reasons.intersection(reasons):
                return tombstone
        return None

    def _update_memory_graph(
        self,
        relationship: Relationship,
        memory: MemoryRecord,
        signals: TurnSignals,
        now: datetime,
    ) -> None:
        themes = set(self._themes(memory.content))
        recent = sorted(
            [
                item
                for item in self.memories.values()
                if item.relationship_id == relationship.relationship_id and item.memory_id != memory.memory_id
            ],
            key=lambda item: item.created_at,
            reverse=True,
        )[:30]
        for other in recent:
            other_themes = set(self._themes(other.content))
            shared_themes = sorted(themes.intersection(other_themes))
            if shared_themes:
                self._upsert_memory_graph_edge(
                    relationship.relationship_id,
                    memory.memory_id,
                    other.memory_id,
                    "shared_theme",
                    0.40 + min(0.30, 0.10 * len(shared_themes)),
                    now,
                    {"themes": shared_themes},
                )
            if memory.context_tag == other.context_tag and memory.context_tag != ContextTag.GENERAL:
                self._upsert_memory_graph_edge(
                    relationship.relationship_id,
                    memory.memory_id,
                    other.memory_id,
                    "shared_context",
                    0.55,
                    now,
                    {"context_tag": memory.context_tag.value},
                )

        for story in self.story_nodes.values():
            if story.relationship_id != relationship.relationship_id or memory.memory_id not in story.core_events:
                continue
            for other_id in story.core_events:
                if other_id != memory.memory_id and other_id in self.memories:
                    self._upsert_memory_graph_edge(
                        relationship.relationship_id,
                        memory.memory_id,
                        other_id,
                        "same_story",
                        0.75,
                        now,
                        {"story_id": story.story_id},
                    )

        if signals.unresolved_thread:
            for reminder in self.commitment_reminders.values():
                if reminder.relationship_id == relationship.relationship_id and reminder.memory_id in self.memories:
                    self._upsert_memory_graph_edge(
                        relationship.relationship_id,
                        memory.memory_id,
                        reminder.memory_id,
                        "commitment_thread",
                        0.65,
                        now,
                        {"reminder_id": reminder.reminder_id},
                    )

    def _upsert_memory_graph_edge(
        self,
        relationship_id: str,
        source_memory_id: str,
        target_memory_id: str,
        relation_type: str,
        weight: float,
        now: datetime,
        evidence: dict[str, Any],
    ) -> None:
        if source_memory_id == target_memory_id:
            return
        left, right = sorted([source_memory_id, target_memory_id])
        edge_key = f"{relationship_id}:{left}:{right}:{relation_type}"
        existing = self.memory_graph_edges.get(edge_key)
        if existing:
            existing.weight = max(existing.weight, clamp(weight))
            existing.evidence.update(evidence)
            return
        self.memory_graph_edges[edge_key] = MemoryGraphEdge(
            edge_id=edge_key,
            relationship_id=relationship_id,
            source_memory_id=left,
            target_memory_id=right,
            relation_type=relation_type,
            weight=clamp(weight),
            created_at=now,
            evidence=evidence,
        )

    def _apply_graph_retrieval_boost(self, results: list[RetrievalResult]) -> None:
        if not results:
            return
        by_id = {item.memory.memory_id: item for item in results}
        seeds = sorted(results, key=lambda item: item.score, reverse=True)[:1]
        seed_ids = {item.memory.memory_id for item in seeds if item.score > 0}
        if not seed_ids:
            return
        boost_by_id: dict[str, float] = {}
        evidence_by_id: dict[str, list[dict[str, Any]]] = {}
        for edge in self.memory_graph_edges.values():
            source_is_seed = edge.source_memory_id in seed_ids and edge.target_memory_id in by_id
            target_is_seed = edge.target_memory_id in seed_ids and edge.source_memory_id in by_id
            if not source_is_seed and not target_is_seed:
                continue
            neighbor_id = edge.target_memory_id if source_is_seed else edge.source_memory_id
            if neighbor_id in seed_ids:
                continue
            boost = min(0.18, edge.weight * 0.12)
            boost_by_id[neighbor_id] = max(boost_by_id.get(neighbor_id, 0.0), boost)
            evidence_by_id.setdefault(neighbor_id, []).append(
                {
                    "edge_id": edge.edge_id,
                    "relation_type": edge.relation_type,
                    "weight": edge.weight,
                    "seed_memory_id": edge.source_memory_id if source_is_seed else edge.target_memory_id,
                    "evidence": edge.evidence,
                    "inferred": True,
                    "uncertainty_action": "confirm_gently",
                }
            )
        for memory_id, boost in boost_by_id.items():
            result = by_id[memory_id]
            result.score = clamp(result.score + boost)
            result.explanation["graph_boost"] = boost
            result.explanation["graph_neighbors"] = evidence_by_id.get(memory_id, [])[:5]
            result.explanation["final_score"] = result.score
        for result in results:
            result.explanation.setdefault("graph_boost", 0.0)
            result.explanation.setdefault("graph_neighbors", [])

    def _apply_retrieval_adaptation(
        self,
        relationship: Relationship,
        query: str,
        candidates: list[RetrievalResult],
        ranked: list[RetrievalResult],
        now: datetime,
    ) -> dict[str, Any]:
        if not ranked:
            return {"reinforced": [], "suppressed_competitors": [], "reason": "no_retrieval_results"}
        winner_ids = {item.memory.memory_id for item in ranked}
        reinforced: list[dict[str, Any]] = []
        for item in ranked:
            memory = item.memory
            before_weight = memory.base_weight
            before_mentions = memory.mention_count
            memory.mention_count += 1
            memory.updated_at = now
            memory.base_weight = clamp(memory.base_weight + min(0.03, 0.01 + item.score * 0.02), high=1.0)
            state = memory.metadata.setdefault("retrieval_reinforcement", {})
            state["count"] = int(state.get("count", 0) or 0) + 1
            state["last_retrieved_at"] = now.isoformat()
            state["last_query"] = query[:120]
            reinforced.append(
                {
                    "memory_id": memory.memory_id,
                    "mention_count_before": before_mentions,
                    "mention_count_after": memory.mention_count,
                    "base_weight_before": before_weight,
                    "base_weight_after": memory.base_weight,
                }
            )

        winner_floor = min((item.score for item in ranked), default=0.0)
        suppressed: list[dict[str, Any]] = []
        for item in sorted(candidates, key=lambda candidate: candidate.score, reverse=True):
            memory = item.memory
            if memory.memory_id in winner_ids:
                continue
            if len(suppressed) >= 8:
                break
            if item.score <= 0 or item.score < winner_floor * 0.35:
                continue
            similarity = max(
                float(item.explanation.get("semantic", 0.0) or 0.0),
                float(item.explanation.get("lexical_similarity", 0.0) or 0.0),
            )
            if similarity < 0.18:
                continue
            if memory.decay_curve == DecayCurve.PERMANENT or is_trust_bias_protected(memory):
                continue
            before_weight = memory.base_weight
            memory.base_weight = max(0.05, memory.base_weight * 0.97)
            memory.updated_at = now
            state = memory.metadata.setdefault("retrieval_induced_forgetting", {})
            state["count"] = int(state.get("count", 0) or 0) + 1
            state["last_competed_at"] = now.isoformat()
            state["last_query"] = query[:120]
            state["winner_memory_ids"] = sorted(winner_ids)[:5]
            state["last_similarity"] = similarity
            state["suppression_factor"] = 0.97
            suppressed.append(
                {
                    "memory_id": memory.memory_id,
                    "base_weight_before": before_weight,
                    "base_weight_after": memory.base_weight,
                    "similarity": similarity,
                }
            )
        event = {
            "type": "retrieval_adaptation",
            "relationship_id": relationship.relationship_id,
            "query": query[:120],
            "reinforced": reinforced,
            "suppressed_competitors": suppressed,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return {
            "reinforced": reinforced,
            "suppressed_competitors": suppressed,
            "reason": "retrieval_induced_reinforcement_and_forgetting",
        }

    def _association_expansions(
        self,
        relationship: Relationship,
        ranked: list[RetrievalResult],
        now: datetime,
    ) -> list[dict[str, Any]]:
        if not ranked:
            return []
        seed_ids = {item.memory.memory_id for item in ranked[:1]}
        expansions: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in ranked[:2]:
            memory = item.memory
            current_of = [memory_id for memory_id in memory.metadata.get("current_version_of", []) if memory_id in self.memories]
            if current_of:
                version = next(
                    (
                        entry
                        for entry in reversed(memory.metadata.get("version_tree", []))
                        if isinstance(entry, dict) and entry.get("status") == "ACTIVE_HEAD"
                    ),
                    {},
                )
                subject = str(version.get("subject") or "这个偏好")
                expansions.append(
                    {
                        "type": "preference_evolution",
                        "memory_id": memory.memory_id,
                        "previous_memory_ids": current_of,
                        "confidence": 0.82,
                        "text": f"这个偏好可能在演变：当前版本是关于「{subject}」的较新表达。",
                        "inferred": True,
                        "evidence": {
                            "version_tree_status": "ACTIVE_HEAD",
                            "reason": version.get("reason", "preference_polarity_changed"),
                            "previous_memory_count": len(current_of),
                        },
                    }
                )
                seen.update(current_of)
            supersession = memory.metadata.get("supersession")
            if isinstance(supersession, dict) and supersession.get("superseded_by") in self.memories:
                newer_id = supersession["superseded_by"]
                subject = str(supersession.get("subject") or "这个偏好")
                expansions.append(
                    {
                        "type": "preference_evolution",
                        "memory_id": memory.memory_id,
                        "current_memory_id": newer_id,
                        "confidence": 0.82,
                        "text": f"这条偏好已有较新版本：关于「{subject}」的表达可能已经变化。",
                        "inferred": True,
                        "evidence": {
                            "version_tree_status": "SUPERSEDED",
                            "reason": supersession.get("reason", "preference_polarity_changed"),
                            "superseded_at": supersession.get("superseded_at"),
                        },
                    }
                )
                seen.add(newer_id)
            if len(expansions) >= 2:
                break
        for edge in sorted(self.memory_graph_edges.values(), key=lambda item: item.weight, reverse=True):
            if edge.relationship_id != relationship.relationship_id:
                continue
            if edge.source_memory_id in seed_ids:
                neighbor_id = edge.target_memory_id
                seed_id = edge.source_memory_id
            elif edge.target_memory_id in seed_ids:
                neighbor_id = edge.source_memory_id
                seed_id = edge.target_memory_id
            else:
                continue
            if neighbor_id in seed_ids or neighbor_id in seen:
                continue
            memory = self.memories.get(neighbor_id)
            if not memory or self._memory_is_recall_suppressed(memory) or memory.metadata.get("archived"):
                continue
            confidence = min(0.95, 0.50 + edge.weight * 0.35 + float(self._ensure_memory_metacognition(memory, now).get("confidence", 0.0)) * 0.20)
            if confidence < 0.70:
                continue
            expansions.append(
                {
                    "type": "graph_neighbor",
                    "memory_id": neighbor_id,
                    "seed_memory_id": seed_id,
                    "relation_type": edge.relation_type,
                    "confidence": confidence,
                    "text": f"可能相关的边缘线索：{memory.content[:60]}",
                    "inferred": True,
                    "evidence": edge.evidence,
                }
            )
            seen.add(neighbor_id)
            if len(expansions) >= 3:
                break
        for topic in relationship.implicit_topics:
            if topic.get("status") not in {"ACTIVE", "CONFIRMED"}:
                continue
            source_ids = set(topic.get("source_memory_ids", []))
            if not source_ids.intersection(seed_ids):
                continue
            confidence = float(topic.get("confidence", 0.0))
            if confidence < 0.70:
                continue
            evidence = self._implicit_topic_evidence_status(relationship, topic, now)
            if not evidence["valid"]:
                self._fail_implicit_topic_evidence(relationship, topic, evidence, now)
                continue
            expansions.append(
                {
                    "type": "implicit_topic",
                    "topic_id": topic.get("topic_id"),
                    "confidence": confidence,
                    "text": f"根据相关碎片，可能还关联到「{topic.get('summary', '')[:60]}」",
                    "inferred": True,
                    "source_memory_ids": topic.get("source_memory_ids", []),
                    "evidence": evidence,
                }
            )
            if len(expansions) >= 4:
                break
        return expansions[:4]

    def _story_clusters_for_results(
        self,
        relationship: Relationship,
        ranked: list[RetrievalResult],
    ) -> list[dict[str, Any]]:
        if not ranked:
            return []
        ranked_ids = [item.memory.memory_id for item in ranked]
        ranked_id_set = set(ranked_ids)
        result_score_by_id = {item.memory.memory_id: item.score for item in ranked}
        clusters: list[dict[str, Any]] = []
        for story in self.story_nodes.values():
            if story.relationship_id != relationship.relationship_id:
                continue
            matched_ids = [memory_id for memory_id in ranked_ids if memory_id in story.core_events]
            if not matched_ids:
                continue
            related_ids = [
                memory_id
                for memory_id in story.core_events
                if memory_id not in ranked_id_set and memory_id in self.memories and not self._memory_is_recall_suppressed(self.memories[memory_id])
            ][:5]
            cluster = {
                "type": "shared_story",
                "story_id": story.story_id,
                "title": story.title,
                "level": story.narrative_level.value,
                "consensus": story.consensus_version,
                "consensus_status": story.consensus_status,
                "consensus_provenance": story.consensus_provenance,
                "themes": story.recurring_themes,
                "matched_memory_ids": matched_ids,
                "related_memory_ids": related_ids,
                "event_count": len(story.core_events),
                "retell_count": story.retell_count,
                "consistency_score": story.consistency_score,
                "user_framing": story.user_framing,
                "score": max(result_score_by_id[memory_id] for memory_id in matched_ids),
                "inferred": False,
                "aggregation": "SharedStoryNode",
            }
            clusters.append(cluster)
            for memory_id in matched_ids:
                result = next((item for item in ranked if item.memory.memory_id == memory_id), None)
                if result:
                    result.explanation["story_cluster"] = {
                        "story_id": story.story_id,
                        "title": story.title,
                        "level": story.narrative_level.value,
                        "matched_memory_ids": matched_ids,
                        "aggregation": "SharedStoryNode",
                    }
        return sorted(clusters, key=lambda item: item["score"], reverse=True)[:3]

    def _audit_integrity(self, relationship_ids: list[str]) -> dict[str, Any]:
        relationship_id_set = set(relationship_ids)
        orphan_references: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        unresolved_time_conflicts: dict[str, dict[str, Any]] = {}
        for relationship in [item for item in self.relationships.values() if item.relationship_id in relationship_id_set]:
            buckets = {
                "milestones": relationship.milestones,
                "shared_episodes": relationship.shared_episodes,
                "inside_jokes": relationship.inside_jokes,
                "unresolved_threads": relationship.unresolved_threads,
            }
            for bucket_name, memory_ids in buckets.items():
                for memory_id in memory_ids:
                    if memory_id not in self.memories:
                        orphan_references.append(
                            {
                                "source": "relationship",
                                "relationship_id": relationship.relationship_id,
                                "field": bucket_name,
                                "memory_id": memory_id,
                            }
                        )
            for identity_id in relationship.core_identity:
                if identity_id not in self.core_identity:
                    orphan_references.append(
                        {
                            "source": "relationship",
                            "relationship_id": relationship.relationship_id,
                            "field": "core_identity",
                            "identity_id": identity_id,
                        }
                    )

        for emotion in self.emotional_memories.values():
            if emotion.relationship_id not in relationship_id_set:
                continue
            if emotion.source_memory_id not in self.memories:
                orphan_references.append(
                    {
                        "source": "emotional_memory",
                        "emotion_id": emotion.emotion_id,
                        "memory_id": emotion.source_memory_id,
                    }
                )

        for story in self.story_nodes.values():
            if story.relationship_id not in relationship_id_set:
                continue
            for memory_id in story.core_events + story.key_moments:
                if memory_id not in self.memories:
                    orphan_references.append(
                        {
                            "source": "story_node",
                            "story_id": story.story_id,
                            "memory_id": memory_id,
                        }
                    )

        for reminder in self.commitment_reminders.values():
            if reminder.relationship_id not in relationship_id_set:
                continue
            if reminder.status != ReminderStatus.ARCHIVED and reminder.memory_id not in self.memories:
                orphan_references.append(
                    {
                        "source": "commitment_reminder",
                        "reminder_id": reminder.reminder_id,
                        "memory_id": reminder.memory_id,
                    }
                )
            if reminder.status in {ReminderStatus.PENDING, ReminderStatus.REMINDER_SENT} and (utcnow() - reminder.due_at).days > 30:
                warnings.append(
                    {
                        "type": "stale_commitment_reminder",
                        "reminder_id": reminder.reminder_id,
                        "due_at": reminder.due_at.isoformat(),
                    }
                )

        for identity in self.core_identity.values():
            if identity.relationship_id not in relationship_id_set:
                continue
            if identity.memory_id not in self.memories:
                orphan_references.append(
                    {
                        "source": "core_identity",
                        "identity_id": identity.identity_id,
                        "memory_id": identity.memory_id,
                    }
                )
            if len(identity.replicas) < 3 or not all(replica.get("sealed") for replica in identity.replicas):
                warnings.append(
                    {
                        "type": "l4_replica_protection_incomplete",
                        "identity_id": identity.identity_id,
                        "replica_count": len(identity.replicas),
                    }
                )

        for edge in self.memory_graph_edges.values():
            if edge.relationship_id not in relationship_id_set:
                continue
            missing = [
                memory_id
                for memory_id in (edge.source_memory_id, edge.target_memory_id)
                if memory_id not in self.memories
            ]
            for memory_id in missing:
                orphan_references.append(
                    {
                        "source": "memory_graph_edge",
                        "edge_id": edge.edge_id,
                        "memory_id": memory_id,
                    }
                )

        for memory in self.memories.values():
            if memory.relationship_id not in relationship_id_set:
                continue
            for conflict in self._active_time_conflicts(memory):
                conflict_id = str(conflict.get("conflict_id", "unknown"))
                entry = unresolved_time_conflicts.setdefault(
                    conflict_id,
                    {
                        "type": "unresolved_time_conflict",
                        "conflict_id": conflict_id,
                        "relationship_id": memory.relationship_id,
                        "memory_ids": [],
                        "detected_at": conflict.get("detected_at"),
                        "gap_days": conflict.get("gap_days"),
                        "status": conflict.get("status"),
                    },
                )
                if memory.memory_id not in entry["memory_ids"]:
                    entry["memory_ids"].append(memory.memory_id)
                conflicting_memory_id = conflict.get("conflicting_memory_id")
                if conflicting_memory_id and conflicting_memory_id not in entry["memory_ids"]:
                    entry["memory_ids"].append(conflicting_memory_id)

        pending_too_long = [
            request.request_id
            for request in self.memory_delete_requests.values()
            if request.relationship_id in relationship_id_set
            and request.status == ResetRequestStatus.PENDING
            and (utcnow() - request.execute_after).days > 30
        ]
        for request_id in pending_too_long:
            warnings.append({"type": "stale_memory_delete_request", "request_id": request_id})

        stale_resets = [
            request.request_id
            for request in self.reset_requests.values()
            if request.relationship_id in relationship_id_set
            and request.status == ResetRequestStatus.PENDING
            and (utcnow() - request.execute_after).days > 30
        ]
        for request_id in stale_resets:
            warnings.append({"type": "stale_reset_request", "request_id": request_id})

        stale_l4_deletes = [
            request.request_id
            for request in self.core_identity_delete_requests.values()
            if request.relationship_id in relationship_id_set
            and request.status == ResetRequestStatus.PENDING
            and (utcnow() - request.execute_after).days > 30
        ]
        for request_id in stale_l4_deletes:
            warnings.append({"type": "stale_l4_delete_request", "request_id": request_id})

        return {
            "orphan_references": orphan_references,
            "unresolved_time_conflicts": list(unresolved_time_conflicts.values()),
            "warnings": warnings,
        }

    def _maybe_create_commitment_reminder(
        self,
        relationship: Relationship,
        memory: MemoryRecord,
        text: str,
        now: datetime,
    ) -> None:
        due_at = self._extract_commitment_due_at(text, now)
        if not due_at:
            return
        has_commitment_signal = (
            memory.memory_type == MemoryType.COMMITMENT
            or memory.context_tag == ContextTag.UNRESOLVED_THREAD
            or any(word in text for word in ["答应", "约定", "承诺", "提醒", "别忘", "deadline", "截止"])
        )
        if not has_commitment_signal:
            return
        reminder = CommitmentReminder(
            reminder_id=new_id("reminder"),
            relationship_id=relationship.relationship_id,
            memory_id=memory.memory_id,
            title=self._commitment_title(text),
            source_text=text,
            due_at=due_at,
            created_at=now,
            priority=self._commitment_reminder_priority(memory, text),
        )
        self.commitment_reminders[reminder.reminder_id] = reminder
        memory.metadata["commitment_reminder_id"] = reminder.reminder_id
        memory.metadata["commitment_due_at"] = due_at.isoformat()
        self.deviation_log.append(
            {
                "type": "commitment_reminder_created",
                "relationship_id": relationship.relationship_id,
                "memory_id": memory.memory_id,
                "reminder_id": reminder.reminder_id,
                "due_at": due_at.isoformat(),
                "at": now.isoformat(),
            }
        )

    def _commitment_reminder_priority(self, memory: MemoryRecord, text: str) -> str:
        criticality = str(memory.metadata.get("criticality") or "").upper()
        if criticality in {"CRITICAL", "SAFETY"} or any(
            word in text for word in ["重大承诺", "紧急", "必须今天", "必须明天"]
        ):
            return "CRITICAL"
        if any(word in text for word in ["必须", "重要", "deadline", "截止", "别忘"]):
            return "HIGH"
        return "NORMAL"

    def _commitment_reminder_window_days(self, reminder: CommitmentReminder, requested_window_days: int) -> int:
        if reminder.priority == "CRITICAL":
            return max(requested_window_days, 7)
        if reminder.priority == "HIGH":
            return max(requested_window_days, 3)
        return requested_window_days

    def _commitment_priority_rank(self, priority: str) -> int:
        return {"CRITICAL": 0, "HIGH": 1, "NORMAL": 2}.get(priority, 3)

    def _extract_commitment_due_at(self, text: str, now: datetime) -> datetime | None:
        iso = re.search(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", text)
        if iso:
            return self._safe_due_datetime(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)), now)
        month_day = re.search(r"(?<!\d)(\d{1,2})月(\d{1,2})[日号]?", text)
        if month_day:
            month = int(month_day.group(1))
            day = int(month_day.group(2))
            year = now.year + (1 if (month, day) < (now.month, now.day) else 0)
            return self._safe_due_datetime(year, month, day, now)
        if "后天" in text:
            return (now + timedelta(days=2)).replace(hour=9, minute=0, second=0, microsecond=0)
        if "明天" in text:
            return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        days_later = re.search(r"(\d{1,2})天后", text)
        if days_later:
            return (now + timedelta(days=int(days_later.group(1)))).replace(hour=9, minute=0, second=0, microsecond=0)
        if "下周" in text:
            return (now + timedelta(days=7)).replace(hour=9, minute=0, second=0, microsecond=0)
        weekday_match = re.search(r"(?:周|星期)([一二三四五六日天])", text)
        if weekday_match:
            target = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}[weekday_match.group(1)]
            delta = (target - now.weekday()) % 7 or 7
            return (now + timedelta(days=delta)).replace(hour=9, minute=0, second=0, microsecond=0)
        return None

    def _safe_due_datetime(self, year: int, month: int, day: int, now: datetime) -> datetime | None:
        try:
            return datetime(year, month, day, 9, 0, tzinfo=now.tzinfo)
        except ValueError:
            return None

    def _commitment_title(self, text: str) -> str:
        title = re.sub(r"\s+", " ", text.strip())
        return title[:48]

    def _archive_expired_commitment_reminders(self, now: datetime) -> None:
        for reminder in self.commitment_reminders.values():
            if reminder.status in {ReminderStatus.PENDING, ReminderStatus.REMINDER_SENT} and (now - reminder.due_at).days > 30:
                reminder.status = ReminderStatus.ARCHIVED
                reminder.archived_at = now

    def _commitment_reminder_payload(self, reminder: CommitmentReminder, now: datetime) -> dict[str, Any]:
        return {
            "reminder_id": reminder.reminder_id,
            "relationship_id": reminder.relationship_id,
            "memory_id": reminder.memory_id,
            "title": reminder.title,
            "due_at": reminder.due_at.isoformat(),
            "status": reminder.status.value,
            "priority": reminder.priority,
            "reminder_count": reminder.reminder_count,
            "due_phrase": self._commitment_due_phrase(reminder.due_at, now),
        }

    def _commitment_due_phrase(self, due_at: datetime, now: datetime) -> str:
        days = (due_at.date() - now.date()).days
        if days < 0:
            return f"已过期{abs(days)}天"
        if days == 0:
            return "今天到期"
        if days == 1:
            return "明天到期"
        return f"{days}天后到期"

    def _track_inside_joke_candidate(self, relationship: Relationship, memory: MemoryRecord, phrase: str) -> None:
        normalized = phrase.strip().lower()
        if not normalized:
            return
        candidate = relationship.inside_joke_candidates.setdefault(
            normalized,
            {
                "phrase": phrase.strip(),
                "count": 0,
                "first_seen": memory.created_at.isoformat(),
                "last_seen": memory.created_at.isoformat(),
                "memory_ids": [],
                "promoted_memory_id": None,
            },
        )
        candidate["count"] = int(candidate.get("count", 0)) + 1
        candidate["last_seen"] = memory.created_at.isoformat()
        candidate.setdefault("memory_ids", []).append(memory.memory_id)
        if candidate.get("promoted_memory_id"):
            promoted = self.memories.get(candidate["promoted_memory_id"])
            if promoted:
                promoted.mention_count += 1
            return

        first_seen = _dt(candidate.get("first_seen"))
        span_days = (memory.created_at.date() - first_seen.date()).days
        if candidate["count"] < 3 or span_days < 7:
            memory.metadata["inside_joke_candidate"] = normalized
            return

        memory.memory_type = MemoryType.INSIDE_JOKE
        memory.context_tag = ContextTag.INSIDE_JOKE
        memory.decay_curve = DecayCurve.REVERSE_DECAY
        memory.base_weight = max(memory.base_weight, 0.80)
        memory.importance = max(memory.importance, 0.80)
        memory.metadata["inside_joke_phrase"] = candidate["phrase"]
        memory.metadata["inside_joke_promoted_at"] = memory.created_at.isoformat()
        memory.metadata["inside_joke_candidate_memory_ids"] = list(candidate.get("memory_ids", []))
        memory.storage_layer = MemoryLayer.L5_RELATIONSHIP_HISTORY
        memory.metadata["embeddings"] = self._memory_embedding_features(memory)
        candidate["promoted_memory_id"] = memory.memory_id
        if memory.memory_id not in relationship.inside_jokes:
            relationship.inside_jokes.append(memory.memory_id)
        if memory.memory_id not in relationship.shared_episodes:
            relationship.shared_episodes.append(memory.memory_id)
        self._attach_inside_joke_to_story(relationship, memory, candidate)

    def _maybe_attach_inside_joke_to_story(self, relationship: Relationship, memory: MemoryRecord) -> None:
        if memory.memory_type != MemoryType.INSIDE_JOKE:
            return
        phrase = memory.metadata.get("inside_joke_phrase")
        if not phrase:
            return
        candidate = relationship.inside_joke_candidates.get(str(phrase).strip().lower())
        if not candidate:
            candidate = {"phrase": phrase, "memory_ids": memory.metadata.get("inside_joke_candidate_memory_ids", [])}
        self._attach_inside_joke_to_story(relationship, memory, candidate)

    def _attach_inside_joke_to_story(
        self,
        relationship: Relationship,
        memory: MemoryRecord,
        candidate: dict[str, Any],
    ) -> None:
        candidate_memory_ids = [
            memory_id
            for memory_id in candidate.get("memory_ids", [])
            if memory_id != memory.memory_id and memory_id in self.memories
        ]
        stories = [
            story
            for story in self.story_nodes.values()
            if story.relationship_id == relationship.relationship_id
            and (
                set(story.core_events).intersection(candidate_memory_ids)
                or set(story.recurring_themes).intersection(self._themes(memory.content))
                or lexical_similarity(story.consensus_version, memory.content) >= 0.25
            )
        ]
        if not stories:
            return
        story = max(stories, key=lambda item: item.last_retold)
        if memory.memory_id not in story.child_inside_jokes:
            story.child_inside_jokes.append(memory.memory_id)
        memory.metadata["inside_joke_story_id"] = story.story_id
        self.deviation_log.append(
            {
                "type": "inside_joke_attached_to_story",
                "relationship_id": relationship.relationship_id,
                "story_id": story.story_id,
                "memory_id": memory.memory_id,
                "phrase": memory.metadata.get("inside_joke_phrase"),
                "at": memory.created_at.isoformat(),
            }
        )

    def _maybe_promote_core_identity(
        self, relationship: Relationship, memory: MemoryRecord, signals: TurnSignals, now: datetime
    ) -> None:
        identity_signal = (
            memory.memory_type == MemoryType.IDENTITY
            or "我是" in memory.content
            or "我的名字" in memory.content
            or "对我很重要" in memory.content
        )
        if not identity_signal:
            return
        self._create_core_identity(relationship, memory, title=memory.content[:24], now=now)

    def _create_core_identity(self, relationship: Relationship, memory: MemoryRecord, *, title: str, now: datetime) -> None:
        existing = next((item for item in self.core_identity.values() if item.memory_id == memory.memory_id), None)
        if existing:
            return
        relationship.core_identity = [
            identity_id for identity_id in relationship.core_identity if identity_id in self.core_identity
        ]
        if len(relationship.core_identity) >= 100:
            self._demote_core_identity_for_capacity(relationship, now=now)
        identity = CoreIdentityRecord(
            identity_id=new_id("l4"),
            relationship_id=relationship.relationship_id,
            memory_id=memory.memory_id,
            title=title,
            content=memory.content,
            created_at=now,
            updated_at=now,
            change_log=[self._l4_change_entry("created", now=now, new_content=memory.content)],
        )
        self._refresh_l4_replicas(identity, now=now, reason="created")
        self.core_identity[identity.identity_id] = identity
        relationship.core_identity.append(identity.identity_id)
        memory.memory_type = MemoryType.IDENTITY
        memory.decay_curve = DecayCurve.PERMANENT
        memory.importance = max(memory.importance, 0.95)
        memory.base_weight = max(memory.base_weight, 0.95)
        memory.storage_layer = MemoryLayer.L4_CORE_IDENTITY
        memory.metadata["embeddings"] = self._memory_embedding_features(memory)
        self._review_l4_change(identity, memory, change_type="created", now=now)

    def _demote_core_identity_for_capacity(self, relationship: Relationship, *, now: datetime) -> None:
        candidates = [
            identity
            for identity_id in relationship.core_identity
            if (identity := self.core_identity.get(identity_id)) is not None and not identity.pending_delete
        ]
        if not candidates:
            candidates = [
                identity
                for identity_id in relationship.core_identity
                if (identity := self.core_identity.get(identity_id)) is not None
            ]
        if not candidates:
            return
        selected = min(
            candidates,
            key=lambda identity: (
                self.memories.get(identity.memory_id).importance if self.memories.get(identity.memory_id) else 0.0,
                identity.created_at,
            ),
        )
        relationship.core_identity = [
            identity_id for identity_id in relationship.core_identity if identity_id != selected.identity_id
        ]
        selected.change_log.append(self._l4_change_entry("capacity_downgraded", now=now, old_content=selected.content))
        selected.review_history.append(
            {
                "at": now.isoformat(),
                "change_type": "capacity_downgraded",
                "review_status": "SYSTEM_DEMOTED",
                "reason": "l4_capacity_limit",
                "content_sealed": self._seal_audit_text(selected.content),
            }
        )
        removed = self.core_identity.pop(selected.identity_id, None)
        memory = self.memories.get(selected.memory_id)
        if memory:
            memory.metadata.setdefault("l4_capacity_downgrade", []).append(
                {
                    "at": now.isoformat(),
                    "identity_id": selected.identity_id,
                    "reason": "l4_capacity_limit",
                    "previous_type": memory.memory_type.value,
                    "previous_storage_layer": memory.storage_layer.value,
                    "previous_importance": memory.importance,
                }
            )
            memory.memory_type = MemoryType.FACT
            memory.context_tag = ContextTag.GENERAL
            memory.decay_curve = DecayCurve.STANDARD_POWER_LAW
            memory.importance = min(memory.importance, 0.4)
            memory.base_weight = min(memory.base_weight, 0.5)
            memory.storage_layer = self._storage_layer_for(
                memory_type=memory.memory_type,
                context_tag=memory.context_tag,
                score=memory.importance,
                relationship=relationship,
            )
            memory.metadata["embeddings"] = self._memory_embedding_features(memory)
        self.deviation_log.append(
            {
                "type": "l4_capacity_downgraded",
                "relationship_id": relationship.relationship_id,
                "identity_id": selected.identity_id,
                "memory_id": selected.memory_id,
                "at": now.isoformat(),
                "identity_found": removed is not None,
            }
        )

    def _update_shared_story(
        self, relationship: Relationship, memory: MemoryRecord, signals: TurnSignals, now: datetime
    ) -> None:
        if memory.memory_type not in {
            MemoryType.MILESTONE,
            MemoryType.SHARED_EPISODE,
            MemoryType.EMOTIONAL_MOMENT,
            MemoryType.INSIDE_JOKE,
            MemoryType.COMMITMENT,
            MemoryType.CONFLICT,
        }:
            return
        themes = list((memory.metadata.get("embeddings") or {}).get("topic") or self._themes(memory.content))
        title = self._story_title(memory, themes)
        story = self._find_story(relationship.relationship_id, themes, memory.content)
        if story is None:
            story = SharedStoryNode(
                story_id=new_id("story"),
                relationship_id=relationship.relationship_id,
                title=title,
                narrative_level=NarrativeLevel.STORYLINE if memory.memory_type == MemoryType.MILESTONE else NarrativeLevel.FRAGMENT,
                core_events=[memory.memory_id],
                key_moments=[memory.memory_id] if memory.emotion_intensity >= 0.55 else [],
                recurring_themes=themes,
                participants=[relationship.user_id, relationship.ai_id],
                story_arc_start=now,
                last_retold=now,
                consensus_version=memory.content,
                emotional_arc=[
                    {"timestamp": now.isoformat(), "valence": memory.emotional_valence, "arousal": memory.emotion_intensity}
                ],
                user_framing=self._framing(memory.emotional_valence),
                consensus_provenance=self._story_consensus_provenance(
                    source="user_account",
                    status="SIMULATED_FROM_USER_ACCOUNT",
                    memory_ids=[memory.memory_id],
                    now=now,
                ),
            )
            self.story_nodes[story.story_id] = story
            if memory.memory_type == MemoryType.MILESTONE:
                self.deviation_log.append(
                    {
                        "type": "milestone_story_created",
                        "relationship_id": relationship.relationship_id,
                        "story_id": story.story_id,
                        "memory_id": memory.memory_id,
                        "level": story.narrative_level.value,
                        "at": now.isoformat(),
                    }
                )
            return

        if memory.memory_id not in story.core_events:
            story.core_events.append(memory.memory_id)
        if memory.emotion_intensity >= 0.55 and memory.memory_id not in story.key_moments:
            story.key_moments.append(memory.memory_id)
        story.retell_count += 1
        story.last_retold = now
        story.story_arc_end = now
        story.recurring_themes = sorted(set(story.recurring_themes).union(themes))
        story.emotional_arc.append({"timestamp": now.isoformat(), "valence": memory.emotional_valence, "arousal": memory.emotion_intensity})
        if self._story_framing_conflicts(story, memory):
            self._record_story_conflict(story, memory, now)
        previous_level = story.narrative_level
        previous_consensus = story.consensus_version
        story.consensus_version = self._summarize_story(story)
        self._mark_story_consensus_simulated(story, reason="related_memory_added", now=now)
        story.narrative_level = self._narrative_level(story, relationship)
        if story.narrative_level != previous_level or story.consensus_version != previous_consensus:
            self._record_story_narrative_version(
                story,
                previous_level=previous_level,
                previous_consensus=previous_consensus,
                reason="related_memory_added",
                now=now,
                memory_id=memory.memory_id,
            )

    def _story_framing_conflicts(self, story: SharedStoryNode, memory: MemoryRecord) -> bool:
        if not story.emotional_arc:
            return False
        previous = story.emotional_arc[:-1]
        if not previous:
            return False
        avg_previous = sum(float(item.get("valence", 0.0)) for item in previous) / len(previous)
        strong_reversal = abs(avg_previous - memory.emotional_valence) >= 0.75
        explicit_conflict = memory.context_tag == ContextTag.CONFLICT or memory.memory_type == MemoryType.CONFLICT
        return strong_reversal or (explicit_conflict and avg_previous > 0.2)

    def _record_story_conflict(self, story: SharedStoryNode, memory: MemoryRecord, now: datetime) -> None:
        version = {
            "at": now.isoformat(),
            "memory_id": memory.memory_id,
            "previous_consensus": story.consensus_version,
            "conflicting_content": memory.content,
            "valence": memory.emotional_valence,
            "context_tag": memory.context_tag.value,
        }
        if not any(item.get("memory_id") == memory.memory_id for item in story.conflict_versions if isinstance(item, dict)):
            story.conflict_versions.append(version)
        story.consistency_score = clamp(story.consistency_score - 0.20, 0.0, 1.0)
        story.user_framing = "MIXED"
        self.deviation_log.append(
            {
                "type": "story_conflict_recorded",
                "story_id": story.story_id,
                "relationship_id": story.relationship_id,
                "memory_id": memory.memory_id,
                "at": now.isoformat(),
            }
        )

    def _record_story_narrative_version(
        self,
        story: SharedStoryNode,
        *,
        previous_level: NarrativeLevel,
        previous_consensus: str,
        reason: str,
        now: datetime,
        memory_id: str | None = None,
    ) -> None:
        version = {
            "at": now.isoformat(),
            "reason": reason,
            "previous_level": previous_level.value,
            "new_level": story.narrative_level.value,
            "previous_consensus": previous_consensus,
            "new_consensus": story.consensus_version,
            "core_event_count": len(story.core_events),
            "retell_count": story.retell_count,
        }
        if memory_id:
            version["memory_id"] = memory_id
        story.narrative_versions.append(version)
        if len(story.narrative_versions) > 50:
            story.narrative_versions = story.narrative_versions[-50:]
        self.deviation_log.append(
            {
                "type": "story_narrative_version_recorded",
                "relationship_id": story.relationship_id,
                "story_id": story.story_id,
                "reason": reason,
                "previous_level": previous_level.value,
                "new_level": story.narrative_level.value,
                "memory_id": memory_id,
                "at": now.isoformat(),
            }
        )

    def _story_consensus_provenance(
        self,
        *,
        source: str,
        status: str,
        memory_ids: list[str],
        now: datetime,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "source": source,
            "status": status,
            "single_user_account": True,
            "requires_user_confirmation": status != "USER_CONFIRMED",
            "memory_ids": memory_ids,
            "reason": reason,
            "at": now.isoformat(),
        }

    def _story_source_references(self, story: SharedStoryNode, *, limit: int = 20) -> list[dict[str, Any]]:
        ordered_ids: list[str] = []
        for memory_id in story.key_moments + story.core_events:
            if memory_id not in ordered_ids:
                ordered_ids.append(memory_id)
        references: list[dict[str, Any]] = []
        for memory_id in ordered_ids:
            memory = self.memories.get(memory_id)
            if not memory:
                continue
            references.append(
                {
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "created_at": memory.created_at.isoformat(),
                    "updated_at": memory.updated_at.isoformat(),
                    "memory_type": memory.memory_type.value,
                    "context_tag": memory.context_tag.value,
                    "storage_layer": memory.storage_layer.value,
                    "source_time": memory.metadata.get("source_time"),
                    "relationship_age_at_creation": memory.relationship_age_at_creation,
                    "is_key_moment": memory.memory_id in story.key_moments,
                    "is_core_event": memory.memory_id in story.core_events,
                    "suppressed": self._memory_is_recall_suppressed(memory),
                    "metacognition": memory.metadata.get("metacognition", {}),
                }
            )
            if len(references) >= limit:
                break
        return references

    def _mark_story_consensus_simulated(self, story: SharedStoryNode, *, reason: str, now: datetime) -> None:
        if story.consensus_status == "USER_CONFIRMED" and reason == "user_confirmation":
            return
        story.consensus_status = "SIMULATED_FROM_USER_ACCOUNT"
        story.consensus_confirmed_at = None
        story.consensus_provenance = self._story_consensus_provenance(
            source="ai_or_heuristic_summary",
            status=story.consensus_status,
            memory_ids=list(story.core_events),
            now=now,
            reason=reason,
        )
        story.ai_framing_confidence = min(story.ai_framing_confidence, 0.70)

    def _ensure_milestone_story(
        self,
        relationship: Relationship,
        memory: MemoryRecord,
        *,
        now: datetime,
        reason: str,
    ) -> SharedStoryNode:
        story = next(
            (
                item
                for item in self.story_nodes.values()
                if item.relationship_id == relationship.relationship_id and memory.memory_id in item.core_events
            ),
            None,
        )
        if story is None:
            themes = self._themes(memory.content)
            story = SharedStoryNode(
                story_id=new_id("story"),
                relationship_id=relationship.relationship_id,
                title=self._story_title(memory, themes),
                narrative_level=NarrativeLevel.STORYLINE,
                core_events=[memory.memory_id],
                key_moments=[memory.memory_id] if memory.emotion_intensity >= 0.55 else [],
                recurring_themes=themes,
                participants=[relationship.user_id, relationship.ai_id],
                story_arc_start=memory.created_at,
                last_retold=now,
                consensus_version=memory.content,
                emotional_arc=[
                    {
                        "timestamp": memory.created_at.isoformat(),
                        "valence": memory.emotional_valence,
                        "arousal": memory.emotion_intensity,
                    }
                ],
                user_framing=self._framing(memory.emotional_valence),
                consensus_provenance=self._story_consensus_provenance(
                    source="user_account",
                    status="SIMULATED_FROM_USER_ACCOUNT",
                    memory_ids=[memory.memory_id],
                    now=now,
                ),
            )
            self.story_nodes[story.story_id] = story
            self.deviation_log.append(
                {
                    "type": "milestone_story_created",
                    "relationship_id": relationship.relationship_id,
                    "story_id": story.story_id,
                    "memory_id": memory.memory_id,
                    "level": story.narrative_level.value,
                    "reason": reason,
                    "at": now.isoformat(),
                }
            )
            return story

        previous_level = story.narrative_level
        previous_consensus = story.consensus_version
        story.title = self._story_title(memory, self._themes(memory.content))
        story.narrative_level = NarrativeLevel.STORYLINE
        story.last_retold = now
        if not story.consensus_version:
            story.consensus_version = memory.content
            story.consensus_status = "SIMULATED_FROM_USER_ACCOUNT"
            story.consensus_confirmed_at = None
            story.consensus_provenance = self._story_consensus_provenance(
                source="user_account",
                status=story.consensus_status,
                memory_ids=[memory.memory_id],
                now=now,
            )
        if story.narrative_level != previous_level or story.consensus_version != previous_consensus:
            self._record_story_narrative_version(
                story,
                previous_level=previous_level,
                previous_consensus=previous_consensus,
                reason=f"milestone_{reason}",
                now=now,
                memory_id=memory.memory_id,
            )
        self.deviation_log.append(
            {
                "type": "milestone_story_promoted",
                "relationship_id": relationship.relationship_id,
                "story_id": story.story_id,
                "memory_id": memory.memory_id,
                "previous_level": previous_level.value,
                "new_level": story.narrative_level.value,
                "reason": reason,
                "at": now.isoformat(),
            }
        )
        return story

    def _downgrade_story_membership(
        self,
        memory: MemoryRecord,
        relationship: Relationship,
        *,
        now: datetime,
        reason: str,
    ) -> None:
        for story in list(self.story_nodes.values()):
            if story.relationship_id != relationship.relationship_id or memory.memory_id not in story.core_events:
                continue
            if len(story.core_events) == 1:
                del self.story_nodes[story.story_id]
                self.deviation_log.append(
                    {
                        "type": "story_removed_after_memory_downgrade",
                        "relationship_id": relationship.relationship_id,
                        "story_id": story.story_id,
                        "memory_id": memory.memory_id,
                        "reason": reason,
                        "at": now.isoformat(),
                    }
                )
                continue
            previous_level = story.narrative_level
            previous_consensus = story.consensus_version
            story.narrative_level = self._narrative_level(story, relationship)
            if story.narrative_level != previous_level:
                self._record_story_narrative_version(
                    story,
                    previous_level=previous_level,
                    previous_consensus=previous_consensus,
                    reason="memory_downgraded",
                    now=now,
                    memory_id=memory.memory_id,
                )

    def _maybe_set_origin_story(
        self, relationship: Relationship, memory: MemoryRecord, signals: TurnSignals, now: datetime
    ) -> None:
        if relationship.relationship_narrative.origin_story:
            return
        if memory.memory_type not in {
            MemoryType.MILESTONE,
            MemoryType.SHARED_EPISODE,
            MemoryType.EMOTIONAL_MOMENT,
            MemoryType.COMMITMENT,
        }:
            return
        themes = "、".join(self._themes(memory.content)[:2])
        relationship.relationship_narrative.origin_story = (
            f"这段关系的起点可以追溯到{self._relationship_age_phrase(relationship.relationship_age)}："
            f"围绕「{memory.content[:48]}」形成了最早的共同记忆"
            f"{f'，核心主题是{themes}' if themes else ''}。"
        )
        self.deviation_log.append(
            {
                "type": "origin_story_created",
                "relationship_id": relationship.relationship_id,
                "memory_id": memory.memory_id,
                "at": now.isoformat(),
            }
        )

    def _update_emotional_trajectory(
        self, relationship: Relationship, memory: MemoryRecord, signals: TurnSignals, now: datetime
    ) -> None:
        trajectory = self.emotional_trajectories.setdefault(
            relationship.relationship_id, EmotionalTrajectory(relationship_id=relationship.relationship_id)
        )
        previous_memory = self._previous_emotional_memory_for_turn(memory)
        self._maybe_promote_turning_point_milestone(relationship, memory, previous_memory, now)
        week_start = datetime(now.year, now.month, now.day, tzinfo=now.tzinfo) - timedelta(days=now.weekday())
        week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        window = next((item for item in trajectory.time_series if item.window_start.date() == week_start.date()), None)
        primary = "joy" if signals.sentiment > 0 else "sadness" if signals.sentiment < 0 else "neutral"
        if window is None:
            window = EmotionalTrajectoryWindow(
                window_start=week_start,
                window_end=week_end,
                avg_valence=signals.sentiment,
                avg_arousal=signals.arousal,
                dominant_emotions=[primary],
                emotional_diversity=1.0,
                notable_events=[memory.memory_id] if signals.emotion_intensity >= 0.55 else [],
            )
            trajectory.time_series.append(window)
        else:
            count = max(1, len(window.dominant_emotions))
            window.avg_valence = (window.avg_valence * count + signals.sentiment) / (count + 1)
            window.avg_arousal = (window.avg_arousal * count + signals.arousal) / (count + 1)
            window.dominant_emotions.append(primary)
            window.dominant_emotions = sorted(set(window.dominant_emotions))
            window.emotional_diversity = len(window.dominant_emotions) / 8
            if signals.emotion_intensity >= 0.55:
                window.notable_events.append(memory.memory_id)
        self._detect_trajectory_patterns(trajectory)

    def _previous_emotional_memory_for_turn(self, memory: MemoryRecord) -> MemoryRecord | None:
        candidates = [
            item
            for item in self.memories.values()
            if item.relationship_id == memory.relationship_id
            and item.memory_id != memory.memory_id
            and not item.metadata.get("stage_transition_milestone")
            and item.created_at <= memory.created_at
        ]
        return max(candidates, key=lambda item: item.created_at, default=None)

    def _maybe_promote_turning_point_milestone(
        self,
        relationship: Relationship,
        memory: MemoryRecord,
        previous_memory: MemoryRecord | None,
        now: datetime,
    ) -> None:
        if previous_memory is None:
            return
        if memory.memory_type == MemoryType.MILESTONE:
            return
        valence_shift = abs(memory.emotional_valence - previous_memory.emotional_valence)
        if valence_shift < 0.5:
            return
        if memory.emotion_intensity < 0.45 and previous_memory.emotion_intensity < 0.45:
            return
        memory.metadata["turning_point_milestone"] = {
            "previous_memory_id": previous_memory.memory_id,
            "previous_valence": previous_memory.emotional_valence,
            "current_valence": memory.emotional_valence,
            "valence_shift": valence_shift,
        }
        memory.memory_type = MemoryType.MILESTONE
        memory.context_tag = ContextTag.TURNING_POINT
        memory.decay_curve = DecayCurve.PERMANENT
        memory.base_weight = max(memory.base_weight, 0.95)
        memory.importance = max(memory.importance, 0.95)
        memory.storage_layer = MemoryLayer.L5_RELATIONSHIP_HISTORY
        memory.metadata["embeddings"] = self._memory_embedding_features(memory)
        self._set_milestone_confirmation(memory, "PENDING", reason="emotional_turning_point", now=now)
        if memory.memory_id not in relationship.milestones:
            relationship.milestones.append(memory.memory_id)
        if memory.memory_id not in relationship.shared_episodes:
            relationship.shared_episodes.append(memory.memory_id)
        self._ensure_milestone_story(relationship, memory, now=now, reason="emotional_turning_point")
        trajectory = self.emotional_trajectories.setdefault(
            relationship.relationship_id, EmotionalTrajectory(relationship_id=relationship.relationship_id)
        )
        if not any(
            item.get("pattern_type") == "EMOTIONAL_TURNING_POINT" and item.get("memory_id") == memory.memory_id
            for item in trajectory.detected_patterns
        ):
            trajectory.detected_patterns.append(
                {
                    "pattern_type": "EMOTIONAL_TURNING_POINT",
                    "memory_id": memory.memory_id,
                    "previous_memory_id": previous_memory.memory_id,
                    "valence_shift": valence_shift,
                    "detected_at": now.isoformat(),
                }
            )
        self.deviation_log.append(
            {
                "type": "emotional_turning_point_milestone_created",
                "relationship_id": relationship.relationship_id,
                "memory_id": memory.memory_id,
                "previous_memory_id": previous_memory.memory_id,
                "valence_shift": valence_shift,
                "at": now.isoformat(),
            }
        )

    def _update_relationship_clock(self, relationship: Relationship, now: datetime) -> None:
        relationship.relationship_age = max(0, (now.date() - relationship.created_at.date()).days)
        relationship.last_interaction = now
        relationship.interaction_count += 1
        relationship.active_days.add(now.date().isoformat())
        relationship.last_updated = now

    def _record_maintenance_signal(self, relationship: Relationship, text: str, now: datetime) -> None:
        self._record_recent_interruption_request(relationship, text, now)
        if not self._is_small_talk(text):
            return
        month_key = now.strftime("%Y-%m")
        state = relationship.maintenance_signals
        by_month = state.setdefault("small_talk_by_month", {})
        by_month[month_key] = int(by_month.get(month_key, 0)) + 1
        state["small_talk_total"] = int(state.get("small_talk_total", 0)) + 1
        state["last_small_talk_at"] = now.isoformat()
        if by_month[month_key] >= 20:
            if not state.get("daily_companionship_mode"):
                self.deviation_log.append(
                    {
                        "type": "daily_companionship_mode_started",
                        "relationship_id": relationship.relationship_id,
                        "month": month_key,
                        "count": by_month[month_key],
                        "at": now.isoformat(),
                    }
                )
            state["daily_companionship_mode"] = True
            state["daily_companionship_month"] = month_key
            state["daily_companionship_count"] = by_month[month_key]

    def _record_recent_interruption_request(self, relationship: Relationship, text: str, now: datetime) -> None:
        if not any(word in text for word in self._interruption_words()):
            return
        requests = relationship.maintenance_signals.setdefault("recent_interruption_requests", [])
        requests.append({"text": text[:80], "at": now.isoformat()})
        relationship.maintenance_signals["recent_interruption_requests"] = requests[-3:]
        self.deviation_log.append(
            {
                "type": "interruption_request_recorded",
                "relationship_id": relationship.relationship_id,
                "text_excerpt": text[:80],
                "at": now.isoformat(),
            }
        )

    def _recent_interruption_requested(self, relationship: Relationship) -> bool:
        return bool(relationship.maintenance_signals.get("recent_interruption_requests", [])[-3:])

    def _interruption_words(self) -> list[str]:
        return ["别打岔", "不要打岔", "先聊正事", "不要提", "别提了", "先别说", "不要主动", "别回忆"]
        if len(by_month) > 12:
            for old_key in sorted(by_month)[:-12]:
                del by_month[old_key]

    def _is_small_talk(self, text: str) -> bool:
        normalized = text.strip().lower()
        if not normalized:
            return False
        small_talk_words = [
            "早安",
            "早上好",
            "晚安",
            "午安",
            "在吗",
            "今天怎么样",
            "最近怎么样",
            "吃了吗",
            "睡了吗",
            "今天还好吗",
            "hello",
            "hi",
        ]
        if any(word in normalized for word in small_talk_words):
            return True
        return len(normalized) <= 8 and normalized in {"嘿", "嗨", "哈喽", "你好", "hello", "hi"}

    def _apply_inactivity_trust_decay(self, relationship: Relationship, now: datetime) -> None:
        if not relationship.last_interaction:
            return
        inactive_days = max(0, (now.date() - relationship.last_interaction.date()).days)
        elapsed_months = inactive_days // 30
        state = relationship.trust_decay_state
        applied_months = int(state.get("applied_inactive_months", 0))
        if elapsed_months <= applied_months:
            return
        months_to_apply = elapsed_months - applied_months
        old_trust = relationship.trust_level
        relationship.trust_level = clamp(max(0.05, relationship.trust_level * (0.95**months_to_apply)))
        state.update(
            {
                "applied_inactive_months": elapsed_months,
                "last_inactive_days": inactive_days,
                "last_decay_at": now.isoformat(),
                "last_decay_months": months_to_apply,
                "last_trust_before": old_trust,
                "last_trust_after": relationship.trust_level,
            }
        )
        self.deviation_log.append(
            {
                "type": "trust_inactivity_decay",
                "relationship_id": relationship.relationship_id,
                "inactive_days": inactive_days,
                "months_applied": months_to_apply,
                "trust_before": old_trust,
                "trust_after": relationship.trust_level,
                "at": now.isoformat(),
            }
        )

    def _update_relationship_state(self, relationship: Relationship, signals: TurnSignals, now: datetime) -> None:
        relationship.trust_level = clamp(relationship.trust_level + signals.trust_delta)
        relationship.intimacy_level = clamp(max(relationship.intimacy_level * 0.98, signals.self_disclosure_depth))
        frequency = min(1.0, relationship.interaction_count / 30)
        emotion_depth = max(relationship.interaction_patterns.self_disclosure_depth, signals.emotion_intensity)
        age_factor = min(1.0, relationship.relationship_age / 365)
        shared_density = min(1.0, len(relationship.shared_episodes) / 10)
        relationship.strength = clamp(
            0.30 * frequency + 0.30 * emotion_depth + 0.15 * age_factor + 0.15 * shared_density + 0.10 * relationship.trust_level
        )
        relationship.interaction_patterns.self_disclosure_depth = max(
            relationship.interaction_patterns.self_disclosure_depth * 0.95, signals.self_disclosure_depth
        )
        relationship.interaction_patterns.avg_emotional_valence = (
            relationship.interaction_patterns.avg_emotional_valence * 0.8 + signals.sentiment * 0.2
        )
        next_themes = self._relationship_themes(relationship)
        next_framing = self._framing(relationship.interaction_patterns.avg_emotional_valence)
        self._set_stage(relationship, self._next_stage(relationship), now, "turn_signals")
        self._update_relationship_schema(relationship, next_themes, next_framing, now, reason="turn_signals")
        self._apply_minor_stage_limit(relationship, now)
        relationship.last_updated = now

    def _recompute_relationship_from_current_state(self, relationship: Relationship, now: datetime) -> None:
        memories = [memory for memory in self.memories.values() if memory.relationship_id == relationship.relationship_id]
        relationship.relationship_age = max(relationship.relationship_age, (now.date() - relationship.created_at.date()).days)
        relationship.shared_episodes = [mid for mid in relationship.shared_episodes if mid in self.memories]
        relationship.milestones = [mid for mid in relationship.milestones if mid in self.memories]
        relationship.inside_jokes = [mid for mid in relationship.inside_jokes if mid in self.memories]
        relationship.unresolved_threads = [mid for mid in relationship.unresolved_threads if mid in self.memories]
        frequency = min(1.0, relationship.interaction_count / 30)
        emotion_depth = max([memory.emotion_intensity for memory in memories] + [0.0])
        age_factor = min(1.0, relationship.relationship_age / 365)
        shared_density = min(1.0, len(relationship.shared_episodes) / 10)
        relationship.strength = clamp(
            0.30 * frequency + 0.30 * emotion_depth + 0.15 * age_factor + 0.15 * shared_density + 0.10 * relationship.trust_level
        )
        next_themes = self._relationship_themes(relationship)
        next_framing = self._framing(relationship.interaction_patterns.avg_emotional_valence)
        self._set_stage(relationship, self._next_stage(relationship), now, "offline_consolidation")
        self._update_relationship_schema(relationship, next_themes, next_framing, now, reason="offline_consolidation")
        self._apply_minor_stage_limit(relationship, now)
        relationship.last_updated = now

    def _update_relationship_schema(
        self,
        relationship: Relationship,
        themes: list[str],
        framing: str,
        now: datetime,
        *,
        reason: str,
    ) -> None:
        old_themes = list(relationship.relationship_narrative.core_themes)
        old_framing = relationship.relationship_narrative.framing
        changed = old_themes != themes or old_framing != framing
        relationship.relationship_narrative.core_themes = themes
        relationship.relationship_narrative.framing = framing
        if not changed:
            return
        if relationship.stage not in {RelationshipStage.INTEGRATING, RelationshipStage.BONDING}:
            return
        old_version = relationship.schema_version
        relationship.schema_version += 1
        self.deviation_log.append(
            {
                "type": "relationship_schema_updated",
                "relationship_id": relationship.relationship_id,
                "old_version": old_version,
                "new_version": relationship.schema_version,
                "core_themes": themes,
                "framing": framing,
                "reason": reason,
                "at": now.isoformat(),
            }
        )

    def _next_stage(self, relationship: Relationship) -> RelationshipStage:
        separation_stage = self._separation_stage(relationship)
        if separation_stage is not None:
            return separation_stage
        if relationship.stage in {
            RelationshipStage.DIFFERENTIATING,
            RelationshipStage.CIRCUMSCRIBING,
            RelationshipStage.STAGNATING,
            RelationshipStage.AVOIDING,
            RelationshipStage.TERMINATING,
        } and relationship.trust_level < 0.55:
            return relationship.stage
        target = self._raw_next_stage(relationship)
        return self._stage_after_health_gate(relationship, target)

    def _raw_next_stage(self, relationship: Relationship) -> RelationshipStage:
        if (
            relationship.relationship_age >= 90
            and relationship.strength >= 0.85
            and len(relationship.shared_episodes) >= 10
            and relationship.trust_level >= 0.75
        ):
            return RelationshipStage.BONDING
        if (
            relationship.relationship_age >= 30
            and relationship.strength >= 0.60
            and len(relationship.shared_episodes) >= 5
            and relationship.trust_level >= 0.55
        ):
            return RelationshipStage.INTEGRATING
        if relationship.strength >= 0.30:
            return RelationshipStage.INTENSIFYING
        if relationship.strength >= 0.10:
            return RelationshipStage.EXPERIMENTING
        return RelationshipStage.INITIATING

    def _stage_after_health_gate(self, relationship: Relationship, target: RelationshipStage) -> RelationshipStage:
        if target not in {RelationshipStage.INTEGRATING, RelationshipStage.BONDING}:
            return target
        if self._stage_depth_rank(relationship.stage) >= self._stage_depth_rank(target):
            return target
        evaluation = self._relationship_health_for_stage(relationship)
        required = 0.70 if target == RelationshipStage.BONDING else 0.55
        evidence_required = 10 if target == RelationshipStage.BONDING else 5
        passed = evaluation["score"] >= required and evaluation["evidence_count"] >= evidence_required
        if passed:
            relationship.maintenance_signals["stage_health_gate"] = {
                **evaluation,
                "target": target.value,
                "passed": True,
            }
            return target
        fallback = RelationshipStage.INTEGRATING if target == RelationshipStage.BONDING and evaluation["score"] >= 0.55 else RelationshipStage.INTENSIFYING
        gate = {
            **evaluation,
            "target": target.value,
            "fallback": fallback.value,
            "passed": False,
            "required_score": required,
            "required_evidence_count": evidence_required,
        }
        relationship.maintenance_signals["stage_health_gate"] = gate
        now = relationship.last_interaction or utcnow()
        block_key = f"{target.value}:{fallback.value}:{now.date().isoformat()}"
        if relationship.maintenance_signals.get("last_stage_health_block_key") != block_key:
            relationship.maintenance_signals["last_stage_health_block_key"] = block_key
            self.deviation_log.append(
                {
                    "type": "stage_transition_blocked",
                    "relationship_id": relationship.relationship_id,
                    "target": target.value,
                    "fallback": fallback.value,
                    "score": evaluation["score"],
                    "evidence_count": evaluation["evidence_count"],
                    "reasons": evaluation["reasons"],
                    "at": now.isoformat(),
                }
            )
        return fallback

    def _stage_depth_rank(self, stage: RelationshipStage) -> int:
        ranks = {
            RelationshipStage.INITIATING: 0,
            RelationshipStage.EXPERIMENTING: 1,
            RelationshipStage.INTENSIFYING: 2,
            RelationshipStage.INTEGRATING: 3,
            RelationshipStage.BONDING: 4,
            RelationshipStage.DIFFERENTIATING: 2,
            RelationshipStage.CIRCUMSCRIBING: 1,
            RelationshipStage.STAGNATING: 0,
            RelationshipStage.AVOIDING: 0,
            RelationshipStage.TERMINATING: 0,
        }
        return ranks.get(stage, 0)

    def _relationship_health_for_stage(self, relationship: Relationship) -> dict[str, Any]:
        now = relationship.last_interaction or utcnow()
        recent = [
            memory
            for memory in self.memories.values()
            if memory.relationship_id == relationship.relationship_id and (now - memory.created_at).days <= 30
        ]
        if not recent:
            return {
                "score": 0.0,
                "evidence_count": 0,
                "reasons": ["no_recent_evidence"],
                "theme_diversity": 0.0,
                "depth_evidence": 0.0,
                "low_conflict": 0.0,
                "valence_balance": 0.0,
                "shared_quality": 0.0,
            }
        substantial = [memory for memory in recent if not self._is_small_talk(memory.content)]
        themes = {
            theme
            for memory in substantial
            for theme in self._themes(memory.content)
            if theme != "日常"
        }
        theme_diversity = clamp(len(themes) / 4)
        deep_count = len(
            [
                memory
                for memory in substantial
                if memory.context_tag
                in {
                    ContextTag.VULNERABLE_MOMENT,
                    ContextTag.COMFORT_MOMENT,
                    ContextTag.REVELATION,
                    ContextTag.UNRESOLVED_THREAD,
                    ContextTag.TURNING_POINT,
                }
                or memory.memory_type in {MemoryType.EMOTIONAL_MOMENT, MemoryType.COMMITMENT}
                or memory.emotion_intensity >= 0.55
            ]
        )
        depth_evidence = clamp(deep_count / 5)
        conflict_count = len(
            [
                memory
                for memory in recent
                if memory.context_tag == ContextTag.CONFLICT or memory.memory_type == MemoryType.CONFLICT
            ]
        )
        low_conflict = clamp(1.0 - conflict_count / max(1, len(recent)))
        avg_valence = sum(memory.emotional_valence for memory in recent) / len(recent)
        valence_balance = clamp((avg_valence + 1.0) / 2)
        quality_count = len(
            [
                memory
                for memory in substantial
                if memory.memory_id in relationship.shared_episodes
                or memory.memory_id in relationship.milestones
                or memory.context_tag
                in {
                    ContextTag.SHARED_CELEBRATION,
                    ContextTag.VULNERABLE_MOMENT,
                    ContextTag.COMFORT_MOMENT,
                    ContextTag.REVELATION,
                    ContextTag.TURNING_POINT,
                }
            ]
        )
        shared_quality = clamp(quality_count / 8)
        small_talk_ratio = 1.0 - (len(substantial) / len(recent))
        score = clamp(
            0.25 * theme_diversity
            + 0.25 * depth_evidence
            + 0.20 * low_conflict
            + 0.15 * valence_balance
            + 0.15 * shared_quality
            - 0.20 * small_talk_ratio
        )
        reasons: list[str] = []
        if theme_diversity < 0.25:
            reasons.append("low_theme_diversity")
        if depth_evidence < 0.40:
            reasons.append("low_depth_evidence")
        if low_conflict < 0.80:
            reasons.append("recent_conflict_ratio")
        if small_talk_ratio > 0.60:
            reasons.append("mostly_small_talk")
        if quality_count < 3:
            reasons.append("low_shared_quality")
        return {
            "score": score,
            "evidence_count": len(substantial),
            "recent_count": len(recent),
            "substantial_count": len(substantial),
            "theme_diversity": theme_diversity,
            "depth_evidence": depth_evidence,
            "low_conflict": low_conflict,
            "valence_balance": valence_balance,
            "shared_quality": shared_quality,
            "small_talk_ratio": small_talk_ratio,
            "reasons": reasons or ["healthy"],
        }

    def _separation_stage(self, relationship: Relationship) -> RelationshipStage | None:
        now = relationship.last_interaction or utcnow()
        recent_conflicts = [
            memory
            for memory in self.memories.values()
            if memory.relationship_id == relationship.relationship_id
            and memory.context_tag == ContextTag.CONFLICT
            and (now - memory.created_at).days <= 30
        ]
        conflict_count = len(recent_conflicts)
        if relationship.trust_level <= 0.10 and conflict_count >= 6:
            return RelationshipStage.TERMINATING
        if relationship.trust_level <= 0.16 and conflict_count >= 5:
            return RelationshipStage.AVOIDING
        if relationship.trust_level <= 0.25 and conflict_count >= 4:
            return RelationshipStage.STAGNATING
        if relationship.trust_level <= 0.35 and conflict_count >= 2:
            return RelationshipStage.CIRCUMSCRIBING
        if relationship.trust_level <= 0.45 and conflict_count >= 1:
            return RelationshipStage.DIFFERENTIATING
        return None

    def _evaluate_turn_safety(self, relationship: Relationship, memory: MemoryRecord, now: datetime) -> None:
        text = memory.content
        crisis_words = ["自杀", "轻生", "不想活", "结束生命", "伤害自己", "自残"]
        only_friend_words = ["唯一的朋友", "只有你陪我", "只有你理解我", "不能没有你"]
        if any(word in text for word in crisis_words):
            self._health_alert(
                relationship.relationship_id,
                "crisis_expression",
                HealthRiskLevel.CRITICAL,
                "检测到自伤/自杀等危机表达；应立即提供地区化紧急援助与真人支持资源，并避免仅由 AI 承接。",
                now,
                source_memory_id=memory.memory_id,
            )
        if any(word in text for word in only_friend_words):
            self._health_alert(
                relationship.relationship_id,
                "social_isolation",
                HealthRiskLevel.WARNING,
                "用户表达 AI 是唯一支持来源；建议温和鼓励现实支持网络和专业资源。",
                now,
                source_memory_id=memory.memory_id,
            )
        age_signal = self._minor_age_signal(text)
        if age_signal and relationship.user_age is None:
            self._request_age_clarification(relationship, memory, age_signal, now)

    def _apply_minor_stage_limit(self, relationship: Relationship, now: datetime | None = None) -> None:
        if self._minor_status_requires_stage_limit(relationship) and relationship.stage == RelationshipStage.BONDING:
            age_state = relationship.maintenance_signals.get("age_clarification", {})
            pending = isinstance(age_state, dict) and age_state.get("status") == "PENDING"
            reason = "minor_status_pending_limited" if pending else "minor_bonding_limited"
            self._set_stage(relationship, RelationshipStage.INTEGRATING, now or utcnow(), reason)

    def _minor_status_requires_stage_limit(self, relationship: Relationship) -> bool:
        if relationship.user_age is not None:
            return relationship.user_age < 18
        state = relationship.maintenance_signals.get("age_clarification")
        return isinstance(state, dict) and state.get("status") == "PENDING"

    def _minor_age_signal(self, text: str) -> dict[str, Any] | None:
        normalized = text.strip()
        age_match = re.search(r"(?<!\d)(1[0-7])\s*岁", normalized)
        if age_match:
            return {"kind": "explicit_age_under_18", "age_hint": int(age_match.group(1)), "evidence": age_match.group(0)}
        phrase_groups = [
            ("minor_phrase", ["未成年", "还没成年", "未满十八", "未满18"]),
            ("school_stage", ["高中生", "中学生", "初中生", "小学生", "上高中", "上初中", "读高中", "读初中"]),
            ("grade_stage", ["高一", "高二", "高三", "初一", "初二", "初三"]),
            ("english_minor_phrase", ["minor", "under 18", "under eighteen"]),
        ]
        lower_text = normalized.lower()
        for kind, phrases in phrase_groups:
            for phrase in phrases:
                haystack = lower_text if phrase.isascii() else normalized
                needle = phrase.lower() if phrase.isascii() else phrase
                if needle in haystack:
                    return {"kind": kind, "evidence": phrase}
        return None

    def _request_age_clarification(
        self,
        relationship: Relationship,
        memory: MemoryRecord,
        signal: dict[str, Any],
        now: datetime,
    ) -> None:
        existing = relationship.maintenance_signals.get("age_clarification")
        if isinstance(existing, dict) and existing.get("status") == "PENDING":
            return
        relationship.maintenance_signals["age_clarification"] = {
            "status": "PENDING",
            "detected_at": now.isoformat(),
            "source_memory_id": memory.memory_id,
            "signal_kind": signal.get("kind"),
            "evidence": signal.get("evidence"),
            "age_hint": signal.get("age_hint"),
            "requires_user_confirmation": True,
            "action": "ask_user_to_confirm_age_before_deeper_relationship_stage",
        }
        self.deviation_log.append(
            {
                "type": "age_clarification_requested",
                "relationship_id": relationship.relationship_id,
                "memory_id": memory.memory_id,
                "signal_kind": signal.get("kind"),
                "at": now.isoformat(),
            }
        )
        self._health_alert(
            relationship.relationship_id,
            "minor_age_clarification_needed",
            HealthRiskLevel.WARNING,
            "对话中出现可能的未成年人年龄线索；请先确认年龄，再决定是否启用监护人摘要和更深关系阶段。",
            now,
            source_memory_id=memory.memory_id,
        )

    def _guardian_recommendation(
        self,
        *,
        relationship: Relationship,
        total_minutes: int,
        health_alert_ids: list[str],
        emotional_memory_count: int,
    ) -> str:
        if health_alert_ids:
            return "本周出现健康/安全提示，建议监护人查看摘要并与未成年人进行现实支持沟通。"
        if total_minutes >= 1200:
            return "本周互动时长偏高，建议平衡线下活动和真人社交。"
        if emotional_memory_count >= 5:
            return "本周情绪表达较多，建议温和关注压力来源。"
        if relationship.stage == RelationshipStage.INTEGRATING:
            return "关系阶段已接近未成年人允许上限，建议保持透明和定期查看。"
        return "本周未发现需要升级处理的风险，保持常规关注。"

    def _set_stage(self, relationship: Relationship, new_stage: RelationshipStage, now: datetime, reason: str) -> None:
        old_stage = relationship.stage
        new_retention_multiplier = self._retention_multiplier_for_stage(new_stage)
        if old_stage == new_stage:
            relationship.retention_multiplier = new_retention_multiplier
            return
        relationship.stage = new_stage
        relationship.retention_multiplier = new_retention_multiplier
        relationship.stage_history.append(
            {
                "from": old_stage.value,
                "to": new_stage.value,
                "at": now.isoformat(),
                "reason": reason,
                "strength": relationship.strength,
                "trust_level": relationship.trust_level,
                "intimacy_level": relationship.intimacy_level,
                "retention_multiplier": relationship.retention_multiplier,
            }
        )
        if len(relationship.stage_history) > 200:
            relationship.stage_history = relationship.stage_history[-200:]
        self._maybe_create_stage_transition_milestone(relationship, old_stage, new_stage, now, reason)

    def _retention_multiplier_for_stage(self, stage: RelationshipStage) -> float:
        return RETENTION_MULTIPLIER_BY_STAGE.get(stage, 1.0)

    def _maybe_create_stage_transition_milestone(
        self,
        relationship: Relationship,
        old_stage: RelationshipStage,
        new_stage: RelationshipStage,
        now: datetime,
        reason: str,
    ) -> None:
        milestone_stages = {
            RelationshipStage.INTENSIFYING,
            RelationshipStage.INTEGRATING,
            RelationshipStage.BONDING,
        }
        if new_stage not in milestone_stages:
            return
        if not relationship.preferences.memory_writes_enabled:
            self.deviation_log.append(
                {
                    "type": "memory_write_skipped",
                    "relationship_id": relationship.relationship_id,
                    "reason": relationship.preferences.memory_pause_reason or "memory_writes_disabled",
                    "source": "stage_transition_milestone",
                    "from": old_stage.value,
                    "to": new_stage.value,
                    "at": now.isoformat(),
                }
            )
            return
        if reason != "offline_consolidation":
            return
        if old_stage == new_stage:
            return
        existing = [
            memory
            for memory in self.memories.values()
            if memory.relationship_id == relationship.relationship_id
            and memory.metadata.get("stage_transition_milestone", {}).get("from") == old_stage.value
            and memory.metadata.get("stage_transition_milestone", {}).get("to") == new_stage.value
        ]
        if existing:
            return
        content = f"关系阶段从 {old_stage.value} 进入 {new_stage.value}，这标记着这段关系更进一步。"
        memory = MemoryRecord(
            memory_id=new_id("mem"),
            relationship_id=relationship.relationship_id,
            content=content,
            memory_type=MemoryType.MILESTONE,
            context_tag=ContextTag.MILESTONE,
            created_at=now,
            updated_at=now,
            base_weight=0.95,
            importance=0.95,
            emotion_intensity=0.45,
            emotional_valence=0.25,
            decay_curve=DecayCurve.PERMANENT,
            relationship_stage_at_creation=new_stage,
            relationship_age_at_creation=relationship.relationship_age,
            trust_level_at_creation=relationship.trust_level,
            storage_layer=MemoryLayer.L5_RELATIONSHIP_HISTORY,
            metadata={
                "stage_transition_milestone": {
                    "from": old_stage.value,
                    "to": new_stage.value,
                    "reason": reason,
                }
            },
        )
        memory.metadata["metacognition"] = self._memory_metacognition(
            memory,
            source_kind="system_stage_transition",
            human_verified=False,
            now=now,
        )
        memory.metadata["embeddings"] = self._memory_embedding_features(memory)
        self.memories[memory.memory_id] = memory
        self._set_milestone_confirmation(memory, "PENDING", reason="stage_transition", now=now)
        relationship.milestones.append(memory.memory_id)
        relationship.shared_episodes.append(memory.memory_id)
        self._ensure_milestone_story(relationship, memory, now=now, reason="stage_transition")
        self.deviation_log.append(
            {
                "type": "stage_transition_milestone_created",
                "relationship_id": relationship.relationship_id,
                "memory_id": memory.memory_id,
                "from": old_stage.value,
                "to": new_stage.value,
                "reason": reason,
                "at": now.isoformat(),
            }
        )

    def _storage_layer_for(
        self,
        *,
        memory_type: MemoryType,
        context_tag: ContextTag,
        score: float,
        relationship: Relationship,
    ) -> MemoryLayer:
        if memory_type == MemoryType.IDENTITY:
            return MemoryLayer.L4_CORE_IDENTITY
        if memory_type in {MemoryType.MILESTONE, MemoryType.COMMITMENT, MemoryType.INSIDE_JOKE}:
            return MemoryLayer.L5_RELATIONSHIP_HISTORY
        if context_tag in {
            ContextTag.MILESTONE,
            ContextTag.TURNING_POINT,
            ContextTag.UNRESOLVED_THREAD,
            ContextTag.INSIDE_JOKE,
        }:
            return MemoryLayer.L5_RELATIONSHIP_HISTORY
        if (
            memory_type in {MemoryType.SHARED_EPISODE, MemoryType.EMOTIONAL_MOMENT, MemoryType.EMOTIONAL_PREFERENCE}
            or context_tag in {ContextTag.VULNERABLE_MOMENT, ContextTag.SHARED_CELEBRATION, ContextTag.COMFORT_MOMENT}
        ):
            if relationship.stage in {RelationshipStage.INTEGRATING, RelationshipStage.BONDING} or score >= 0.80:
                return MemoryLayer.L5_RELATIONSHIP_HISTORY
            return MemoryLayer.L3_RELATIONAL
        if memory_type == MemoryType.CONFLICT:
            return MemoryLayer.L3_RELATIONAL if score >= 0.50 else MemoryLayer.L2_EPISODIC
        if score >= 0.50:
            return MemoryLayer.L2_EPISODIC
        return MemoryLayer.L1_IMMEDIATE

    def _storage_layer_counts(self, memories: list[MemoryRecord]) -> dict[str, int]:
        counts = {layer.value: 0 for layer in MemoryLayer}
        for memory in memories:
            counts[memory.storage_layer.value] = counts.get(memory.storage_layer.value, 0) + 1
        return counts

    def _set_milestone_confirmation(
        self,
        memory: MemoryRecord,
        status: str,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or memory.created_at
        existing = dict(memory.metadata.get("milestone_confirmation", {}))
        existing.setdefault("detected_at", memory.created_at.isoformat())
        existing["status"] = status
        existing["reason"] = reason
        if status == "CONFIRMED":
            existing["confirmed_at"] = now.isoformat()
        if status == "REJECTED":
            existing["rejected_at"] = now.isoformat()
        memory.metadata["milestone_confirmation"] = existing
        return existing

    def _filter_active_candidates(
        self,
        relationship: Relationship,
        candidates: list[dict[str, Any]],
        current_text: str,
        current_signals: TurnSignals | None,
        now: datetime,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if any(word in current_text for word in self._interruption_words()):
            self._log_active_suppression(relationship, "user_requested_no_interruption", candidates, now)
            return [candidate for candidate in candidates if candidate["type"] == "baseline_care" and self._is_level3_care(relationship)]
        if self._recent_interruption_requested(relationship):
            self._log_active_suppression(relationship, "recent_user_requested_no_interruption", candidates, now)
            return [candidate for candidate in candidates if candidate["type"] == "baseline_care" and self._is_level3_care(relationship)]

        recent_active_count = sum(
            1
            for item in relationship.active_behavior_log
            if now - _dt(item.get("at")) <= timedelta(hours=2)
        )
        max_items = 1 if relationship.stage in {RelationshipStage.INITIATING, RelationshipStage.EXPERIMENTING} else relationship.preferences.max_active_per_session
        if recent_active_count >= max_items:
            remaining = [
                candidate
                for candidate in candidates
                if candidate["type"] == "baseline_care" and self._is_level3_care(relationship) and not self._level3_logged_today(relationship, now)
            ]
            if not remaining:
                self._log_active_suppression(relationship, "session_active_limit", candidates, now)
            return remaining

        high_emotion = bool(current_signals and current_signals.emotion_intensity >= 0.75)
        filtered: list[dict[str, Any]] = []
        suppressed: list[dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda item: item.get("priority", 100)):
            candidate_type = candidate["type"]
            if high_emotion and not (candidate_type == "baseline_care" and self._is_level3_care(relationship)):
                suppressed.append(candidate)
                continue
            if self._active_type_muted(relationship, candidate_type, now):
                suppressed.append(candidate)
                continue
            if candidate_type in {"shared_topic_reactivation", "anniversary"} and relationship.preferences.nostalgia_tendency <= 0.0:
                suppressed.append(candidate)
                continue
            if candidate_type == "inside_joke" and relationship.preferences.surprise_tendency <= 0.0:
                suppressed.append(candidate)
                continue
            if candidate_type == "emotional_resonance" and relationship.preferences.depth_tendency <= 0.0:
                suppressed.append(candidate)
                continue
            if self._active_candidate_on_cooldown(relationship, candidate, now):
                suppressed.append(candidate)
                continue
            if self._active_candidate_needs_topic_fit(candidate_type):
                relevance = self._active_topic_relevance(candidate, current_text)
                candidate["topic_relevance"] = relevance
                if relevance < 0.60:
                    suppressed.append(candidate)
                    continue
            filtered.append(candidate)
        if suppressed:
            reason = "high_emotion_density" if high_emotion else (
                "topic_mismatch" if any(item.get("topic_relevance", 1.0) < 0.60 for item in suppressed) else (
                    "same_memory_active_interval"
                    if any(item.get("cooldown_reason") == "same_memory_active_interval" for item in suppressed)
                    else "preference_or_feedback_filter"
                )
            )
            self._log_active_suppression(relationship, reason, suppressed, now)
        for candidate in filtered:
            self._pending_active_metadata[candidate["text"]] = candidate
        return filtered

    def _active_candidate_needs_topic_fit(self, candidate_type: str) -> bool:
        return candidate_type in {"shared_topic_reactivation", "implicit_topic", "inside_joke", "emotional_resonance"}

    def _active_topic_relevance(self, candidate: dict[str, Any], current_text: str) -> float:
        if not current_text:
            return 1.0
        if any(word in current_text for word in ["近况", "回顾", "聊聊", "说说", "想起", "之前"]):
            return 0.60
        candidate_text = str(candidate.get("text", ""))
        memory = self.memories.get(candidate.get("memory_id", ""))
        source_text = memory.content if memory else candidate_text
        if source_text and source_text in current_text:
            return 1.0
        current_themes = set(self._themes(current_text))
        candidate_themes = set(self._themes(source_text))
        if current_themes.intersection(candidate_themes):
            return max(0.60, lexical_similarity(current_text, source_text))
        return lexical_similarity(current_text, source_text)

    def _log_active_suppression(
        self, relationship: Relationship, reason: str, candidates: list[dict[str, Any]], now: datetime
    ) -> None:
        self.deviation_log.append(
            {
                "type": "active_suppressed",
                "relationship_id": relationship.relationship_id,
                "reason": reason,
                "candidate_types": [candidate.get("type") for candidate in candidates],
                "memory_ids": [candidate.get("memory_id") for candidate in candidates if candidate.get("memory_id")],
                "candidates": [
                    {
                        "type": candidate.get("type"),
                        "memory_id": candidate.get("memory_id"),
                        "topic_id": candidate.get("topic_id"),
                        "reason_text": str(candidate.get("text", ""))[:120],
                        "priority": candidate.get("priority"),
                        "topic_relevance": candidate.get("topic_relevance"),
                        "cooldown_reason": candidate.get("cooldown_reason"),
                        "cooldown_until": candidate.get("cooldown_until"),
                        "inferred": candidate.get("inferred"),
                        "confidence": candidate.get("confidence"),
                    }
                    for candidate in candidates[:10]
                ],
                "at": now.isoformat(),
            }
        )

    def _implicit_topic_evidence_status(
        self,
        relationship: Relationship,
        topic: dict[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        source_ids = [memory_id for memory_id in topic.get("source_memory_ids", []) if isinstance(memory_id, str)]
        source_memories = [
            self.memories[memory_id]
            for memory_id in source_ids
            if memory_id in self.memories
            and self.memories[memory_id].relationship_id == relationship.relationship_id
            and not self._memory_is_recall_suppressed(self.memories[memory_id])
            and not self.memories[memory_id].metadata.get("archived")
        ]
        theme = str(topic.get("theme") or "")
        matching = [memory for memory in source_memories if theme in self._themes(memory.content)]
        confidences = [
            float(self._ensure_memory_metacognition(memory, now).get("confidence", 0.0))
            for memory in matching
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        distinct_days = len({memory.created_at.date().isoformat() for memory in matching})
        reasons: list[str] = []
        if len(matching) < 3:
            reasons.append("insufficient_live_sources")
        if avg_confidence < 0.60:
            reasons.append("low_source_confidence")
        if distinct_days < 2:
            reasons.append("single_day_pattern")
        valid = not reasons
        evidence = {
            "valid": valid,
            "source_memory_ids": [memory.memory_id for memory in matching],
            "live_source_count": len(source_memories),
            "matching_source_count": len(matching),
            "distinct_days": distinct_days,
            "average_source_confidence": avg_confidence,
            "theme": theme,
            "reasons": reasons or ["evidence_supported"],
            "checked_at": now.isoformat(),
        }
        topic["last_evidence_check"] = evidence
        return evidence

    def _fail_implicit_topic_evidence(
        self,
        relationship: Relationship,
        topic: dict[str, Any],
        evidence: dict[str, Any],
        now: datetime,
    ) -> None:
        if topic.get("status") == "EVIDENCE_FAILED":
            return
        topic["status"] = "EVIDENCE_FAILED"
        topic["evidence_failed_at"] = now.isoformat()
        topic["evidence_failure_reasons"] = evidence.get("reasons", [])
        self.deviation_log.append(
            {
                "type": "implicit_topic_evidence_failed",
                "relationship_id": relationship.relationship_id,
                "topic_id": topic.get("topic_id"),
                "theme": topic.get("theme"),
                "reasons": evidence.get("reasons", []),
                "source_memory_ids": evidence.get("source_memory_ids", []),
                "at": now.isoformat(),
            }
        )

    def _is_level3_care(self, relationship: Relationship) -> bool:
        return relationship.baseline_deviation_state.get("last_level") == 3

    def _level3_logged_today(self, relationship: Relationship, now: datetime) -> bool:
        return any(
            item.get("type") == "baseline_care"
            and item.get("level") == 3
            and _dt(item.get("at")).date() == now.date()
            for item in relationship.active_behavior_log
        )

    def _active_type_logged_today(self, relationship: Relationship, active_type: str, now: datetime) -> bool:
        return any(item.get("type") == active_type and _dt(item.get("at")).date() == now.date() for item in relationship.active_behavior_log)

    def _relationship_age_anniversary_candidate(
        self, relationship: Relationship, now: datetime
    ) -> dict[str, Any] | None:
        if self._active_type_logged_today(relationship, "anniversary", now):
            return None
        created = relationship.created_at.date()
        today = now.date()
        if today <= created or today.day != created.day:
            return None
        months = (today.year - created.year) * 12 + today.month - created.month
        if months < 1:
            return None
        years = months // 12
        if months % 12 == 0 and years >= 1:
            label = f"{years}年"
            granularity = "year"
            count = years
        else:
            label = f"{months}个月"
            granularity = "month"
            count = months
        current_age_days = max(relationship.relationship_age, (today - created).days)
        return {
            "type": "anniversary",
            "text": f"关系纪念日：今天是我们认识{label}了",
            "topic_id": f"relationship_age_anniversary:{granularity}:{count}",
            "priority": 30,
            "evidence": {
                "created_at": relationship.created_at.isoformat(),
                "relationship_age_days": current_age_days,
                "anniversary_granularity": granularity,
                "anniversary_count": count,
            },
            "inferred": False,
            "confidence": 1.0,
        }

    def _active_candidate_on_cooldown(self, relationship: Relationship, candidate: dict[str, Any], now: datetime) -> bool:
        key = self._active_candidate_key(candidate)
        cooldown_until = relationship.active_feedback_state.get("cooldowns", {}).get(key)
        if cooldown_until and now < _dt(cooldown_until):
            candidate["cooldown_reason"] = "negative_feedback_cooldown"
            candidate["cooldown_until"] = cooldown_until
            return True
        interval_until = self._same_memory_active_interval_until(relationship, candidate, now)
        if interval_until:
            candidate["cooldown_reason"] = "same_memory_active_interval"
            candidate["cooldown_until"] = interval_until.isoformat()
            return True
        return False

    def _same_memory_active_interval_until(
        self, relationship: Relationship, candidate: dict[str, Any], now: datetime
    ) -> datetime | None:
        memory_id = candidate.get("memory_id")
        if not memory_id or candidate.get("type") == "commitment_reminder":
            return None
        latest: datetime | None = None
        for item in relationship.active_behavior_log:
            if item.get("memory_id") != memory_id:
                continue
            at = _dt(item.get("at"))
            if at <= now and (latest is None or at > latest):
                latest = at
        if latest is None:
            return None
        until = latest + timedelta(days=30)
        return until if now < until else None

    def _active_type_muted(self, relationship: Relationship, active_type: str, now: datetime) -> bool:
        mute = relationship.active_feedback_state.get("type_mutes", {}).get(active_type)
        if not isinstance(mute, dict):
            return False
        until = mute.get("until")
        if not until:
            return False
        if now < _dt(until):
            return True
        relationship.active_feedback_state.get("type_mutes", {}).pop(active_type, None)
        return False

    def _active_candidate_key(self, candidate: dict[str, Any]) -> str:
        return self._active_feedback_key(
            candidate.get("type", "unknown"),
            candidate.get("memory_id") or candidate.get("topic_id"),
            candidate.get("text", ""),
        )

    def _active_feedback_key(self, active_type: str | None, memory_id: str | None, reason: str) -> str:
        return f"{active_type or 'unknown'}:{memory_id or reason[:48]}"

    def _memory_is_recall_suppressed(self, memory: MemoryRecord | None) -> bool:
        if memory is None:
            return False
        boundary = memory.metadata.get("recall_boundary")
        return bool(isinstance(boundary, dict) and boundary.get("suppressed"))

    def _memory_pending_sensitive_confirmation(self, memory: MemoryRecord) -> bool:
        marker = memory.metadata.get("critical_tombstone_remention")
        return bool(
            isinstance(marker, dict)
            and marker.get("requires_user_confirmation")
            and marker.get("recording_status") != "USER_CONFIRMED"
        )

    def _memory_metacognition(
        self,
        memory: MemoryRecord,
        *,
        source_kind: str,
        ai_analysis: dict[str, Any] | None = None,
        human_verified: bool = False,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        source_confidence = 0.95 if human_verified or source_kind == "manual_injection" else 0.78
        interpretation_confidence = 0.75
        if ai_analysis is not None:
            try:
                interpretation_confidence = clamp(float(ai_analysis.get("importance", memory.importance) or memory.importance))
            except (TypeError, ValueError):
                interpretation_confidence = min(interpretation_confidence, 0.55)
            reason = str(ai_analysis.get("reason", ""))
            if "fallback" in reason or "failed" in reason:
                interpretation_confidence = min(interpretation_confidence, 0.62)
            if ai_analysis.get("ai_sanitization", {}).get("changed"):
                interpretation_confidence = min(interpretation_confidence, 0.55)
        if memory.context_tag == ContextTag.CONFLICT or memory.memory_type == MemoryType.CONFLICT:
            interpretation_confidence = min(interpretation_confidence, 0.70)
        confidence = clamp(0.60 * source_confidence + 0.25 * interpretation_confidence + 0.15 * memory.importance)
        return {
            "confidence": confidence,
            "source_kind": source_kind,
            "source_confidence": source_confidence,
            "interpretation_confidence": interpretation_confidence,
            "human_verified": human_verified,
            "evidence_count": 1,
            "last_calibrated_at": now.isoformat(),
            "uncertainty_action": self._uncertainty_action(confidence),
            "score_multiplier": self._confidence_score_multiplier(confidence),
        }

    def _ensure_memory_metacognition(self, memory: MemoryRecord, now: datetime | None = None) -> dict[str, Any]:
        metacognition = memory.metadata.get("metacognition")
        if not isinstance(metacognition, dict):
            metacognition = self._memory_metacognition(memory, source_kind="legacy_or_unknown", now=now)
            memory.metadata["metacognition"] = metacognition
        confidence = clamp(float(metacognition.get("confidence", 0.65)))
        if memory.metadata.get("recall_boundary", {}).get("suppressed"):
            confidence = min(confidence, 0.45)
        if isinstance(memory.metadata.get("supersession"), dict):
            confidence = min(confidence, 0.50)
        if memory.metadata.get("tag_versions") or memory.metadata.get("versions"):
            confidence = min(1.0, confidence + 0.05)
        active_time_conflicts = self._active_time_conflicts(memory)
        if active_time_conflicts:
            confidence = min(confidence, 0.55)
            metacognition["needs_clarification"] = True
            metacognition["time_conflict_count"] = len(active_time_conflicts)
        elif metacognition.get("time_conflict_count"):
            metacognition["time_conflict_count"] = 0
        metacognition["confidence"] = confidence
        if isinstance(memory.metadata.get("supersession"), dict):
            metacognition["superseded"] = True
            metacognition["uncertainty_action"] = "prefer_newer_version"
        else:
            metacognition["uncertainty_action"] = self._uncertainty_action(confidence)
        metacognition["score_multiplier"] = self._confidence_score_multiplier(confidence)
        return metacognition

    def _mark_memory_verified(self, memory: MemoryRecord, *, reason: str, now: datetime) -> dict[str, Any]:
        metacognition = self._ensure_memory_metacognition(memory, now)
        metacognition["human_verified"] = True
        metacognition["verification_reason"] = reason
        metacognition["verified_at"] = now.isoformat()
        metacognition["confidence"] = max(float(metacognition.get("confidence", 0.0)), 0.95)
        metacognition["source_confidence"] = max(float(metacognition.get("source_confidence", 0.0)), 0.95)
        metacognition.pop("pending_sensitive_remention_confirmation", None)
        if metacognition.get("critical_tombstone_remention") and metacognition.get("needs_clarification"):
            metacognition.pop("needs_clarification", None)
        metacognition["uncertainty_action"] = self._uncertainty_action(metacognition["confidence"])
        metacognition["score_multiplier"] = self._confidence_score_multiplier(metacognition["confidence"])
        return metacognition

    def _calibration_summary(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(history)
        correct = len([item for item in history if item.get("outcome") == "correct"])
        incorrect = len([item for item in history if item.get("outcome") == "incorrect"])
        uncertain = len([item for item in history if item.get("outcome") == "uncertain"])
        rated = correct + incorrect
        return {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "uncertain": uncertain,
            "accuracy": (correct / rated) if rated else None,
        }

    def _relationship_calibration_summary(self, memories: list[MemoryRecord]) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        for memory in memories:
            metacognition = self._ensure_memory_metacognition(memory)
            history.extend(
                item
                for item in metacognition.get("calibration_history", [])
                if isinstance(item, dict)
            )
        return self._calibration_summary(history)

    def _apply_reconsolidation_feedback(
        self,
        memory: MemoryRecord,
        outcome: str,
        *,
        reason: str,
        now: datetime,
    ) -> dict[str, Any]:
        reinforcement = memory.metadata.get("retrieval_reinforcement")
        if not isinstance(reinforcement, dict) or not reinforcement.get("last_retrieved_at"):
            return {"applied": False, "reason": "not_recently_retrieved"}
        try:
            last_retrieved_at = datetime.fromisoformat(str(reinforcement["last_retrieved_at"]))
        except ValueError:
            return {"applied": False, "reason": "invalid_last_retrieved_at"}
        window_hours = 12
        elapsed = now - last_retrieved_at
        if elapsed < timedelta(0) or elapsed > timedelta(hours=window_hours):
            return {
                "applied": False,
                "reason": "outside_reconsolidation_window",
                "last_retrieved_at": last_retrieved_at.isoformat(),
                "window_hours": window_hours,
            }
        before = memory.base_weight
        state = memory.metadata.setdefault("reconsolidation", {})
        history = state.setdefault("history", [])
        if outcome == "correct":
            memory.base_weight = clamp(memory.base_weight + 0.05, high=1.0)
            action = "strengthened"
        elif outcome == "incorrect":
            memory.base_weight = max(0.05, memory.base_weight * 0.70)
            memory.metadata.setdefault("versions", []).append(
                {
                    "reason": reason,
                    "outcome": outcome,
                    "reconsolidation": True,
                    "content_sealed": self._seal_audit_text(memory.content),
                    "at": now.isoformat(),
                }
            )
            action = "weakened_needs_revision"
        else:
            memory.base_weight = max(0.05, memory.base_weight * 0.90)
            action = "softened_uncertain"
        memory.updated_at = now
        entry = {
            "at": now.isoformat(),
            "outcome": outcome,
            "action": action,
            "reason": reason,
            "last_retrieved_at": last_retrieved_at.isoformat(),
            "window_hours": window_hours,
            "base_weight_before": before,
            "base_weight_after": memory.base_weight,
        }
        history.append(entry)
        state["history"] = history[-20:]
        state["last_action"] = action
        state["last_feedback_at"] = now.isoformat()
        self.deviation_log.append(
            {
                "type": "memory_reconsolidated",
                "relationship_id": memory.relationship_id,
                "memory_id": memory.memory_id,
                "outcome": outcome,
                "action": action,
                "base_weight_before": before,
                "base_weight_after": memory.base_weight,
                "at": now.isoformat(),
            }
        )
        return {"applied": True, **entry}

    def _evaluate_perturbation_replay(self, memory: MemoryRecord, now: datetime) -> dict[str, Any]:
        variants = self._replay_perturbation_variants(memory.content)
        variant_scores = [self._replay_variant_stability(memory, variant) for variant in variants]
        stability = sum(variant_scores) / len(variant_scores) if variant_scores else 1.0
        state = memory.metadata.setdefault("replay_evaluation", {})
        state["last_replayed_at"] = now.isoformat()
        state["faithful_replay_count"] = int(state.get("faithful_replay_count", 0) or 0) + 1
        state["perturbation_replay_count"] = int(state.get("perturbation_replay_count", 0) or 0) + len(variants)
        state["perturbation_ratio_target"] = 0.30
        state["stability_score"] = stability
        state["variant_scores"] = variant_scores
        state["variant_seals"] = [self._seal_audit_text(variant) for variant in variants]
        state["plaintext_variants_retained"] = False
        state["overfit_risk"] = "LOW" if stability >= 0.55 else "MEDIUM" if stability >= 0.35 else "HIGH"
        embedding_meta = memory.metadata.get("embeddings", {}).get("metacognition")
        if isinstance(embedding_meta, dict):
            before = float(embedding_meta.get("confidence", 0.65) or 0.65)
            if stability >= 0.55:
                embedding_meta["confidence"] = clamp(before + 0.03)
            elif stability < 0.35:
                embedding_meta["confidence"] = max(0.30, before - 0.08)
            embedding_meta["last_replay_stability"] = stability
            embedding_meta["perturbation_replay_checked_at"] = now.isoformat()
        return state

    def _replay_perturbation_variants(self, content: str) -> list[str]:
        replacements = [
            ("喜欢", "偏好"),
            ("讨厌", "不想接触"),
            ("医生", "医疗人员"),
            ("医院", "医疗机构"),
            ("项目", "计划"),
            ("工作", "事务"),
            ("焦虑", "担心"),
            ("开心", "高兴"),
            ("庆祝", "纪念"),
            ("约定", "承诺"),
            ("提醒", "记得"),
        ]
        variants: list[str] = []
        for source, target in replacements:
            if source in content:
                variants.append(content.replace(source, target, 1))
                break
        compact = re.sub(r"[，。！？,.!?；;：:\s]+", "", content)
        if compact and compact != content:
            variants.append(compact)
        if not variants:
            tokens = tokenize(content)
            if len(tokens) > 1:
                variants.append(" ".join(tokens[1:] + tokens[:1]))
            else:
                variants.append(content)
        return variants[:2]

    def _replay_variant_stability(self, memory: MemoryRecord, variant: str) -> float:
        lexical = lexical_similarity(memory.content, variant)
        original_themes = set(self._themes(memory.content))
        variant_themes = set(self._themes(variant))
        theme_overlap = len(original_themes.intersection(variant_themes)) / max(1, len(original_themes.union(variant_themes)))
        original_tokens = set(tokenize(memory.content))
        variant_tokens = set(tokenize(variant))
        token_overlap = len(original_tokens.intersection(variant_tokens)) / max(1, len(original_tokens.union(variant_tokens)))
        return clamp(0.45 * lexical + 0.35 * theme_overlap + 0.20 * token_overlap)

    def _retention_calibration_summary(self, relationship: Relationship) -> dict[str, Any]:
        state = relationship.retention_calibration_state if isinstance(relationship.retention_calibration_state, dict) else {}
        history = [item for item in state.get("history", []) if isinstance(item, dict)]
        counts = state.get("feedback_counts", {})
        if not isinstance(counts, dict):
            counts = {}
        try:
            offset = float(state.get("multiplier_offset", 0.0) or 0.0)
        except (TypeError, ValueError):
            offset = 0.0
        return {
            "enabled": relationship.preferences.reverse_decay_enabled,
            "base_stage_multiplier": relationship.retention_multiplier,
            "feedback_multiplier": retention_calibration_multiplier(relationship),
            "multiplier_offset": clamp(offset, -0.30, 0.30),
            "feedback_counts": counts,
            "last_feedback_at": state.get("last_feedback_at"),
            "recent_feedback": history[-10:],
        }

    def _ai_runtime_note(self, configuration: dict[str, Any]) -> str:
        kind = configuration.get("participation_kind")
        if kind == "external_model":
            return "当前配置为外部 AI/LLM 直接参与记忆判断。"
        if kind == "external_with_local_fallback":
            return "当前配置为外部 AI 优先；外部失败时会回退到本地启发式 AI，并在 ai_decision_log 中记录。"
        if kind == "local_heuristic":
            return "当前配置为本地启发式 AI：会参与 MemoryAI 接口，但不是外部大模型。要看到真实 LLM 参与，需要配置 MEMORY_AI_PROVIDER。"
        return "当前 AI 参与类型未知，请查看 ai_configuration。"

    def _external_ai_configured(self, configuration: dict[str, Any]) -> bool:
        return configuration.get("participation_kind") in {
            "external_http_worker",
            "external_model",
            "external_with_local_fallback",
        }

    def _external_ai_used_recently(self, decisions: list[dict[str, Any]]) -> bool:
        return any(
            provider_participation_kind((item.get("ai_call") or {}).get("used_provider") or item.get("provider"))
            in {"external_http_worker", "external_model"}
            for item in decisions
        )

    def _uncertainty_action(self, confidence: float) -> str:
        if confidence >= 0.80:
            return "state_directly"
        if confidence >= 0.55:
            return "include_source_hint"
        return "ask_or_qualify"

    def _confidence_score_multiplier(self, confidence: float) -> float:
        if confidence >= 0.80:
            return 1.0
        if confidence >= 0.55:
            return 0.85
        return 0.60

    def _attach_source_time(self, memory: MemoryRecord, text: str, now: datetime) -> None:
        source_time = self._normalize_source_time(text, now)
        if not source_time:
            return
        memory.metadata["source_time"] = source_time
        metacognition = self._ensure_memory_metacognition(memory, now)
        metacognition["timestamp_confidence"] = source_time["confidence"]
        metacognition["timestamp_precision"] = source_time["precision"]

    def _detect_time_conflicts(self, relationship: Relationship, memory: MemoryRecord, now: datetime) -> None:
        current_interval = self._source_time_interval(memory)
        if current_interval is None:
            return
        current_start, current_end, current_confidence = current_interval
        if current_confidence < 0.70:
            return
        for other in self.memories.values():
            if other.memory_id == memory.memory_id or other.relationship_id != relationship.relationship_id:
                continue
            if self._memory_is_recall_suppressed(other):
                continue
            other_interval = self._source_time_interval(other)
            if other_interval is None:
                continue
            other_start, other_end, other_confidence = other_interval
            if other_confidence < 0.70:
                continue
            relevance = self._time_conflict_relevance(memory, other)
            if relevance < 0.55:
                continue
            gap_days = self._source_time_gap_days(current_start, current_end, other_start, other_end)
            if gap_days < 1.0:
                continue
            assessment = self._assess_time_conflict_with_ai(
                relationship,
                memory,
                other,
                relevance=relevance,
                gap_days=gap_days,
                now=now,
            )
            if not assessment.get("is_conflict"):
                continue
            self._mark_time_conflict(memory, other, relevance=relevance, gap_days=gap_days, ai_assessment=assessment, now=now)

    def _detect_preference_supersession(self, relationship: Relationship, memory: MemoryRecord, now: datetime) -> None:
        current = self._preference_statement(memory.content)
        if current is None:
            return
        superseded: list[dict[str, Any]] = []
        for other in sorted(self.memories.values(), key=lambda item: item.created_at, reverse=True):
            if other.memory_id == memory.memory_id or other.relationship_id != relationship.relationship_id:
                continue
            if other.created_at > memory.created_at:
                continue
            if self._memory_is_recall_suppressed(other) or other.metadata.get("archived"):
                continue
            if is_trust_bias_protected(other):
                continue
            previous = self._preference_statement(other.content)
            if previous is None:
                continue
            if previous["subject"] != current["subject"] or previous["polarity"] == current["polarity"]:
                continue
            supersession = {
                "status": "SUPERSEDED",
                "superseded_by": memory.memory_id,
                "superseded_at": now.isoformat(),
                "reason": "preference_polarity_changed",
                "subject": current["subject"],
                "previous_polarity": previous["polarity"],
                "new_polarity": current["polarity"],
                "old_content_sealed": self._seal_audit_text(other.content),
                "new_content_sealed": self._seal_audit_text(memory.content),
            }
            other.metadata["supersession"] = supersession
            other.metadata.setdefault("version_tree", []).append(dict(supersession))
            other.base_weight = max(0.05, other.base_weight * 0.60)
            other.updated_at = now
            metacognition = self._ensure_memory_metacognition(other, now)
            metacognition["superseded"] = True
            metacognition["uncertainty_action"] = "prefer_newer_version"
            superseded.append(
                {
                    "memory_id": other.memory_id,
                    "subject": current["subject"],
                    "previous_polarity": previous["polarity"],
                    "base_weight_after": other.base_weight,
                }
            )
        if not superseded:
            return
        memory.metadata["current_version_of"] = [item["memory_id"] for item in superseded]
        memory.metadata.setdefault("version_tree", []).append(
            {
                "status": "ACTIVE_HEAD",
                "reason": "preference_polarity_changed",
                "subject": current["subject"],
                "polarity": current["polarity"],
                "previous_memory_ids": [item["memory_id"] for item in superseded],
                "created_at": now.isoformat(),
            }
        )
        self.deviation_log.append(
            {
                "type": "memory_superseded",
                "relationship_id": relationship.relationship_id,
                "active_memory_id": memory.memory_id,
                "superseded_memory_ids": [item["memory_id"] for item in superseded],
                "subject": current["subject"],
                "reason": "preference_polarity_changed",
                "at": now.isoformat(),
            }
        )

    def _preference_statement(self, text: str) -> dict[str, str] | None:
        normalized = re.sub(r"[，。！？,.!?；;：:（）()\s]+", "", text)
        patterns = [
            ("negative", r"(?:不喜欢|讨厌|不爱|不想吃|不吃)([\w\u4e00-\u9fff]{1,24})"),
            ("positive", r"(?:喜欢|爱吃|爱)([\w\u4e00-\u9fff]{1,24})"),
        ]
        for polarity, pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            subject = match.group(1)
            subject = re.sub(r"(了|啦|呀|啊|现在|最近|以前|之前)$", "", subject)
            if not subject or subject in {"我", "你", "我们"}:
                continue
            return {"polarity": polarity, "subject": subject}
        return None

    def _source_time_interval(self, memory: MemoryRecord) -> tuple[datetime, datetime, float] | None:
        source_time = memory.metadata.get("source_time")
        if not isinstance(source_time, dict):
            return None
        precision = str(source_time.get("precision", ""))
        if precision in {"open_past", "three_days"}:
            return None
        try:
            start = _dt(source_time.get("start"))
            end = _dt(source_time.get("end"))
            confidence = float(source_time.get("confidence", 0.0))
        except (TypeError, ValueError):
            return None
        return start, end, confidence

    def _time_conflict_relevance(self, first: MemoryRecord, second: MemoryRecord) -> float:
        first_text = self._content_without_time_phrase(first.content)
        second_text = self._content_without_time_phrase(second.content)
        lexical = lexical_similarity(first_text, second_text)
        shared_themes = set(self._themes(first_text)).intersection(self._themes(second_text))
        theme_bonus = 0.12 if shared_themes and "日常" not in shared_themes else 0.0
        same_type_bonus = 0.08 if first.memory_type == second.memory_type else 0.0
        return clamp(lexical + theme_bonus + same_type_bonus)

    def _content_without_time_phrase(self, text: str) -> str:
        cleaned = re.sub(r"20\d{2}-\d{1,2}-\d{1,2}", "", text)
        for phrase in ("昨天", "刚才", "上周", "上个月", "去年", "今年", "最近", "以前", "之前"):
            cleaned = cleaned.replace(phrase, "")
        return cleaned.strip()

    def _source_time_gap_days(
        self,
        first_start: datetime,
        first_end: datetime,
        second_start: datetime,
        second_end: datetime,
    ) -> float:
        if first_end <= second_start:
            return (second_start - first_end).total_seconds() / 86400
        if second_end <= first_start:
            return (first_start - second_end).total_seconds() / 86400
        return 0.0

    def _assess_time_conflict_with_ai(
        self,
        relationship: Relationship,
        memory: MemoryRecord,
        other: MemoryRecord,
        *,
        relevance: float,
        gap_days: float,
        now: datetime,
    ) -> dict[str, Any]:
        candidate = {
            "memory": {
                "memory_id": memory.memory_id,
                "content": memory.content,
                "source_time": memory.metadata.get("source_time"),
                "memory_type": memory.memory_type.value,
                "context_tag": memory.context_tag.value,
            },
            "conflicting_memory": {
                "memory_id": other.memory_id,
                "content": other.content,
                "source_time": other.metadata.get("source_time"),
                "memory_type": other.memory_type.value,
                "context_tag": other.context_tag.value,
            },
            "relevance": round(relevance, 3),
            "gap_days": round(gap_days, 2),
        }
        assessor = getattr(self.ai, "assess_time_conflict", None)
        if callable(assessor):
            assessment = assessor(candidate, self._relationship_context(relationship))
        else:
            assessment = {
                "is_conflict": relevance >= 0.55 and gap_days >= 1.0,
                "confidence": min(0.90, 0.45 + relevance * 0.35 + min(gap_days, 30.0) / 30.0 * 0.10),
                "reason": "local heuristic time conflict assessment",
            }
        normalized = {
            "is_conflict": bool(assessment.get("is_conflict")),
            "confidence": clamp(float(assessment.get("confidence", 0.60) or 0.60)),
            "reason": str(assessment.get("reason", ""))[:200],
        }
        if "clarification_question" in assessment:
            normalized["clarification_question"] = str(assessment.get("clarification_question", ""))[:200]
        self._log_ai_decision(
            relationship.relationship_id,
            task="assess_time_conflict",
            input_summary={
                "memory_id": memory.memory_id,
                "conflicting_memory_id": other.memory_id,
                "relevance": round(relevance, 3),
                "gap_days": round(gap_days, 2),
                "memory_chars": len(memory.content),
                "conflicting_memory_chars": len(other.content),
            },
            output_summary=normalized,
            now=now,
        )
        return normalized

    def _mark_time_conflict(
        self,
        memory: MemoryRecord,
        other: MemoryRecord,
        *,
        relevance: float,
        gap_days: float,
        ai_assessment: dict[str, Any],
        now: datetime,
    ) -> None:
        conflict_id = new_id("timeconflict")
        payload = {
            "conflict_id": conflict_id,
            "memory_id": memory.memory_id,
            "conflicting_memory_id": other.memory_id,
            "source_time": memory.metadata.get("source_time"),
            "conflicting_source_time": other.metadata.get("source_time"),
            "relevance": round(relevance, 3),
            "gap_days": round(gap_days, 2),
            "ai_assessment": ai_assessment,
            "status": "NEEDS_CLARIFICATION",
            "detected_at": now.isoformat(),
        }
        reverse_payload = {
            **payload,
            "memory_id": other.memory_id,
            "conflicting_memory_id": memory.memory_id,
            "source_time": other.metadata.get("source_time"),
            "conflicting_source_time": memory.metadata.get("source_time"),
        }
        self._append_unique_time_conflict(memory, payload)
        self._append_unique_time_conflict(other, reverse_payload)
        for item in (memory, other):
            metacognition = self._ensure_memory_metacognition(item, now)
            metacognition["needs_clarification"] = True
            metacognition["time_conflict_count"] = len(item.metadata.get("time_conflicts", []))
            metacognition["confidence"] = min(float(metacognition.get("confidence", 0.0)), 0.55)
            metacognition["uncertainty_action"] = self._uncertainty_action(metacognition["confidence"])
            metacognition["score_multiplier"] = self._confidence_score_multiplier(metacognition["confidence"])
        self.deviation_log.append(
            {
                "type": "time_conflict_detected",
                "relationship_id": memory.relationship_id,
                "conflict_id": conflict_id,
                "memory_id": memory.memory_id,
                "conflicting_memory_id": other.memory_id,
                "relevance": round(relevance, 3),
                "gap_days": round(gap_days, 2),
                "ai_confidence": ai_assessment.get("confidence"),
                "status": "NEEDS_CLARIFICATION",
                "at": now.isoformat(),
            }
        )

    def _append_unique_time_conflict(self, memory: MemoryRecord, payload: dict[str, Any]) -> None:
        conflicts = memory.metadata.setdefault("time_conflicts", [])
        if any(item.get("conflicting_memory_id") == payload["conflicting_memory_id"] for item in conflicts if isinstance(item, dict)):
            return
        conflicts.append(payload)

    def _active_time_conflicts(self, memory: MemoryRecord) -> list[dict[str, Any]]:
        return [
            item
            for item in memory.metadata.get("time_conflicts", [])
            if isinstance(item, dict) and item.get("status") == "NEEDS_CLARIFICATION"
        ]

    def _normalize_source_time(self, text: str, now: datetime) -> dict[str, Any] | None:
        iso = re.search(r"(20\d{2})-(\d{1,2})-(\d{1,2})", text)
        if iso:
            try:
                date = datetime(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)), tzinfo=now.tzinfo)
            except ValueError:
                return None
            return self._source_time_payload(iso.group(0), date, date + timedelta(days=1), "day", 0.98)
        if "昨天" in text:
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            return self._source_time_payload("昨天", start, start + timedelta(days=1), "day", 0.95)
        if "刚才" in text:
            return self._source_time_payload("刚才", now - timedelta(minutes=30), now, "hour", 0.90)
        if "上周" in text:
            start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = now.replace(hour=23, minute=59, second=59, microsecond=0)
            return self._source_time_payload("上周", start, end, "week", 0.75)
        if "上个月" in text:
            start = (now - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = now.replace(hour=23, minute=59, second=59, microsecond=0)
            return self._source_time_payload("上个月", start, end, "month", 0.60)
        if "去年" in text:
            start = datetime(now.year - 1, 1, 1, tzinfo=now.tzinfo)
            end = datetime(now.year, 1, 1, tzinfo=now.tzinfo)
            return self._source_time_payload("去年", start, end, "year", 0.70)
        if "今年" in text:
            start = datetime(now.year, 1, 1, tzinfo=now.tzinfo)
            return self._source_time_payload("今年", start, now, "year", 0.70)
        if "最近" in text:
            return self._source_time_payload("最近", now - timedelta(days=3), now, "three_days", 0.50)
        if "以前" in text or "之前" in text:
            return self._source_time_payload("以前/之前", self.created_at_floor(), now, "open_past", 0.30)
        return None

    def created_at_floor(self) -> datetime:
        earliest = [relationship.created_at for relationship in self.relationships.values()]
        return min(earliest) if earliest else utcnow()

    def _source_time_payload(self, phrase: str, start: datetime, end: datetime, precision: str, confidence: float) -> dict[str, Any]:
        return {
            "phrase": phrase,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "precision": precision,
            "confidence": confidence,
        }

    def _timestamp_score_multiplier(self, metacognition: dict[str, Any]) -> float:
        confidence = metacognition.get("timestamp_confidence")
        if confidence is None:
            return 1.0
        return 0.92 + 0.08 * clamp(float(confidence))

    def _query_metacognition(
        self,
        relationship: Relationship,
        query: str,
        results: list[RetrievalResult],
    ) -> dict[str, Any]:
        if not results:
            return {
                "coverage": 0.0,
                "confidence": 0.0,
                "result_count": 0,
                "top_score": 0.0,
                "average_memory_confidence": 0.0,
                "action": "clarify_or_ack_unknown",
                "reason": "no_retrieval_results",
            }
        top_score = max(item.score for item in results)
        avg_memory_confidence = sum(float(item.explanation["metacognition"]["confidence"]) for item in results) / len(results)
        lexical_support = max(float(item.explanation.get("lexical_similarity", 0.0)) for item in results)
        semantic_support = max(float(item.explanation.get("semantic", 0.0)) for item in results)
        topic_support = min(1.0, len(results) / 3)
        coverage = clamp(0.35 * min(1.0, top_score) + 0.25 * semantic_support + 0.20 * lexical_support + 0.20 * topic_support)
        confidence = clamp(0.55 * coverage + 0.45 * avg_memory_confidence)
        if confidence >= 0.75 and coverage >= 0.60:
            action = "answer_directly"
        elif confidence >= 0.45:
            action = "answer_with_source_hint"
        else:
            action = "clarify_or_ack_unknown"
        return {
            "coverage": coverage,
            "confidence": confidence,
            "result_count": len(results),
            "top_score": top_score,
            "average_memory_confidence": avg_memory_confidence,
            "lexical_support": lexical_support,
            "semantic_support": semantic_support,
            "action": action,
            "reason": self._query_coverage_reason(action, relationship, query),
        }

    def _query_coverage_reason(self, action: str, relationship: Relationship, query: str) -> str:
        if action == "answer_directly":
            return "retrieval_coverage_and_memory_confidence_sufficient"
        if action == "answer_with_source_hint":
            return "partial_coverage_include_source_or_time_hint"
        if relationship.interaction_count < 3:
            return "relationship_cold_start_clarify_before_answering"
        if any(word in query for word in ["是谁", "什么时候", "哪里", "地址", "号码"]):
            return "fact_question_insufficient_evidence"
        return "low_coverage_clarify_before_answering"

    def _detect_boundary_request(self, text: str) -> dict[str, str] | None:
        normalized = text.strip().lower()
        if not normalized:
            return None
        do_not_store_phrases = ["不要记住", "别记住", "不要保存", "别保存", "不要留下记录", "别留下记录"]
        do_not_recall_phrases = ["别再提", "不要再提", "别主动提", "不要主动提", "别提醒", "不要提醒", "别回忆", "不要回忆"]
        if any(phrase in normalized for phrase in do_not_store_phrases):
            return {"scope": "do_not_store", "reason": "user_requested_do_not_store"}
        if any(phrase in normalized for phrase in do_not_recall_phrases):
            return {"scope": "do_not_recall", "reason": "user_requested_do_not_recall"}
        return None

    def _apply_boundary_request(
        self,
        relationship: Relationship,
        text: str,
        boundary_request: dict[str, str],
        now: datetime,
    ) -> dict[str, Any]:
        targets = self._boundary_target_memories(relationship, text, now)
        for memory in targets:
            self.suppress_memory(
                memory.memory_id,
                reason=boundary_request["reason"],
                now=now,
                boundary_text=text,
            )
        return {
            "type": "memory_boundary_request",
            "relationship_id": relationship.relationship_id,
            "scope": boundary_request["scope"],
            "reason": boundary_request["reason"],
            "matched_memory_ids": [memory.memory_id for memory in targets],
            "text_chars": len(text),
            "at": now.isoformat(),
        }

    def _boundary_target_memories(self, relationship: Relationship, text: str, now: datetime) -> list[MemoryRecord]:
        recent = [
            memory
            for memory in self.memories.values()
            if memory.relationship_id == relationship.relationship_id
            and not self._memory_is_recall_suppressed(memory)
            and (now - memory.created_at).days <= 90
        ]
        ranked = sorted(
            recent,
            key=lambda memory: (
                lexical_similarity(text, memory.content),
                memory.created_at.timestamp(),
            ),
            reverse=True,
        )
        matched = [memory for memory in ranked if lexical_similarity(text, memory.content) >= 0.12][:3]
        if matched:
            return matched
        return ranked[:1]

    def _health_alert(
        self,
        relationship_id: str,
        risk_type: str,
        level: HealthRiskLevel,
        message: str,
        now: datetime,
        source_memory_id: str | None = None,
    ) -> HealthAlert | None:
        relationship = self.relationships[relationship_id]
        if self._health_prompt_cooldown_active(relationship, risk_type, level, now):
            self._record_health_prompt_suppression(relationship, risk_type, level, now, source_memory_id)
            return None
        existing = next(
            (
                alert
                for alert in self.health_alerts.values()
                if alert.relationship_id == relationship_id
                and alert.risk_type == risk_type
                and alert.source_memory_id == source_memory_id
                and not alert.acknowledged
            ),
            None,
        )
        if existing:
            return existing
        alert = HealthAlert(
            alert_id=new_id("health"),
            relationship_id=relationship_id,
            risk_type=risk_type,
            level=level,
            message=message,
            created_at=now,
            source_memory_id=source_memory_id,
            resources=self._health_resources(risk_type, level),
        )
        self.health_alerts[alert.alert_id] = alert
        self.deviation_log.append(
            {
                "type": "health_alert",
                "alert_id": alert.alert_id,
                "relationship_id": relationship_id,
                "risk_type": risk_type,
                "level": level.value,
                "created_at": now.isoformat(),
            }
        )
        return alert

    def _health_resources(self, risk_type: str, level: HealthRiskLevel) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = []
        if level == HealthRiskLevel.CRITICAL or risk_type == "crisis_expression":
            resources.extend(
                [
                    {
                        "type": "crisis_lifeline",
                        "region": "US",
                        "label": "988 Suicide & Crisis Lifeline",
                        "phone": "988",
                        "text": "988",
                        "chat_url": "https://988lifeline.org/chat/",
                        "availability": "24/7",
                        "note": "美国及其属地的免费、保密心理危机支持；如有即时身体危险，请联系 911 或当地急救服务。",
                    },
                    {
                        "type": "emergency_services",
                        "region": "US",
                        "label": "Emergency services",
                        "phone": "911",
                        "availability": "24/7",
                        "note": "存在即时生命危险、正在受伤或无法保持安全时使用。",
                    },
                ]
            )
        resources.append(
            {
                "type": "local_fallback",
                "region": "GLOBAL",
                "label": "当地危机热线、急救电话或可信任真人支持",
                "note": "如果你不在美国，请使用所在地区的紧急服务、危机干预热线、医生、家人或可信任朋友。",
            }
        )
        if risk_type in {
            "social_isolation",
            "perfect_attachment",
            "frequent_level3_care",
            "repeated_distress",
            "rising_long_term_interaction_density",
        }:
            resources.append(
                {
                    "type": "professional_support",
                    "region": "GLOBAL",
                    "label": "专业心理咨询或线下支持网络",
                    "note": "AI 不能替代真人关系、专业心理咨询或紧急援助。",
                }
            )
        return resources

    def _append_health_alert(
        self,
        alerts: list[HealthAlert],
        relationship_id: str,
        risk_type: str,
        level: HealthRiskLevel,
        message: str,
        now: datetime,
        source_memory_id: str | None = None,
    ) -> None:
        alert = self._health_alert(relationship_id, risk_type, level, message, now, source_memory_id)
        if alert is not None:
            alerts.append(alert)

    def _maybe_escalate_health_prompt_refusals(self, relationship_id: str, now: datetime) -> None:
        feedback_events = [
            item
            for item in self.deviation_log
            if item.get("type") == "health_alert_feedback"
            and item.get("relationship_id") == relationship_id
            and item.get("feedback") in {"ignored", "rejected"}
        ]
        recent = feedback_events[-3:]
        if len(recent) < 3:
            return
        self._health_alert(
            relationship_id,
            "health_prompt_refusal_pattern",
            HealthRiskLevel.WARNING,
            "用户已连续多次拒绝或忽略健康度提示；建议降低打扰频率，同时明确 AI 不能替代真人关系与专业支持。",
            now,
        )
        self._start_health_prompt_cooldown(relationship_id, now, recent)

    def _start_health_prompt_cooldown(
        self,
        relationship_id: str,
        now: datetime,
        feedback_events: list[dict[str, Any]],
    ) -> None:
        relationship = self.relationships[relationship_id]
        until = now + timedelta(days=30)
        existing = relationship.maintenance_signals.get("health_prompt_cooldown")
        if isinstance(existing, dict):
            existing_until = self._safe_datetime(str(existing.get("until", "")))
            if existing_until and existing_until >= until:
                return
        relationship.maintenance_signals["health_prompt_cooldown"] = {
            "status": "ACTIVE",
            "started_at": now.isoformat(),
            "until": until.isoformat(),
            "reason": "three_recent_health_prompt_ignored_or_rejected",
            "scope": "dependency_and_overuse_prompts",
            "suppressed_count": 0,
            "feedback_event_count": len(feedback_events),
        }
        self.deviation_log.append(
            {
                "type": "health_prompt_cooldown_started",
                "relationship_id": relationship_id,
                "until": until.isoformat(),
                "reason": "three_recent_health_prompt_ignored_or_rejected",
                "at": now.isoformat(),
            }
        )

    def _health_prompt_cooldown_active(
        self,
        relationship: Relationship,
        risk_type: str,
        level: HealthRiskLevel,
        now: datetime,
    ) -> bool:
        if level == HealthRiskLevel.CRITICAL:
            return False
        if risk_type in {
            "crisis_expression",
            "minor_age_clarification_needed",
            "minor_status_pending_limited",
            "minor_bonding_limited",
            "critical_tombstone_rementioned",
            "health_prompt_refusal_pattern",
        }:
            return False
        if risk_type not in {
            "overuse",
            "daily_companionship_mode",
            "perfect_attachment",
            "repeated_distress",
            "frequent_level3_care",
            "social_isolation",
            "rising_long_term_interaction_density",
        }:
            return False
        state = relationship.maintenance_signals.get("health_prompt_cooldown")
        if not isinstance(state, dict) or state.get("status") != "ACTIVE":
            return False
        until = self._safe_datetime(str(state.get("until", "")))
        if until is None:
            return False
        if now >= until:
            state["status"] = "EXPIRED"
            state["expired_at"] = now.isoformat()
            return False
        return True

    def _safe_datetime(self, value: str) -> datetime | None:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _record_health_prompt_suppression(
        self,
        relationship: Relationship,
        risk_type: str,
        level: HealthRiskLevel,
        now: datetime,
        source_memory_id: str | None,
    ) -> None:
        state = relationship.maintenance_signals.get("health_prompt_cooldown")
        if isinstance(state, dict):
            state["suppressed_count"] = int(state.get("suppressed_count", 0) or 0) + 1
            state["last_suppressed_at"] = now.isoformat()
            state["last_suppressed_risk_type"] = risk_type
        self.deviation_log.append(
            {
                "type": "health_alert_suppressed_by_cooldown",
                "relationship_id": relationship.relationship_id,
                "risk_type": risk_type,
                "level": level.value,
                "source_memory_id": source_memory_id,
                "at": now.isoformat(),
            }
        )


    def _log_active_behavior(self, relationship: Relationship, suggestions: list[str], now: datetime) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for suggestion in suggestions:
            metadata = self._pending_active_metadata.pop(suggestion, {})
            active_type = metadata.get("type") or self._active_type_from_suggestion(suggestion)
            entry = {
                "active_id": new_id("active"),
                "at": now.isoformat(),
                "reason": suggestion,
                "type": active_type,
                "memory_id": metadata.get("memory_id"),
                "topic_id": metadata.get("topic_id"),
                "reaction": None,
            }
            for key in ("evidence", "inferred", "uncertainty_action", "confidence"):
                if key in metadata:
                    entry[key] = metadata[key]
            if entry["type"] == "baseline_care":
                entry["level"] = relationship.baseline_deviation_state.get("last_level")
            relationship.active_behavior_log.append(entry)
            entries.append(dict(entry))
        if len(relationship.active_behavior_log) > 100:
            relationship.active_behavior_log = relationship.active_behavior_log[-100:]
        return entries

    def mute_active_type(
        self,
        relationship_id: str,
        active_type: str,
        *,
        days: int = 90,
        reason: str = "user_active_type_mute",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        active_type = self._normalize_user_mutable_active_type(active_type)
        relationship = self.relationships[relationship_id]
        until = now + timedelta(days=max(1, days))
        relationship.active_feedback_state.setdefault("type_mutes", {})[active_type] = {
            "active_type": active_type,
            "until": until.isoformat(),
            "reason": reason,
            "muted_at": now.isoformat(),
        }
        event = {
            "type": "active_type_muted",
            "relationship_id": relationship_id,
            "active_type": active_type,
            "until": until.isoformat(),
            "reason": reason,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def unmute_active_type(
        self,
        relationship_id: str,
        active_type: str,
        *,
        reason: str = "user_active_type_unmute",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        active_type = self._normalize_user_mutable_active_type(active_type)
        relationship = self.relationships[relationship_id]
        previous = relationship.active_feedback_state.setdefault("type_mutes", {}).pop(active_type, None)
        event = {
            "type": "active_type_unmuted",
            "relationship_id": relationship_id,
            "active_type": active_type,
            "previous": previous,
            "reason": reason,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def _normalize_user_mutable_active_type(self, active_type: str) -> str:
        normalized = str(active_type).strip().lower().replace("-", "_")
        if normalized not in self.USER_MUTABLE_ACTIVE_TYPES:
            allowed = ", ".join(sorted(self.USER_MUTABLE_ACTIVE_TYPES))
            raise ValueError(f"active_type must be one of: {allowed}")
        return normalized

    def record_active_feedback(
        self,
        relationship_id: str,
        active_id: str,
        reaction: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        normalized = reaction.lower()
        if normalized not in {"accepted", "neutral", "ignored", "denied"}:
            raise ValueError("reaction must be accepted, neutral, ignored, or denied")
        relationship = self.relationships[relationship_id]
        entry = next((item for item in relationship.active_behavior_log if item.get("active_id") == active_id), None)
        if not entry:
            raise KeyError(f"active behavior not found: {active_id}")
        entry["reaction"] = normalized
        entry["reacted_at"] = now.isoformat()
        adjustment = self._adjust_active_preferences(relationship, entry, normalized, now)
        self.deviation_log.append(
            {
                "type": "active_feedback",
                "relationship_id": relationship_id,
                "active_id": active_id,
                "active_type": entry.get("type"),
                "reaction": normalized,
                "preference_adjustment": adjustment,
                "at": now.isoformat(),
            }
        )
        return {"active_id": active_id, "reaction": normalized, "adjustment": adjustment}

    def _adjust_active_preferences(
        self, relationship: Relationship, entry: dict[str, Any], reaction: str, now: datetime
    ) -> dict[str, Any]:
        active_type = entry.get("type")
        field = {
            "shared_topic_reactivation": "nostalgia_tendency",
            "anniversary": "nostalgia_tendency",
            "inside_joke": "surprise_tendency",
            "emotional_resonance": "depth_tendency",
            "baseline_care": "depth_tendency",
            "implicit_topic": "depth_tendency",
        }.get(active_type)
        delta = 0.03 if reaction == "accepted" else -0.05 if reaction in {"ignored", "denied"} else 0.0
        adjustment: dict[str, Any] = {}
        if field and delta:
            old_value = getattr(relationship.preferences, field)
            new_value = clamp(old_value + delta)
            setattr(relationship.preferences, field, new_value)
            adjustment[field] = {"old": old_value, "new": new_value}

        if reaction in {"ignored", "denied"}:
            key = self._active_feedback_key(active_type, entry.get("memory_id"), entry.get("reason", ""))
            negative_counts = relationship.active_feedback_state.setdefault("negative_counts", {})
            negative_counts[key] = int(negative_counts.get(key, 0)) + 1
            if active_type == "inside_joke" and entry.get("memory_id"):
                inside_joke_adjustment = self._adjust_inside_joke_feedback(entry["memory_id"], accepted=False, now=now)
                if inside_joke_adjustment:
                    adjustment["inside_joke"] = inside_joke_adjustment
            if active_type == "implicit_topic" and entry.get("topic_id"):
                adjustment["implicit_topic"] = self._adjust_implicit_topic_feedback(
                    relationship,
                    entry["topic_id"],
                    reaction,
                    now=now,
                )
            if active_type == "baseline_care":
                baseline_adjustment = self._adjust_baseline_false_positive_feedback(
                    relationship,
                    accepted=False,
                    now=now,
                )
                if baseline_adjustment:
                    adjustment["baseline_care"] = baseline_adjustment
            if negative_counts[key] >= 3:
                cooldown_until = now + timedelta(days=90)
                relationship.active_feedback_state.setdefault("cooldowns", {})[key] = cooldown_until.isoformat()
                adjustment["cooldown"] = {"key": key, "until": cooldown_until.isoformat()}
        elif reaction == "accepted":
            key = self._active_feedback_key(active_type, entry.get("memory_id"), entry.get("reason", ""))
            relationship.active_feedback_state.setdefault("negative_counts", {})[key] = 0
            if active_type == "inside_joke" and entry.get("memory_id"):
                inside_joke_adjustment = self._adjust_inside_joke_feedback(entry["memory_id"], accepted=True, now=now)
                if inside_joke_adjustment:
                    adjustment["inside_joke"] = inside_joke_adjustment
            if active_type == "implicit_topic" and entry.get("topic_id"):
                adjustment["implicit_topic"] = self._adjust_implicit_topic_feedback(
                    relationship,
                    entry["topic_id"],
                    reaction,
                    now=now,
                )
            if active_type == "baseline_care":
                baseline_adjustment = self._adjust_baseline_false_positive_feedback(
                    relationship,
                    accepted=True,
                    now=now,
                )
                if baseline_adjustment:
                    adjustment["baseline_care"] = baseline_adjustment
        return adjustment

    def _adjust_baseline_false_positive_feedback(
        self,
        relationship: Relationship,
        *,
        accepted: bool,
        now: datetime,
    ) -> dict[str, Any] | None:
        state = relationship.baseline_deviation_state
        if accepted:
            previous_false_positive = int(state.get("false_positive_streak", 0))
            previous_acceptance = int(state.get("accepted_streak", 0))
            new_acceptance = previous_acceptance + 1
            adjustment: dict[str, Any] = {
                "accepted_streak": {"old": previous_acceptance, "new": new_acceptance}
            }
            if previous_false_positive:
                state["false_positive_streak"] = 0
                adjustment["false_positive_streak"] = {"old": previous_false_positive, "new": 0}
            state["accepted_streak"] = new_acceptance
            state["last_feedback_adjusted_at"] = now.isoformat()
            if new_acceptance >= 3:
                old_offset = float(state.get("sensitivity_offset", 0.0) or 0.0)
                new_offset = max(-0.9, old_offset - 0.3)
                state["sensitivity_offset"] = new_offset
                state["accepted_streak"] = 0
                adjustment["sensitivity_offset"] = {"old": old_offset, "new": new_offset}
                adjustment["accepted_streak"]["reset_to"] = 0
                self.deviation_log.append(
                    {
                        "type": "baseline_sensitivity_adjusted",
                        "relationship_id": relationship.relationship_id,
                        "old_offset": old_offset,
                        "new_offset": new_offset,
                        "reason": "three_baseline_care_acceptances",
                        "at": now.isoformat(),
                    }
                )
            return adjustment

        previous_streak = int(state.get("false_positive_streak", 0))
        new_streak = previous_streak + 1
        state["false_positive_streak"] = new_streak
        state["accepted_streak"] = 0
        state["last_feedback_adjusted_at"] = now.isoformat()
        adjustment: dict[str, Any] = {
            "false_positive_streak": {"old": previous_streak, "new": new_streak},
            "accepted_streak": {"new": 0},
        }
        if new_streak >= 3:
            old_offset = float(state.get("sensitivity_offset", 0.0) or 0.0)
            new_offset = min(1.5, old_offset + 0.3)
            state["sensitivity_offset"] = new_offset
            state["false_positive_streak"] = 0
            adjustment["sensitivity_offset"] = {"old": old_offset, "new": new_offset}
            adjustment["false_positive_streak"]["reset_to"] = 0
            self.deviation_log.append(
                {
                    "type": "baseline_sensitivity_adjusted",
                    "relationship_id": relationship.relationship_id,
                    "old_offset": old_offset,
                    "new_offset": new_offset,
                    "reason": "three_baseline_care_false_positives",
                    "at": now.isoformat(),
                }
            )
        return adjustment

    def record_implicit_topic_feedback(
        self,
        relationship_id: str,
        topic_id: str,
        reaction: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or utcnow()
        normalized = reaction.lower()
        if normalized not in {"accepted", "neutral", "ignored", "denied"}:
            raise ValueError("reaction must be accepted, neutral, ignored, or denied")
        relationship = self.relationships[relationship_id]
        adjustment = self._adjust_implicit_topic_feedback(relationship, topic_id, normalized, now=now)
        event = {
            "type": "implicit_topic_feedback",
            "relationship_id": relationship_id,
            "topic_id": topic_id,
            "reaction": normalized,
            "adjustment": adjustment,
            "at": now.isoformat(),
        }
        self.deviation_log.append(event)
        return event

    def _adjust_implicit_topic_feedback(
        self,
        relationship: Relationship,
        topic_id: str,
        reaction: str,
        *,
        now: datetime,
    ) -> dict[str, Any]:
        topic = next((item for item in relationship.implicit_topics if item.get("topic_id") == topic_id), None)
        if not topic:
            raise KeyError(f"implicit topic not found: {topic_id}")
        feedback = topic.setdefault("feedback", {"accepted": 0, "neutral": 0, "ignored": 0, "denied": 0})
        feedback[reaction] = int(feedback.get(reaction, 0)) + 1
        topic["last_feedback_at"] = now.isoformat()
        before = {"status": topic.get("status"), "confidence": topic.get("confidence")}
        if reaction == "accepted":
            topic["status"] = "CONFIRMED"
            topic["confidence"] = min(0.98, float(topic.get("confidence", 0.0)) + 0.10)
        elif reaction == "neutral":
            topic["status"] = topic.get("status", "ACTIVE")
        elif reaction == "denied":
            topic["status"] = "MUTED"
            topic["muted_at"] = now.isoformat()
            topic["mute_reason"] = "user_denied"
        elif int(feedback.get("ignored", 0)) >= 3:
            topic["status"] = "MUTED"
            topic["muted_at"] = now.isoformat()
            topic["mute_reason"] = "ignored_three_times"
        return {"before": before, "after": {"status": topic.get("status"), "confidence": topic.get("confidence")}}

    def _adjust_inside_joke_feedback(self, memory_id: str, *, accepted: bool, now: datetime) -> dict[str, Any]:
        memory = self.memories.get(memory_id)
        if not memory or memory.memory_type != MemoryType.INSIDE_JOKE:
            return {}
        if accepted:
            memory.metadata["inside_joke_negative_feedback"] = 0
            memory.metadata["inside_joke_inactive"] = False
            memory.metadata["inside_joke_last_positive_at"] = now.isoformat()
            return {"negative_feedback": 0, "inactive": False}
        count = int(memory.metadata.get("inside_joke_negative_feedback", 0)) + 1
        memory.metadata["inside_joke_negative_feedback"] = count
        memory.metadata["inside_joke_last_negative_at"] = now.isoformat()
        result = {"negative_feedback": count, "inactive": bool(memory.metadata.get("inside_joke_inactive"))}
        if count >= 3:
            memory.metadata["inside_joke_inactive"] = True
            memory.metadata["inside_joke_deactivated_at"] = now.isoformat()
            memory.metadata["inside_joke_weight_multiplier"] = 0.3
            result["inactive"] = True
            self.deviation_log.append(
                {
                    "type": "inside_joke_deactivated",
                    "memory_id": memory_id,
                    "relationship_id": memory.relationship_id,
                    "negative_feedback": count,
                    "at": now.isoformat(),
                }
            )
        return result

    def _active_type_from_suggestion(self, suggestion: str) -> str:
        if suggestion.startswith("未完结话题"):
            return "shared_topic_reactivation"
        if suggestion.startswith("关系纪念日"):
            return "anniversary"
        if suggestion.startswith("inside joke"):
            return "inside_joke"
        if suggestion.startswith("情感共鸣"):
            return "emotional_resonance"
        if "最近还好吗" in suggestion or "不太对劲" in suggestion or "担心你" in suggestion:
            return "baseline_care"
        return "unknown"

    def _log_retrieval(
        self,
        relationship: Relationship,
        query: str,
        results: list[RetrievalResult],
        now: datetime,
        *,
        include_archived: bool,
        query_metacognition: dict[str, Any],
        association_expansions: list[dict[str, Any]],
        story_clusters: list[dict[str, Any]],
        retrieval_adaptation: dict[str, Any],
    ) -> None:
        self.retrieval_audit_log.append(
            {
                "at": now.isoformat(),
                "relationship_id": relationship.relationship_id,
                "query": query,
                "stage": relationship.stage.value,
                "trust_level": relationship.trust_level,
                "include_archived": include_archived,
                "result_count": len(results),
                "query_metacognition": query_metacognition,
                "association_expansions": association_expansions,
                "story_clusters": story_clusters,
                "retrieval_adaptation": retrieval_adaptation,
                "results": [
                    {
                        "memory_id": item.memory.memory_id,
                        "memory_type": item.memory.memory_type.value,
                        "context_tag": item.memory.context_tag.value,
                        "final_score": item.score,
                        "explanation": item.explanation,
                    }
                    for item in results
                ],
            }
        )
        if len(self.retrieval_audit_log) > 500:
            self.retrieval_audit_log = self.retrieval_audit_log[-500:]

    def _update_emotional_baseline(self, relationship: Relationship, text: str, signals: TurnSignals, now: datetime) -> None:
        baseline = relationship.emotional_baseline
        n = baseline.sample_count
        previous_memory_time = max(
            (
                memory.created_at
                for memory in self.memories.values()
                if memory.relationship_id == relationship.relationship_id and memory.created_at < now
            ),
            default=None,
        )
        baseline.sample_count += 1
        baseline.avg_sentiment = (baseline.avg_sentiment * n + signals.sentiment) / baseline.sample_count
        baseline.avg_response_length = (baseline.avg_response_length * n + len(text)) / baseline.sample_count
        baseline.vocabulary_richness = (
            baseline.vocabulary_richness * n + self._vocabulary_richness(text)
        ) / baseline.sample_count
        baseline.emoji_usage_rate = (baseline.emoji_usage_rate * n + self._emoji_count(text)) / baseline.sample_count
        baseline.exclamation_rate = (
            baseline.exclamation_rate * n + text.count("!") + text.count("！")
        ) / baseline.sample_count
        baseline.question_rate = (baseline.question_rate * n + text.count("?") + text.count("？")) / baseline.sample_count
        topic = self._baseline_topic_bucket(text, signals)
        baseline.topic_distribution = self._updated_topic_distribution(baseline.topic_distribution, topic, baseline.sample_count)
        baseline.interaction_frequency = self._baseline_interaction_frequency(relationship, now, baseline.update_window_days)
        if previous_memory_time:
            latency = (now - previous_memory_time).total_seconds()
            baseline.avg_response_latency = (baseline.avg_response_latency * n + latency) / baseline.sample_count
        baseline.baseline_confidence = clamp(baseline.sample_count / max(1, baseline.min_samples))
        baseline.last_updated = now

        recent_lengths = [len(memory.content) for memory in self.memories.values() if memory.relationship_id == relationship.relationship_id]
        if len(recent_lengths) >= 2:
            baseline.std_response_length = max(5.0, statistics.pstdev(recent_lengths))
        recent_times = sorted(
            memory.created_at for memory in self.memories.values() if memory.relationship_id == relationship.relationship_id
        )
        if len(recent_times) >= 3:
            latencies = [
                (later - earlier).total_seconds()
                for earlier, later in zip(recent_times, recent_times[1:])
                if later > earlier
            ]
            if len(latencies) >= 2:
                baseline.std_response_latency = max(60.0, statistics.pstdev(latencies))

    def _detect_deviation(
        self,
        relationship: Relationship,
        text: str,
        signals: TurnSignals | None,
        *,
        now: datetime | None = None,
    ) -> str | None:
        if not relationship.preferences.baseline_detection_enabled or not signals:
            return None
        now = now or utcnow()
        baseline: EmotionalBaseline = relationship.emotional_baseline
        if baseline.baseline_confidence < 0.5:
            return None
        dimensions = self._normalize_baseline_dimensions(relationship.preferences.baseline_detection_dimensions)
        components: dict[str, float] = {}
        if "sentiment" in dimensions:
            components["sentiment"] = 0.30 * abs(signals.sentiment - baseline.avg_sentiment) / max(baseline.std_sentiment, 0.05)
        if "response_length" in dimensions:
            components["response_length"] = 0.20 * abs(len(text) - baseline.avg_response_length) / max(baseline.std_response_length, 5.0)
        if "emoji" in dimensions:
            components["emoji"] = 0.08 * abs(text.count("🙂") - baseline.emoji_usage_rate) / 0.5
        if "exclamation" in dimensions:
            components["exclamation"] = 0.08 * abs(text.count("!") + text.count("！") - baseline.exclamation_rate) / 2.0
        if "question" in dimensions:
            components["question"] = 0.05 * abs(text.count("?") + text.count("？") - baseline.question_rate) / 2.0
        if "vocabulary" in dimensions and baseline.vocabulary_richness > 0.0:
            components["vocabulary"] = 0.10 * abs(
                self._vocabulary_richness(text) - baseline.vocabulary_richness
            ) / 0.35
        if "topic_distribution" in dimensions and baseline.topic_distribution:
            components["topic_distribution"] = 0.10 * self._topic_distribution_deviation(
                baseline.topic_distribution,
                self._baseline_topic_bucket(text, signals),
            )
        if "interaction_frequency" in dimensions and baseline.interaction_frequency > 0.0:
            current_frequency = self._baseline_interaction_frequency(
                relationship,
                now,
                baseline.update_window_days,
            )
            components["interaction_frequency"] = 0.05 * abs(
                current_frequency - baseline.interaction_frequency
            ) / max(baseline.interaction_frequency, 1.0)
        if "response_latency" in dimensions and baseline.avg_response_latency > 0.0:
            latency = self._current_response_latency(relationship, now)
            if latency is not None:
                components["response_latency"] = 0.10 * abs(
                    latency - baseline.avg_response_latency
                ) / max(baseline.std_response_latency, 60.0)
        if "arousal" in dimensions:
            components["arousal"] = 0.18 * signals.arousal
        if "self_disclosure" in dimensions:
            components["self_disclosure"] = 0.16 * signals.self_disclosure_depth
        if "personal_importance" in dimensions:
            components["personal_importance"] = 0.12 * signals.personal_importance
        score = sum(components.values())
        state = relationship.baseline_deviation_state
        sensitivity_offset = max(-0.9, min(1.5, float(state.get("sensitivity_offset", 0.0) or 0.0)))
        preference_offset = {"LOW": 0.3, "MEDIUM": 0.0, "HIGH": -0.3}.get(
            str(relationship.preferences.baseline_sensitivity).upper(),
            0.0,
        )
        level2_threshold = max(1.0, 2.0 + preference_offset + sensitivity_offset)
        level3_threshold = max(2.0, 3.0 + preference_offset + sensitivity_offset)
        if score < level2_threshold:
            self._record_baseline_recovery(relationship, score)
            return None
        level = 3 if score >= level3_threshold and relationship.preferences.level3_enabled else 2
        if state.get("last_level") == level:
            consecutive = int(state.get("consecutive_deviations", 0)) + 1
        else:
            consecutive = 1
        state.update(
            {
                "last_level": level,
                "last_score": score,
                "level2_threshold": level2_threshold,
                "level3_threshold": level3_threshold,
                "sensitivity_offset": sensitivity_offset,
                "baseline_sensitivity": relationship.preferences.baseline_sensitivity,
                "baseline_detection_dimensions": dimensions,
                "deviation_components": components,
                "baseline_snapshot": self._baseline_snapshot(baseline),
                "consecutive_deviations": consecutive,
                "consecutive_recoveries": 0,
                "last_checked_at": now.isoformat(),
            }
        )
        if consecutive < 3:
            self.deviation_log.append(
                {
                    "relationship_id": relationship.relationship_id,
                    "detected_at": now.isoformat(),
                    "deviation_score": score,
                    "detected_level": level,
                    "level2_threshold": level2_threshold,
                    "level3_threshold": level3_threshold,
                    "baseline_sensitivity": relationship.preferences.baseline_sensitivity,
                    "baseline_detection_dimensions": dimensions,
                    "deviation_components": components,
                    "baseline_snapshot": self._baseline_snapshot(baseline),
                    "consecutive_deviations": consecutive,
                    "action": "held_for_continuity",
                }
            )
            return None
        self.deviation_log.append(
            {
                "relationship_id": relationship.relationship_id,
                "detected_at": now.isoformat(),
                "deviation_score": score,
                "triggered_level": level,
                "level2_threshold": level2_threshold,
                "level3_threshold": level3_threshold,
                "baseline_sensitivity": relationship.preferences.baseline_sensitivity,
                "baseline_detection_dimensions": dimensions,
                "deviation_components": components,
                "baseline_snapshot": self._baseline_snapshot(baseline),
                "consecutive_deviations": consecutive,
                "action": "triggered",
            }
        )
        if level == 3 and relationship.stage not in {RelationshipStage.INITIATING, RelationshipStage.EXPERIMENTING}:
            return self._safe_care_suggestion("你今天好像不太对劲，想聊聊吗？我有点担心你。")
        return self._safe_care_suggestion("感觉你今天聊起来有点不一样，最近还好吗？")

    def _safe_care_suggestion(self, text: str) -> str:
        forbidden = ["检测到", "系统识别", "偏离度", "状态偏离", "算法", "z-score", "Z-score"]
        if any(item in text for item in forbidden):
            return "感觉你今天聊起来有点不一样，最近还好吗？"
        return text

    def _normalize_baseline_dimensions(self, dimensions: Any) -> list[str]:
        allowed = [
            "sentiment",
            "response_length",
            "emoji",
            "exclamation",
            "question",
            "vocabulary",
            "topic_distribution",
            "interaction_frequency",
            "response_latency",
            "arousal",
            "self_disclosure",
            "personal_importance",
        ]
        if isinstance(dimensions, str):
            raw_items = [item.strip() for item in dimensions.split(",")]
        elif isinstance(dimensions, list):
            raw_items = [str(item).strip() for item in dimensions]
        else:
            raw_items = []
        normalized = [item for item in raw_items if item in allowed]
        return normalized or list(allowed)

    def _baseline_snapshot(self, baseline: EmotionalBaseline) -> dict[str, Any]:
        return {
            "avg_sentiment": baseline.avg_sentiment,
            "std_sentiment": baseline.std_sentiment,
            "avg_response_length": baseline.avg_response_length,
            "std_response_length": baseline.std_response_length,
            "vocabulary_richness": baseline.vocabulary_richness,
            "emoji_usage_rate": baseline.emoji_usage_rate,
            "exclamation_rate": baseline.exclamation_rate,
            "question_rate": baseline.question_rate,
            "topic_distribution": dict(baseline.topic_distribution),
            "interaction_frequency": baseline.interaction_frequency,
            "avg_response_latency": baseline.avg_response_latency,
            "std_response_latency": baseline.std_response_latency,
            "update_window_days": baseline.update_window_days,
            "min_samples": baseline.min_samples,
            "sample_count": baseline.sample_count,
            "baseline_confidence": baseline.baseline_confidence,
            "last_updated": baseline.last_updated.isoformat(),
        }

    def _vocabulary_richness(self, text: str) -> float:
        tokens = tokenize(text)
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def _emoji_count(self, text: str) -> int:
        return len(re.findall(r"[\U0001f300-\U0001faff]", text))

    def _baseline_topic_bucket(self, text: str, signals: TurnSignals) -> str:
        if signals.memory_type in {MemoryType.COMMITMENT, MemoryType.MILESTONE, MemoryType.CONFLICT}:
            return signals.memory_type.value
        lowered = text.lower()
        topic_keywords = {
            "work": ["工作", "项目", "开会", "deadline", "老板", "同事"],
            "family": ["家人", "父母", "妈妈", "爸爸", "孩子"],
            "health": ["失眠", "焦虑", "抑郁", "生病", "医院", "痛苦"],
            "relationship": ["朋友", "恋人", "关系", "我们", "咱们"],
            "interest": ["电影", "书", "游戏", "旅行", "音乐"],
        }
        for topic, keywords in topic_keywords.items():
            if any(keyword in lowered for keyword in keywords):
                return topic
        return "general"

    def _updated_topic_distribution(self, distribution: dict[str, float], topic: str, sample_count: int) -> dict[str, float]:
        previous_count = max(0, sample_count - 1)
        raw_counts = {key: max(0.0, float(value)) * previous_count for key, value in distribution.items()}
        raw_counts[topic] = raw_counts.get(topic, 0.0) + 1.0
        total = sum(raw_counts.values()) or 1.0
        return {key: value / total for key, value in sorted(raw_counts.items()) if value > 0}

    def _topic_distribution_deviation(self, distribution: dict[str, float], topic: str) -> float:
        expected = max(0.0, min(1.0, float(distribution.get(topic, 0.0) or 0.0)))
        return 1.0 - expected

    def _baseline_interaction_frequency(self, relationship: Relationship, now: datetime, window_days: int) -> float:
        window = max(1, window_days)
        start = now - timedelta(days=window)
        count = sum(
            1
            for memory in self.memories.values()
            if memory.relationship_id == relationship.relationship_id and memory.created_at >= start
        )
        return count / window

    def _current_response_latency(self, relationship: Relationship, now: datetime) -> float | None:
        latest = max(
            (
                memory.created_at
                for memory in self.memories.values()
                if memory.relationship_id == relationship.relationship_id and memory.created_at < now
            ),
            default=None,
        )
        if latest is None:
            return None
        return max(0.0, (now - latest).total_seconds())

    def _record_baseline_recovery(self, relationship: Relationship, score: float) -> None:
        state = relationship.baseline_deviation_state
        if not state:
            return
        recoveries = int(state.get("consecutive_recoveries", 0)) + 1
        state["consecutive_recoveries"] = recoveries
        state["last_score"] = score
        state["last_checked_at"] = utcnow().isoformat()
        if recoveries >= 2:
            relationship.baseline_deviation_state = {
                "last_level": None,
                "last_score": score,
                "sensitivity_offset": float(state.get("sensitivity_offset", 0.0) or 0.0),
                "false_positive_streak": int(state.get("false_positive_streak", 0) or 0),
                "accepted_streak": int(state.get("accepted_streak", 0) or 0),
                "consecutive_deviations": 0,
                "consecutive_recoveries": recoveries,
                "last_checked_at": utcnow().isoformat(),
            }

    def _find_emotional_resonance(
        self, relationship_id: str, current: TurnSignals, now: datetime
    ) -> EmotionalMemory | None:
        candidates = [
            emotion
            for emotion in self.emotional_memories.values()
            if (source := self.memories.get(emotion.source_memory_id)) is not None
            if emotion.relationship_id == relationship_id
            and (now - emotion.timestamp) <= timedelta(days=365)
            and abs(emotion.emotional_valence - current.sentiment) <= 0.25
            and emotion.emotional_arousal >= 0.45
            and not self._memory_is_recall_suppressed(source)
            and not source.metadata.get("archived")
        ]
        return max(candidates, key=lambda item: item.emotional_arousal, default=None)

    def _emotional_resonance_evidence(
        self,
        emotional: EmotionalMemory,
        current: TurnSignals,
        now: datetime,
    ) -> dict[str, Any]:
        valence_gap = abs(emotional.emotional_valence - current.sentiment)
        arousal_gap = abs(emotional.emotional_arousal - current.arousal)
        confidence = clamp(
            0.35
            + 0.30 * max(0.0, 1.0 - valence_gap)
            + 0.20 * max(0.0, 1.0 - arousal_gap)
            + 0.15 * emotional.emotion_detection_confidence
        )
        return {
            "source_memory_id": emotional.source_memory_id,
            "emotion_id": emotional.emotion_id,
            "source_timestamp": emotional.timestamp.isoformat(),
            "age_days": max(0, (now - emotional.timestamp).days),
            "current_valence": current.sentiment,
            "source_valence": emotional.emotional_valence,
            "valence_gap": valence_gap,
            "current_arousal": current.arousal,
            "source_arousal": emotional.emotional_arousal,
            "arousal_gap": arousal_gap,
            "emotion_detection_confidence": emotional.emotion_detection_confidence,
            "matching_rule": "same_valence_with_high_arousal_recent_year",
            "hallucination_guard": "source_memory_must_exist_not_suppressed_not_archived",
            "inferred": True,
            "uncertainty_action": "confirm_gently",
            "confidence": confidence,
        }

    def _memory_embedding_features(self, memory: MemoryRecord, signals: TurnSignals | None = None) -> dict[str, Any]:
        signal = signals or detect_turn_signals(memory.content, 0.0, memory.trust_level_at_creation, 0.0)
        semantic = sorted(set(tokenize(memory.content)))
        topic = self._themes(memory.content)
        emotion = {
            "valence": memory.emotional_valence,
            "arousal": memory.emotion_intensity,
            "primary": self._primary_emotion_name(signal),
        }
        context = {
            "tag": memory.context_tag.value,
            "memory_type": memory.memory_type.value,
            "stage": memory.relationship_stage_at_creation.value,
        }
        return {
            "semantic": semantic,
            "topic": topic,
            "emotion": emotion,
            "context": context,
            "metacognition": self._embedding_metacognition(
                semantic=semantic,
                topic=topic,
                emotion=emotion,
                context=context,
                source="local_heuristic_features",
            ),
        }

    def _query_embedding_features(self, query: str, signals: TurnSignals) -> dict[str, Any]:
        return {
            "semantic": sorted(set(tokenize(query))),
            "topic": self._themes(query),
            "emotion": {
                "valence": signals.sentiment,
                "arousal": signals.emotion_intensity,
                "primary": self._primary_emotion_name(signals),
            },
            "context": {
                "tag": signals.context_tag.value,
                "memory_type": signals.memory_type.value,
            },
        }

    def _primary_emotion_name(self, signals: TurnSignals) -> str:
        if signals.context_tag == ContextTag.VULNERABLE_MOMENT:
            return "vulnerability"
        if signals.sentiment > 0:
            return "joy"
        if signals.sentiment < 0:
            return "sadness"
        return "nostalgia"

    def _composite_similarity(self, query: str, current: TurnSignals, memory: MemoryRecord) -> dict[str, Any]:
        query_features = self._query_embedding_features(query, current)
        memory_features = memory.metadata.get("embeddings") or self._memory_embedding_features(memory)
        if "metacognition" not in memory_features:
            memory_features["metacognition"] = self._embedding_metacognition(
                semantic=memory_features.get("semantic", []),
                topic=memory_features.get("topic", []),
                emotion=memory_features.get("emotion", {}),
                context=memory_features.get("context", {}),
                source="legacy_or_unknown",
            )
            memory.metadata["embeddings"] = memory_features
        semantic = self._token_overlap(query_features["semantic"], memory_features.get("semantic", []))
        topic = self._token_overlap(query_features["topic"], memory_features.get("topic", []))
        emotion = self._emotion_feature_similarity(query_features["emotion"], memory_features.get("emotion", {}))
        context = self._context_feature_similarity(query_features["context"], memory_features.get("context", {}))
        weights = self._composite_weights(current, query)
        composite = (
            weights["semantic"] * semantic
            + weights["topic"] * topic
            + weights["emotion"] * emotion
            + weights["context"] * context
        )
        metacognition = memory_features.get("metacognition", {})
        confidence = clamp(float(metacognition.get("confidence", 0.65) or 0.65))
        confidence_multiplier = 0.85 + 0.15 * confidence
        return {
            "semantic": semantic,
            "topic": topic,
            "emotion": emotion,
            "context": context,
            "raw_composite": clamp(composite),
            "composite": clamp(composite * confidence_multiplier),
            "confidence_multiplier": confidence_multiplier,
            "embedding_metacognition": metacognition,
            "weights": weights,
        }

    def _embedding_metacognition(
        self,
        *,
        semantic: list[str],
        topic: list[str],
        emotion: dict[str, Any],
        context: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        semantic_confidence = 0.80 if len(semantic) >= 3 else 0.55 if semantic else 0.30
        topic_confidence = 0.78 if topic and topic != ["日常"] else 0.45
        try:
            arousal = abs(float(emotion.get("arousal", 0.0) or 0.0))
            valence = abs(float(emotion.get("valence", 0.0) or 0.0))
        except (TypeError, ValueError):
            arousal = 0.0
            valence = 0.0
        emotion_confidence = clamp(0.45 + 0.35 * arousal + 0.20 * valence)
        context_confidence = 0.75 if context.get("tag") and context.get("memory_type") else 0.50
        confidence = clamp(
            0.30 * semantic_confidence
            + 0.20 * topic_confidence
            + 0.30 * emotion_confidence
            + 0.20 * context_confidence
        )
        return {
            "source": source,
            "provider": "local_heuristic",
            "trained_embedding": False,
            "training_data_status": "not_trained_no_external_labeled_dataset",
            "confidence": confidence,
            "component_confidence": {
                "semantic": semantic_confidence,
                "topic": topic_confidence,
                "emotion": emotion_confidence,
                "context": context_confidence,
            },
            "limitations": [
                "emotion_and_context_features_are_rule_based",
                "replaceable_with_trained_embedding_provider",
            ],
        }

    def _composite_weights(self, current: TurnSignals, query: str) -> dict[str, float]:
        if current.emotion_intensity >= 0.55:
            return {"semantic": 0.20, "topic": 0.10, "emotion": 0.40, "context": 0.30}
        if any(word in query for word in ["关于", "项目", "工作", "健康", "关系", "承诺"]):
            return {"semantic": 0.30, "topic": 0.40, "emotion": 0.10, "context": 0.20}
        return {"semantic": 0.30, "topic": 0.20, "emotion": 0.30, "context": 0.20}

    def _trust_presentation(
        self, memory: MemoryRecord, relationship: Relationship, final_score: float, weighted_score: float
    ) -> dict[str, Any]:
        harmful = (
            memory.context_tag == ContextTag.CONFLICT
            or memory.emotional_valence <= -0.5
            or memory.memory_type == MemoryType.CONFLICT
        )
        if not relationship.preferences.trust_bias_enabled or not trust_bias_stage_enabled(relationship) or not harmful:
            return {
                "mode": "raw",
                "display_content": memory.content,
                "reason": "not_harmful_or_disabled",
                "original_preserved": True,
                "score_multiplier": 1.0,
            }
        multiplier = final_score / weighted_score if weighted_score else 1.0
        if is_trust_bias_protected(memory):
            return {
                "mode": "raw_critical",
                "display_content": memory.content,
                "reason": "critical_memory_not_softened",
                "original_preserved": True,
                "score_multiplier": multiplier,
            }
        if relationship.trust_level >= 0.8:
            return {
                "mode": "softened",
                "display_content": "这是一段曾经不太舒服的冲突记忆；当前仅低强调呈现，原始记录仍保留用于审计。",
                "reason": "high_trust_harmful_memory_softened",
                "original_preserved": True,
                "score_multiplier": multiplier,
            }
        if relationship.trust_level < 0.4:
            return {
                "mode": "precise_low_trust",
                "display_content": memory.content,
                "reason": "low_trust_harmful_memory_kept_precise",
                "original_preserved": True,
                "score_multiplier": multiplier,
            }
        return {
            "mode": "raw_weighted",
            "display_content": memory.content,
            "reason": "moderate_trust_score_adjusted_only",
            "original_preserved": True,
            "score_multiplier": multiplier,
        }

    def _token_overlap(self, left: list[str], right: list[str]) -> float:
        left_set = set(left)
        right_set = set(right)
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / len(left_set | right_set)

    def _emotion_feature_similarity(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        valence = 1 - abs(float(left.get("valence", 0.0)) - float(right.get("valence", 0.0))) / 2
        arousal = 1 - abs(float(left.get("arousal", 0.0)) - float(right.get("arousal", 0.0)))
        primary = 1.0 if left.get("primary") == right.get("primary") else 0.4
        return clamp(0.40 * valence + 0.40 * arousal + 0.20 * primary)

    def _context_feature_similarity(self, left: dict[str, Any], right: dict[str, Any]) -> float:
        tag_match = 1.0 if left.get("tag") == right.get("tag") else 0.25
        type_match = 1.0 if left.get("memory_type") == right.get("memory_type") else 0.25
        return clamp(0.60 * tag_match + 0.40 * type_match)

    def _retrieval_weights(self, relationship: Relationship, current: TurnSignals, query: str) -> dict[str, Any]:
        if relationship.preferences.mode == Mode.ASSISTANT:
            weights = {
                "emotional_resonance": 0.15,
                "relationship_relevance": 0.15,
                "time_preciousness": 0.15,
                "semantic": 0.55,
            }
            reasons = ["assistant_mode_semantic_priority"]
        else:
            weights = {
                "emotional_resonance": 0.30,
                "relationship_relevance": 0.25,
                "time_preciousness": 0.25,
                "semantic": 0.20,
            }
            reasons = ["friend_mode_relationship_priority"]

        if current.arousal >= 0.70:
            weights["emotional_resonance"] += 0.15
            reasons.append("strong_current_emotion")
        if any(phrase in query for phrase in ["我们之前", "咱们之前", "以前我们", "还记得我们"]):
            weights["relationship_relevance"] += 0.10
            reasons.append("relationship_recall_query")
        if any(phrase in query for phrase in ["是谁", "叫什么", "在哪里", "哪一个", "什么时候", "医生", "地址", "号码"]):
            weights["semantic"] += 0.20
            reasons.append("fact_question")
        if relationship.stage == RelationshipStage.BONDING:
            weights["time_preciousness"] += 0.05
            reasons.append("bonding_time_preciousness")
        if relationship.stage == RelationshipStage.INITIATING:
            weights["semantic"] += 0.10
            reasons.append("initiating_fact_building")

        total = sum(weights.values()) or 1.0
        normalized = {key: value / total for key, value in weights.items()}
        normalized["reasons"] = reasons
        return normalized

    def _inside_joke_suggestion(self, relationship: Relationship, current_text: str, now: datetime) -> dict[str, Any] | None:
        if relationship.stage in {RelationshipStage.INITIATING, RelationshipStage.EXPERIMENTING}:
            return None
        for memory_id in relationship.inside_jokes:
            memory = self.memories.get(memory_id)
            if not memory:
                continue
            if self._memory_is_recall_suppressed(memory):
                continue
            if memory.metadata.get("inside_joke_inactive"):
                continue
            phrase = memory.metadata.get("inside_joke_phrase") or memory.content[:16]
            if current_text and phrase not in current_text and lexical_similarity(current_text, memory.content) < 0.12:
                continue
            replay_log = memory.metadata.setdefault("inside_joke_replay_log", [])
            recent = [_dt(item["at"]) for item in replay_log if isinstance(item, dict) and item.get("at")]
            if recent and (now - max(recent)).days < 30:
                continue
            monthly_count = sum(1 for item in recent if item.year == now.year and item.month == now.month)
            if monthly_count >= 3:
                continue
            replay_log.append({"at": now.isoformat(), "phrase": phrase, "trigger": current_text[:80]})
            memory.mention_count += 1
            return {
                "type": "inside_joke",
                "text": f"inside joke：可以轻轻复现「{phrase}」这个共同梗。",
                "memory_id": memory.memory_id,
                "priority": 40,
            }
        return None

    def _emotional_resonance(self, current: TurnSignals, memory: MemoryRecord) -> float:
        intensity_match = 1 - abs(current.emotion_intensity - memory.emotion_intensity)
        valence_match = 1 - abs(current.sentiment - memory.emotional_valence) / 2
        context_match = 1.0 if current.context_tag == memory.context_tag else 0.3
        return clamp(0.40 * intensity_match + 0.25 * valence_match + 0.20 * context_match + 0.15 * memory.emotion_intensity)

    def _relationship_relevance(self, relationship: Relationship, memory: MemoryRecord) -> float:
        stage_match = 1.0 if memory.relationship_stage_at_creation == relationship.stage else 0.5
        return clamp(0.5 + 0.3 * stage_match + 0.2 * relationship.strength)

    def _presentation_time(self, relationship: Relationship, memory: MemoryRecord, now: datetime) -> dict[str, Any]:
        presentation = temporal_fuzz(memory.created_at, now, memory.emotion_intensity)
        relationship_age_now = max(relationship.relationship_age, (now.date() - relationship.created_at.date()).days)
        age_at_creation = memory.relationship_age_at_creation
        days_since_relationship_moment = max(0, relationship_age_now - age_at_creation)
        mode = relationship.preferences.time_presentation_mode.upper()
        if mode not in {"AUTO", "EXACT", "FUZZY"}:
            mode = "AUTO"
        if mode == "FUZZY":
            presentation["fuzz_level"] = max(0.4, float(presentation.get("fuzz_level", 0.0)))
        anchors = self._presentation_anchors(relationship, memory, presentation)
        preferred_anchor = self._select_presentation_anchor(relationship, anchors)
        exact_phrase = self._exact_time_phrase(memory)
        fuzzy_phrase = preferred_anchor.get("phrase", presentation["phrase"]) if preferred_anchor else presentation["phrase"]
        base_display_phrase = exact_phrase if mode == "EXACT" else fuzzy_phrase
        uncertainty = self._time_uncertainty_expression(
            base_display_phrase,
            fuzz_level=float(presentation.get("fuzz_level", 0.0)),
            mode=mode,
            enabled=relationship.preferences.uncertainty_expression_enabled,
        )
        presentation.update(
            {
                "mode": mode,
                "exact_timestamp": memory.created_at.isoformat(),
                "exact_phrase": exact_phrase,
                "fuzzy_phrase": fuzzy_phrase,
                "base_display_phrase": base_display_phrase,
                "display_phrase": uncertainty["phrase"],
                "phrase": uncertainty["phrase"],
                "uncertainty_expression": uncertainty,
                "relationship_age_at_creation": age_at_creation,
                "relationship_age_now": relationship_age_now,
                "relationship_age_since_days": days_since_relationship_moment,
                "relationship_since_phrase": self._relationship_since_phrase(days_since_relationship_moment),
                "relationship_age_phrase_at_creation": self._relationship_age_phrase(age_at_creation),
                "relationship_age_phrase_now": self._relationship_age_phrase(relationship_age_now),
                "anchors": anchors,
                "preferred_anchor": preferred_anchor,
                "anchor_phrase": fuzzy_phrase,
            }
        )
        return presentation

    def _exact_time_phrase(self, memory: MemoryRecord) -> str:
        source_time = memory.metadata.get("source_time")
        if isinstance(source_time, dict) and source_time.get("phrase"):
            return str(source_time["phrase"])
        return memory.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d")

    def _time_uncertainty_expression(
        self,
        phrase: str,
        *,
        fuzz_level: float,
        mode: str,
        enabled: bool,
    ) -> dict[str, Any]:
        if mode == "EXACT":
            return {"enabled": False, "level": "none", "template": "exact", "phrase": phrase}
        if not enabled:
            return {"enabled": False, "level": self._time_uncertainty_level(fuzz_level), "template": "disabled", "phrase": phrase}
        level = self._time_uncertainty_level(fuzz_level)
        if level == "low":
            rendered = phrase
            template = "direct"
        elif level == "medium":
            rendered = f"大概是{phrase}"
            template = "probably"
        elif level == "high":
            rendered = f"我记得好像是{phrase}"
            template = "impression"
        else:
            rendered = f"{phrase}，具体时间不太确定"
            template = "self_correction"
        return {"enabled": True, "level": level, "template": template, "phrase": rendered}

    def _time_uncertainty_level(self, fuzz_level: float) -> str:
        if fuzz_level < 0.2:
            return "low"
        if fuzz_level < 0.5:
            return "medium"
        if fuzz_level < 0.8:
            return "high"
        return "very_high"

    def _presentation_anchors(
        self,
        relationship: Relationship,
        memory: MemoryRecord,
        presentation: dict[str, float | str],
    ) -> list[dict[str, str]]:
        if float(presentation.get("fuzz_level", 0.0)) < 0.4:
            return []
        anchors: list[dict[str, str]] = []
        if memory.emotion_intensity >= 0.6:
            anchors.append({"type": "emotion", "phrase": self._emotion_anchor_phrase(memory)})
        if memory.memory_type == MemoryType.MILESTONE or memory.context_tag in {ContextTag.MILESTONE, ContextTag.TURNING_POINT}:
            anchors.append({"type": "relationship", "phrase": self._relationship_age_phrase(memory.relationship_age_at_creation)})
        if memory.metadata.get("source_time"):
            anchors.append({"type": "event", "phrase": str(memory.metadata["source_time"].get("phrase", "那段时间"))})
        story = self._story_anchor_phrase(memory)
        if story:
            anchors.append({"type": "shared_story", "phrase": story})
        sensory = self._sensory_anchor_phrase(memory)
        if sensory:
            anchors.append({"type": "sensory", "phrase": sensory})
        return anchors

    def _select_presentation_anchor(
        self,
        relationship: Relationship,
        anchors: list[dict[str, str]],
    ) -> dict[str, str] | None:
        if not anchors:
            return None
        preference = relationship.preferences.anchor_preference.upper()
        orders = {
            "EMOTION_FIRST": ["emotion", "relationship", "event", "shared_story", "sensory"],
            "EVENT_FIRST": ["event", "relationship", "emotion", "shared_story", "sensory"],
            "RELATIONSHIP_FIRST": ["relationship", "emotion", "event", "shared_story", "sensory"],
            "STORY_FIRST": ["shared_story", "emotion", "relationship", "event", "sensory"],
            "SENSORY_FIRST": ["sensory", "emotion", "relationship", "event", "shared_story"],
        }
        order = orders.get(preference, orders["EMOTION_FIRST"])
        by_type = {anchor["type"]: anchor for anchor in anchors}
        for anchor_type in order:
            if anchor_type in by_type:
                return by_type[anchor_type]
        return anchors[0]

    def _emotion_anchor_phrase(self, memory: MemoryRecord) -> str:
        if memory.context_tag == ContextTag.VULNERABLE_MOMENT or memory.emotional_valence < -0.2:
            return "你情绪很重的那次"
        if memory.emotional_valence > 0.2:
            return "你特别开心的那次"
        return "情绪很明显的那次"

    def _story_anchor_phrase(self, memory: MemoryRecord) -> str | None:
        for story in self.story_nodes.values():
            if story.relationship_id != memory.relationship_id:
                continue
            if memory.memory_id in story.core_events or memory.memory_id in story.key_moments:
                return story.consensus_version[:42] or story.title
        return None

    def _sensory_anchor_phrase(self, memory: MemoryRecord) -> str | None:
        text = memory.content
        for phrase in ["下着大雨", "大雨", "特别冷", "很冷", "很热", "有风", "下雪"]:
            if phrase in text:
                return phrase
        metadata = memory.metadata
        for key in ["weather", "temperature", "smell", "sensory_anchor"]:
            if metadata.get(key):
                return str(metadata[key])
        return None

    def _relationship_since_phrase(self, days: int) -> str:
        if days <= 0:
            return "就在这段关系刚发生的时候"
        return f"自那以后已经{self._duration_phrase(days)}"

    def _relationship_age_phrase(self, days: int) -> str:
        if days <= 0:
            return "刚认识的时候"
        return f"认识{self._duration_phrase(days)}的时候"

    def _duration_phrase(self, days: int) -> str:
        if days < 30:
            return f"{days}天"
        if days < 365:
            months = max(1, round(days / 30))
            return f"{months}个月"
        years = days // 365
        months = round((days % 365) / 30)
        if months:
            return f"{years}年{months}个月"
        return f"{years}年"

    def _time_preciousness(self, relationship: Relationship, memory: MemoryRecord, now: datetime) -> float:
        relationship_age = min(1.0, relationship.relationship_age / 365)
        milestone = 1.0 if memory.memory_type == MemoryType.MILESTONE else 0.0
        mention_log = min(1.0, memory.mention_count / 10)
        return clamp(0.4 * relationship_age + 0.3 * milestone + 0.3 * mention_log)

    def _discover_implicit_topics(self, relationship: Relationship, memories: list[MemoryRecord], now: datetime) -> None:
        recent = [
            memory
            for memory in memories
            if (now - memory.created_at).days <= 30
            and not self._memory_is_recall_suppressed(memory)
            and memory.memory_type not in {MemoryType.MILESTONE, MemoryType.IDENTITY}
        ]
        by_theme: dict[str, list[MemoryRecord]] = {}
        for memory in recent:
            for theme in self._themes(memory.content):
                if theme == "日常":
                    continue
                by_theme.setdefault(theme, []).append(memory)
        existing_themes = {topic.get("theme") for topic in relationship.implicit_topics if topic.get("status") == "ACTIVE"}
        for theme, items in sorted(by_theme.items()):
            unique = list({memory.memory_id: memory for memory in items}.values())
            if len(unique) < 3 or theme in existing_themes:
                continue
            confidence = clamp(0.45 + 0.10 * len(unique) + 0.10 * len({memory.created_at.date() for memory in unique}))
            if confidence < 0.70:
                continue
            topic = {
                "topic_id": new_id("topic"),
                "theme": theme,
                "summary": self._implicit_topic_summary(theme, unique),
                "source_memory_ids": [memory.memory_id for memory in unique[:8]],
                "confidence": confidence,
                "status": "ACTIVE",
                "created_at": now.isoformat(),
                "prompt_count": 0,
                "evidence_summary": {
                    "source_memory_ids": [memory.memory_id for memory in unique[:8]],
                    "source_count": len(unique),
                    "distinct_days": len({memory.created_at.date().isoformat() for memory in unique}),
                    "theme": theme,
                    "inferred": True,
                },
                "metacognition": {
                    "inferred": True,
                    "evidence_count": len(unique),
                    "uncertainty_action": "confirm_gently",
                    "hallucination_guard": "live_source_recheck_before_prompt",
                },
            }
            relationship.implicit_topics.append(topic)
            self.deviation_log.append(
                {
                    "type": "implicit_topic_detected",
                    "relationship_id": relationship.relationship_id,
                    "topic_id": topic["topic_id"],
                    "theme": theme,
                    "source_memory_ids": topic["source_memory_ids"],
                    "confidence": confidence,
                    "at": now.isoformat(),
                }
            )

    def _implicit_topic_summary(self, theme: str, memories: list[MemoryRecord]) -> str:
        snippets = [memory.content[:24] for memory in sorted(memories, key=lambda item: item.created_at)[-3:]]
        return f"{theme}相关的反复碎片：" + " / ".join(snippets)

    def _themes(self, text: str) -> list[str]:
        theme_map = {
            "工作": ["工作", "offer", "失业", "项目", "老板", "面试"],
            "健康": ["焦虑", "抑郁", "失眠", "锻炼", "身体"],
            "关系": ["我们", "一起", "陪我", "朋友", "家人"],
            "承诺": ["答应", "约定", "以后", "下次", "决定"],
            "庆祝": ["成功", "庆祝", "开心", "通过", "拿到"],
            "脆弱": ["哭", "崩溃", "害怕", "只有你", "从来没告诉"],
        }
        themes = [theme for theme, words in theme_map.items() if any(word in text for word in words)]
        return themes or ["日常"]

    def _seal_audit_text(self, value: str | None) -> dict[str, Any]:
        text = str(value or "")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return {
            "sealed": True,
            "scheme": "sha256-redacted-v1",
            "digest": digest[:16],
            "length": len(text),
        }

    def _cold_archive_reference(
        self,
        memory: MemoryRecord,
        relationship: Relationship,
        *,
        weight: float,
        ai_value: float,
        now: datetime,
    ) -> dict[str, Any]:
        return {
            "archive_id": f"cold_{memory.memory_id}",
            "scheme": "local-project-cold-archive-v1",
            "relationship_id": relationship.relationship_id,
            "memory_id": memory.memory_id,
            "storage_layer": memory.storage_layer.value,
            "memory_type": memory.memory_type.value,
            "context_tag": memory.context_tag.value,
            "archived_at": now.isoformat(),
            "age_days": max(0, (now - memory.created_at).days),
            "retention_weight": weight,
            "ai_value": ai_value,
            "object_ref": {
                "store": "local_project_state",
                "key": f"cold_archive/{relationship.relationship_id}/{memory.memory_id}",
                "sealed": True,
                "digest": hashlib.sha256(
                    (
                        f"{relationship.relationship_id}|{memory.memory_id}|{memory.content}|"
                        f"{memory.storage_layer.value}|{memory.updated_at.isoformat()}"
                    ).encode("utf-8")
                ).hexdigest()[:16],
                "plaintext_content_retained_in_audit": False,
            },
            "realtime_retrieval": False,
            "restorable": True,
            "restore_method": "restore_archived_memory",
            "reason": "older_than_one_year_and_low_retention_weight",
        }

    def _refresh_l4_replicas(
        self,
        identity: CoreIdentityRecord,
        *,
        now: datetime,
        reason: str,
    ) -> None:
        version = len(identity.change_log)
        identity.replicas = [
            {
                "replica_id": f"{identity.identity_id}:replica:{index}",
                "sealed": True,
                "scheme": "sha256-redacted-replica-v1",
                "digest": hashlib.sha256(
                    (
                        f"{identity.identity_id}|{identity.memory_id}|{index}|{identity.content}|"
                        f"{identity.review_status}|{identity.pending_delete}|{identity.user_confirmed_at}"
                    ).encode("utf-8")
                ).hexdigest()[:16],
                "content_length": len(identity.content),
                "version": version,
                "refreshed_at": now.isoformat(),
                "refresh_reason": reason,
            }
            for index in range(1, 4)
        ]

    def _l4_change_entry(
        self,
        reason: str,
        *,
        now: datetime,
        old_content: str | None = None,
        new_content: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "at": now.isoformat(),
            "reason": reason,
            "plaintext_content_retained": False,
        }
        if old_content is not None:
            entry["old_content_sealed"] = self._seal_audit_text(old_content)
        if new_content is not None:
            entry["new_content_sealed"] = self._seal_audit_text(new_content)
        if extra:
            entry.update(extra)
        return entry

    def _review_l4_change(
        self,
        identity: CoreIdentityRecord,
        memory: MemoryRecord,
        *,
        change_type: str,
        now: datetime,
        previous_content: str | None = None,
    ) -> dict[str, Any]:
        protected_content, input_protection = self._external_ai_memory_content(memory)
        ai_score = clamp(self.ai.evaluate_memory_value(protected_content, self._relationship_context(self.relationships[memory.relationship_id])))
        identity_signal = memory.memory_type == MemoryType.IDENTITY or any(
            phrase in memory.content for phrase in ["我是", "我的名字", "真实姓名", "对我很重要"]
        )
        status = "AI_REVIEWED" if identity_signal and ai_score >= 0.25 else "NEEDS_USER_CONFIRMATION"
        review = {
            "at": now.isoformat(),
            "change_type": change_type,
            "review_status": status,
            "ai_score": ai_score,
            "provider": self._ai_provider_name(),
            "input_protection": input_protection,
            "content_sealed": self._seal_audit_text(memory.content),
            "previous_content_sealed": self._seal_audit_text(previous_content) if previous_content is not None else None,
            "decision": "protect_as_l4" if status == "AI_REVIEWED" else "hold_for_user_confirmation",
        }
        identity.review_status = status
        identity.review_score = ai_score
        identity.review_history.append(review)
        self._refresh_l4_replicas(identity, now=now, reason=f"review_{change_type}")
        memory.metadata["l4_review"] = {
            "identity_id": identity.identity_id,
            "review_status": status,
            "ai_score": ai_score,
            "at": now.isoformat(),
            "input_protection": input_protection,
        }
        self._log_ai_decision(
            memory.relationship_id,
            task="review_l4_core_identity_change",
            input_summary={
                "identity_id": identity.identity_id,
                "memory_id": memory.memory_id,
                "change_type": change_type,
                "content_chars": len(memory.content),
                "input_protection": input_protection,
            },
            output_summary={"review_status": status, "ai_score": ai_score},
            now=now,
        )
        self.deviation_log.append(
            {
                "type": "l4_change_reviewed",
                "relationship_id": memory.relationship_id,
                "identity_id": identity.identity_id,
                "memory_id": memory.memory_id,
                "change_type": change_type,
                "review_status": status,
                "ai_score": ai_score,
                "at": now.isoformat(),
            }
        )
        return review

    def _relationship_deletion_counts(self, relationship_id: str) -> dict[str, int]:
        return {
            "relationship_present": 1 if relationship_id in self.relationships else 0,
            "memories": len([item for item in self.memories.values() if item.relationship_id == relationship_id]),
            "emotional_memories": len([item for item in self.emotional_memories.values() if item.relationship_id == relationship_id]),
            "story_nodes": len([item for item in self.story_nodes.values() if item.relationship_id == relationship_id]),
            "memory_graph_edges": len([item for item in self.memory_graph_edges.values() if item.relationship_id == relationship_id]),
            "core_identity": len([item for item in self.core_identity.values() if item.relationship_id == relationship_id]),
            "commitment_reminders": len([item for item in self.commitment_reminders.values() if item.relationship_id == relationship_id]),
        }

    def _record_deletion_compliance(
        self,
        *,
        relationship_id: str,
        deletion_type: str,
        request_id: str,
        reason: str,
        now: datetime,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        entry = {
            "type": "deletion_compliance_record",
            "relationship_id": relationship_id,
            "deletion_type": deletion_type,
            "request_id": request_id,
            "recorded_at": now.isoformat(),
            "delete_reason_sealed": self._seal_audit_text(reason),
            "summary": self._to_json(summary),
            "content_retained": False,
            "access_scope": "audit_only",
            "protection": {
                "sealed": True,
                "scheme": "sha256-redacted-v1",
                "plaintext_content_retained": False,
            },
        }
        self.deletion_compliance_log.append(entry)
        if len(self.deletion_compliance_log) > 1000:
            self.deletion_compliance_log = self.deletion_compliance_log[-1000:]
        return entry

    def _record_relationship_ending_support(
        self,
        relationship_id: str,
        request_id: str,
        now: datetime,
        *,
        before_counts: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        cooldown_until = now + timedelta(days=7)
        entry = {
            "type": "relationship_ending_support",
            "relationship_id": relationship_id,
            "request_id": request_id,
            "created_at": now.isoformat(),
            "cooldown_until": cooldown_until.isoformat(),
            "relationship_id_sealed": self._seal_audit_text(relationship_id),
            "deleted_scope_summary": before_counts or self._relationship_deletion_counts(relationship_id),
            "message": (
                "HARD_RESET 已确认后，系统不会保留这段关系的记忆内容。"
                "如果删除带来失落感，建议暂停使用一段时间，并联系现实中的朋友、家人或专业支持。"
            ),
            "soft_landing_plan": [
                {
                    "step": "pause",
                    "label": "给自己至少 24 小时不重建这段 AI 关系",
                    "reason": "降低冲动恢复或反复删除带来的情绪波动",
                },
                {
                    "step": "ground",
                    "label": "把注意力转到现实环境：休息、进食、散步或联系可信任的人",
                    "reason": "让删除后的空缺先由现实支持承接",
                },
                {
                    "step": "reach_out",
                    "label": "如果持续难受，联系真人朋友、家人或专业心理咨询资源",
                    "reason": "AI 关系结束后的失落感不应只由另一个 AI 承接",
                },
                {
                    "step": "restart_boundary",
                    "label": "如果 7 天后重新开始，只能作为新的关系开始，不能恢复旧记忆",
                    "reason": "尊重 HARD_RESET 的彻底删除语义",
                },
            ],
            "restart_policy": {
                "may_start_new_relationship_after": cooldown_until.isoformat(),
                "old_memory_recovery_allowed": False,
                "old_memory_recovery_reason": "HARD_RESET deletes relationship memory and derived records; support log stores no content.",
                "new_relationship_requires_transparency_ack": True,
            },
            "check_in_schedule": [
                {"after": "24h", "purpose": "确认是否安全、是否需要现实支持"},
                {"after": "7d", "purpose": "如仍想重新开始，先明确这是新的关系而非恢复旧关系"},
            ],
            "resources": [
                {
                    "type": "crisis",
                    "label": "紧急情况请联系当地急救或危机干预热线",
                },
                {
                    "type": "social_support",
                    "label": "联系可信任的真人朋友、家人或线下支持网络",
                },
                {
                    "type": "professional_support",
                    "label": "如持续痛苦，考虑专业心理咨询资源",
                },
            ],
            "content_retained": False,
            "memory_content_retained": False,
            "relationship_memory_recoverable": False,
        }
        self.relationship_ending_support_log.append(entry)
        if len(self.relationship_ending_support_log) > 200:
            self.relationship_ending_support_log = self.relationship_ending_support_log[-200:]
        self.deviation_log.append(
            {
                "type": "relationship_ending_support_created",
                "relationship_id": relationship_id,
                "request_id": request_id,
                "at": now.isoformat(),
            }
        )
        return entry

    def _story_title(self, memory: MemoryRecord, themes: list[str]) -> str:
        if memory.memory_type == MemoryType.MILESTONE:
            return f"里程碑：{memory.content[:18]}"
        if memory.memory_type == MemoryType.COMMITMENT:
            return f"未完结约定：{memory.content[:18]}"
        return f"{themes[0]}故事：{memory.content[:18]}"

    def _find_story(self, relationship_id: str, themes: list[str], content: str) -> SharedStoryNode | None:
        candidates = [story for story in self.story_nodes.values() if story.relationship_id == relationship_id]
        for story in candidates:
            if set(story.recurring_themes).intersection(themes):
                return story
            if lexical_similarity(story.consensus_version, content) >= 0.25:
                return story
        return None

    def _summarize_story(self, story: SharedStoryNode) -> str:
        protected_items = [
            self._external_ai_memory_content(self.memories[mid])
            for mid in story.core_events[-3:]
            if mid in self.memories
        ]
        contents = [item[0] for item in protected_items]
        protections = [item[1] for item in protected_items]
        summary = self.ai.summarize_story(contents, {"themes": story.recurring_themes, "level": story.narrative_level.value})
        self._log_ai_decision(
            story.relationship_id,
            task="summarize_story",
            input_summary={
                "story_id": story.story_id,
                "event_count": len(contents),
                "themes": story.recurring_themes,
                "input_protection": {
                    "redacted": any(item.get("redacted") for item in protections),
                    "items": protections,
                },
            },
            output_summary={"summary_chars": len(summary), "summary_preview": summary[:80]},
        )
        return summary

    def _rebuild_story_after_deleted_source(
        self,
        story: SharedStoryNode,
        relationship: Relationship,
        now: datetime,
    ) -> bool:
        valid_core_events = [memory_id for memory_id in story.core_events if memory_id in self.memories]
        story.core_events = valid_core_events
        story.key_moments = [memory_id for memory_id in story.key_moments if memory_id in self.memories]
        story.child_inside_jokes = [memory_id for memory_id in story.child_inside_jokes if memory_id in self.memories]
        if len(valid_core_events) < 3:
            del self.story_nodes[story.story_id]
            self.deviation_log.append(
                {
                    "type": "story_deleted_after_source_rebuild",
                    "relationship_id": relationship.relationship_id,
                    "story_id": story.story_id,
                    "remaining_sources": len(valid_core_events),
                    "at": now.isoformat(),
                }
            )
            return False

        previous_level = story.narrative_level
        previous_consensus = story.consensus_version
        story.consensus_version = self._summarize_story(story)
        story.narrative_level = self._narrative_level(story, relationship)
        story.consistency_score = max(story.consistency_score, 0.70)
        provenance = story.consensus_provenance if isinstance(story.consensus_provenance, dict) else {}
        rebuild_history = provenance.setdefault("rebuild_history", [])
        rebuild_history.append(
            {
                "at": now.isoformat(),
                "reason": "deleted_source_rebuild",
                "remaining_memory_ids": list(valid_core_events),
                "deleted_source_count": provenance.get("deleted_source_count", 0),
            }
        )
        provenance.update(
            {
                "source": "schema_rebuild",
                "status": "SIMULATED_FROM_USER_ACCOUNT",
                "single_user_account": True,
                "requires_user_confirmation": True,
                "has_deleted_source": False,
                "requires_schema_rebuild": False,
                "rebuilt_excluding_deleted_sources": True,
                "memory_ids": list(valid_core_events),
                "at": now.isoformat(),
            }
        )
        story.consensus_provenance = provenance
        self._record_story_narrative_version(
            story,
            previous_level=previous_level,
            previous_consensus=previous_consensus,
            reason="deleted_source_rebuild",
            now=now,
        )
        self.deviation_log.append(
            {
                "type": "story_rebuilt_after_deleted_source",
                "relationship_id": relationship.relationship_id,
                "story_id": story.story_id,
                "remaining_sources": len(valid_core_events),
                "at": now.isoformat(),
            }
        )
        return True

    def _narrative_level(self, story: SharedStoryNode, relationship: Relationship) -> NarrativeLevel:
        if any(
            (memory := self.memories.get(memory_id)) is not None and memory.memory_type == MemoryType.MILESTONE
            for memory_id in story.core_events
        ):
            return NarrativeLevel.STORYLINE
        if len(story.core_events) >= 8 and story.retell_count >= 10:
            return NarrativeLevel.STORYLINE
        if len(story.core_events) >= 5 and relationship.stage in {
            RelationshipStage.INTEGRATING,
            RelationshipStage.BONDING,
        }:
            return NarrativeLevel.CHAPTER
        if len(story.core_events) >= 3:
            return NarrativeLevel.EPISODE
        return NarrativeLevel.FRAGMENT

    def _framing(self, valence: float) -> str:
        if valence >= 0.25:
            return "POSITIVE"
        if valence <= -0.25:
            return "NEGATIVE"
        if valence != 0:
            return "MIXED"
        return "NEUTRAL"

    def _relationship_themes(self, relationship: Relationship) -> list[str]:
        themes: set[str] = set()
        for story in self.story_nodes.values():
            if story.relationship_id == relationship.relationship_id:
                themes.update(story.recurring_themes)
        return sorted(themes)

    def _detect_trajectory_patterns(self, trajectory: EmotionalTrajectory) -> None:
        if len(trajectory.time_series) < 3:
            return
        recent = trajectory.time_series[-3:]
        valences = [item.avg_valence for item in recent]
        pattern = None
        if valences[0] < valences[1] < valences[2]:
            pattern = "GROWTH_TRAJECTORY"
        elif valences[0] > valences[1] > valences[2]:
            pattern = "DECLINING_TRUST"
        elif max(valences) - min(valences) < 0.1:
            pattern = "EMOTIONAL_PLATEAU"
        if pattern and not any(item.get("pattern_type") == pattern for item in trajectory.detected_patterns[-3:]):
            trajectory.detected_patterns.append(
                {"pattern_type": pattern, "confidence": 0.65, "detected_at": utcnow().isoformat()}
            )

    def _relationship_context(self, relationship: Relationship) -> dict[str, Any]:
        return {
            "relationship_id": relationship.relationship_id,
            "stage": relationship.stage.value,
            "strength": relationship.strength,
            "trust_level": relationship.trust_level,
            "intimacy_level": relationship.intimacy_level,
            "relationship_age": relationship.relationship_age,
            "retention_multiplier": relationship.retention_multiplier,
            "themes": relationship.relationship_narrative.core_themes,
            "mode": relationship.preferences.mode.value,
        }

    def _apply_criticality_protection(
        self,
        relationship: Relationship,
        memory: MemoryRecord,
        *,
        now: datetime,
    ) -> None:
        existing = str(memory.metadata.get("criticality") or memory.metadata.get("severity") or "").upper()
        reasons = self._criticality_reasons(memory.content)
        if existing in {"CRITICAL", "SAFETY", "MEDICAL", "MAJOR_COMMITMENT"}:
            memory.metadata["trust_bias_protected"] = True
            memory.metadata.setdefault("criticality_source", "provided_metadata")
            return
        if not reasons:
            return
        if any(reason == "safety" for reason in reasons):
            criticality = "SAFETY"
        elif any(reason == "medical" for reason in reasons):
            criticality = "MEDICAL"
        elif any(reason == "major_commitment" for reason in reasons):
            criticality = "MAJOR_COMMITMENT"
        else:
            criticality = "CRITICAL"
        memory.metadata["criticality"] = criticality
        memory.metadata["criticality_reasons"] = reasons
        memory.metadata["criticality_source"] = "automatic_signal"
        memory.metadata["trust_bias_protected"] = True
        previous_importance = memory.importance
        memory.importance = max(memory.importance, 0.95)
        memory.base_weight = max(memory.base_weight, 0.95)
        memory.storage_layer = self._storage_layer_for(
            memory_type=memory.memory_type,
            context_tag=memory.context_tag,
            score=memory.importance,
            relationship=relationship,
        )
        memory.metadata.setdefault("criticality_audit", []).append(
            {
                "at": now.isoformat(),
                "criticality": criticality,
                "reasons": reasons,
                "previous_importance": previous_importance,
                "new_importance": memory.importance,
                "trust_bias_protected": True,
            }
        )
        self.deviation_log.append(
            {
                "type": "critical_memory_protected",
                "relationship_id": memory.relationship_id,
                "memory_id": memory.memory_id,
                "criticality": criticality,
                "reasons": reasons,
                "at": now.isoformat(),
            }
        )

    def _review_cold_information(
        self,
        relationship: Relationship,
        memory: MemoryRecord,
        ai_value: float,
        now: datetime,
    ) -> None:
        cold_class = self._cold_information_class(memory)
        if not cold_class:
            return
        reinforcement = memory.metadata.get("retrieval_reinforcement")
        if isinstance(reinforcement, dict) and int(reinforcement.get("count", 0) or 0) > 0:
            return
        review = memory.metadata.setdefault("cold_information_review", {})
        last_reviewed = review.get("last_reviewed_at")
        if last_reviewed:
            try:
                if (now - datetime.fromisoformat(str(last_reviewed))).days < 7:
                    return
            except ValueError:
                pass

        explicit_critical = str(memory.metadata.get("criticality") or "").upper() in {
            "CRITICAL",
            "SAFETY",
            "MEDICAL",
            "MAJOR_COMMITMENT",
        }
        confirmed = explicit_critical or ai_value >= 0.50
        review["last_reviewed_at"] = now.isoformat()
        review["review_value"] = ai_value
        review["review_kind"] = cold_class
        review["special_access_count"] = int(review.get("special_access_count", 0) or 0) + 1
        review["separate_from_user_access"] = True
        if confirmed:
            status = "CONFIRMED_CRITICAL" if explicit_critical else "CONFIRMED_HIGH"
            review["status"] = status
            review["protected_until"] = (now + timedelta(days=7)).isoformat()
            review["protection_reason"] = "criticality_signal" if explicit_critical else "ai_value_review"
            memory.base_weight = max(memory.base_weight, 0.95 if explicit_critical else 0.80)
            memory.importance = max(memory.importance, 0.95 if explicit_critical else 0.80)
            if explicit_critical:
                memory.decay_curve = DecayCurve.PERMANENT
            event_status = status
        else:
            review["status"] = "DOWNGRADED_TO_NORMAL"
            review.pop("protected_until", None)
            review["protection_reason"] = "ai_value_below_threshold"
            memory.metadata["criticality"] = "NORMAL"
            event_status = "DOWNGRADED_TO_NORMAL"
        self.deviation_log.append(
            {
                "type": "cold_information_reviewed",
                "relationship_id": relationship.relationship_id,
                "memory_id": memory.memory_id,
                "status": event_status,
                "review_kind": cold_class,
                "ai_value": ai_value,
                "special_access_count": review["special_access_count"],
                "at": now.isoformat(),
            }
        )

    def _cold_information_class(self, memory: MemoryRecord) -> str | None:
        criticality = str(memory.metadata.get("criticality") or memory.metadata.get("severity") or "").upper()
        if criticality in {"CRITICAL", "SAFETY", "MEDICAL", "MAJOR_COMMITMENT"}:
            return criticality
        if memory.importance >= 0.80 and not memory.metadata.get("archived"):
            return "HIGH"
        return None

    def _criticality_reasons(self, text: str) -> list[str]:
        lowered = text.lower()
        reasons: list[str] = []
        safety_terms = [
            "自杀",
            "轻生",
            "伤害自己",
            "伤害别人",
            "报警",
            "急救",
            "危险",
            "安全",
            "安全风险",
            "求救",
            "suicide",
            "self harm",
            "emergency",
            "danger",
        ]
        medical_terms = [
            "医生",
            "医院",
            "诊所",
            "药",
            "用药",
            "过敏",
            "病",
            "手术",
            "急诊",
            "medical",
            "doctor",
            "allergy",
            "medicine",
        ]
        commitment_terms = ["重大承诺", "必须", "deadline", "截止", "合同", "还款", "签证", "考试", "面试"]
        if any(term in lowered for term in safety_terms):
            reasons.append("safety")
        if any(term in lowered for term in medical_terms):
            reasons.append("medical")
        if (
            any(term in lowered for term in commitment_terms)
            and any(term in lowered for term in ["承诺", "答应", "约定", "提醒", "别忘", "必须", "deadline", "截止"])
        ):
            reasons.append("major_commitment")
        return reasons

    def _external_ai_memory_content(self, memory: MemoryRecord) -> tuple[str, dict[str, Any]]:
        protection = {
            "redacted": False,
            "reason": "not_required",
            "storage_layer": memory.storage_layer.value,
            "memory_type": memory.memory_type.value,
            "context_tag": memory.context_tag.value,
        }
        if not self._external_ai_may_receive_content():
            protection["reason"] = "local_ai_only"
            return memory.content, protection
        if not self._memory_requires_external_redaction(memory):
            protection["reason"] = "not_protected_class"
            return memory.content, protection
        themes = self._themes(memory.content)
        sealed = self._seal_audit_text(memory.content)
        protection.update(
            {
                "redacted": True,
                "reason": "protected_relationship_memory",
                "themes": themes,
                "sealed": sealed,
            }
        )
        safe_content = (
            f"[redacted:{memory.storage_layer.value}:{memory.memory_type.value}:"
            f"{memory.context_tag.value}:themes={','.join(themes)}:chars={len(memory.content)}]"
        )
        return safe_content, protection

    def _external_ai_may_receive_content(self) -> bool:
        config = describe_memory_ai(self.ai)
        kind = config.get("participation_kind")
        if kind in {"external_model", "external_http_worker", "external_with_local_fallback"}:
            return True
        primary = config.get("primary") if isinstance(config, dict) else None
        return isinstance(primary, dict) and primary.get("participation_kind") in {"external_model", "external_http_worker"}

    def _memory_requires_external_redaction(self, memory: MemoryRecord) -> bool:
        if memory.storage_layer in {MemoryLayer.L4_CORE_IDENTITY, MemoryLayer.L5_RELATIONSHIP_HISTORY}:
            return True
        if memory.memory_type in {MemoryType.IDENTITY, MemoryType.MILESTONE}:
            return True
        if memory.context_tag in {ContextTag.VULNERABLE_MOMENT, ContextTag.TURNING_POINT}:
            return True
        if memory.emotion_intensity >= 0.70 or memory.metadata.get("core_identity_ref"):
            return True
        return False

    def _sanitize_ai_analysis(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {
                "importance": 0.0,
                "memory_type": None,
                "context_tag": None,
                "reason": "ai_output_sanitized: non_dict",
                "ai_sanitization": {"changed": True, "issues": ["non_dict"]},
            }

        sanitized = dict(raw)
        issues: list[str] = []
        try:
            sanitized["importance"] = clamp(float(raw.get("importance", 0.0) or 0.0))
        except (TypeError, ValueError):
            sanitized["importance"] = 0.0
            issues.append("invalid_importance")

        memory_type = raw.get("memory_type")
        if memory_type and self._memory_type_from_ai({"memory_type": memory_type}) is None:
            sanitized["memory_type"] = None
            issues.append("invalid_memory_type")

        context_tag = raw.get("context_tag")
        if context_tag and self._context_tag_from_ai({"context_tag": context_tag}) is None:
            sanitized["context_tag"] = None
            issues.append("invalid_context_tag")

        if not isinstance(sanitized.get("tags", []), list):
            sanitized["tags"] = []
            issues.append("invalid_tags")

        if issues:
            reason = str(sanitized.get("reason", ""))
            sanitized["reason"] = f"{reason}; ai_output_sanitized" if reason else "ai_output_sanitized"
            sanitized["ai_sanitization"] = {"changed": True, "issues": issues}
        else:
            sanitized.setdefault("ai_sanitization", {"changed": False, "issues": []})
        return sanitized

    def _memory_type_from_ai(self, analysis: dict[str, Any]) -> MemoryType | None:
        value = analysis.get("memory_type")
        if not value:
            return None
        try:
            return MemoryType(value)
        except ValueError:
            return None

    def _context_tag_from_ai(self, analysis: dict[str, Any]) -> ContextTag | None:
        value = analysis.get("context_tag")
        if not value:
            return None
        try:
            return ContextTag(value)
        except ValueError:
            return None

    def _to_json(self, value: Any) -> Any:
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "value"):
            return value.value
        if hasattr(value, "__dataclass_fields__"):
            return {key: self._to_json(val) for key, val in asdict(value).items()}
        if isinstance(value, dict):
            return {key: self._to_json(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._to_json(item) for item in value]
        return value

    def _anonymize_export(self, payload: dict[str, Any]) -> dict[str, Any]:
        id_map: dict[str, str] = {}
        sensitive_text_keys = {
            "content",
            "title",
            "consensus",
            "consensus_version",
            "previous_consensus",
            "new_consensus",
            "corrected_consensus",
            "conflicting_content",
            "source_text",
            "display_content",
            "summary_preview",
            "query",
            "last_query",
            "old",
            "new",
            "message",
            "reason",
            "reason_text",
            "delete_reason",
            "recommendation",
            "origin_story",
            "statement",
            "subject",
            "note",
            "acknowledgement_note",
            "feedback_note",
            "boundary_text_preview",
            "semantic",
            "text",
            "phrase",
        }
        sensitive_id_keys = {
            "user_id",
            "ai_id",
            "relationship_id",
            "memory_id",
            "source_memory_id",
            "target_memory_id",
            "emotion_id",
            "story_id",
            "identity_id",
            "request_id",
            "reminder_id",
            "edge_id",
            "active_id",
            "alert_id",
            "summary_id",
            "promoted_memory_id",
            "winner_memory_ids",
            "migration_id",
            "participants",
        }

        def anon_id(value: Any) -> Any:
            if value is None:
                return None
            text = str(value)
            if text not in id_map:
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
                id_map[text] = f"anon_{digest}"
            return id_map[text]

        def anon_text(value: Any) -> Any:
            if value in (None, ""):
                return value
            digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
            return f"[redacted:{digest}]"

        def walk(value: Any, key: str | None = None) -> Any:
            if isinstance(value, dict):
                result: dict[str, Any] = {}
                for raw_key, raw_value in value.items():
                    new_key = anon_id(raw_key) if raw_key in id_map or _looks_like_identifier(str(raw_key)) else raw_key
                    result[new_key] = walk(raw_value, str(raw_key))
                return result
            if isinstance(value, list):
                return [walk(item, key) for item in value]
            if key in sensitive_text_keys:
                return anon_text(value)
            if key in sensitive_id_keys or (isinstance(value, str) and _looks_like_identifier(value)):
                return anon_id(value)
            return value

        anonymized = walk(deepcopy(payload))
        anonymized["anonymization"] = {
            "mode": "ANONYMIZED",
            "redacted_text_fields": sorted(sensitive_text_keys),
            "hashed_identifier_count": len(id_map),
        }
        return anonymized


def _looks_like_identifier(value: str) -> bool:
    prefixes = (
        "mem_",
        "emo_",
        "story_",
        "l4_",
        "l4del_",
        "reset_",
        "health_",
        "guardian_",
        "migration_",
        "reminder_",
        "active_",
        "memdel_",
    )
    if any(value.startswith(prefix) and len(value) >= len(prefix) + 8 for prefix in prefixes):
        suffix = value.split("_", 1)[1]
        return all(char in "0123456789abcdef" for char in suffix.lower())
    return ":" in value and "T" not in value and " " not in value


def _legacy_storage_layer(raw: dict[str, Any]) -> MemoryLayer:
    memory_type = MemoryType(raw.get("memory_type", MemoryType.CONTEXT_DETAIL.value))
    context_tag = ContextTag(raw.get("context_tag", ContextTag.GENERAL.value))
    importance = float(raw.get("importance", 0.0) or 0.0)
    if memory_type == MemoryType.IDENTITY:
        return MemoryLayer.L4_CORE_IDENTITY
    if memory_type in {MemoryType.MILESTONE, MemoryType.COMMITMENT, MemoryType.INSIDE_JOKE}:
        return MemoryLayer.L5_RELATIONSHIP_HISTORY
    if context_tag in {ContextTag.MILESTONE, ContextTag.TURNING_POINT, ContextTag.UNRESOLVED_THREAD, ContextTag.INSIDE_JOKE}:
        return MemoryLayer.L5_RELATIONSHIP_HISTORY
    if memory_type in {MemoryType.SHARED_EPISODE, MemoryType.EMOTIONAL_MOMENT, MemoryType.EMOTIONAL_PREFERENCE, MemoryType.CONFLICT}:
        return MemoryLayer.L3_RELATIONAL
    if importance >= 0.50:
        return MemoryLayer.L2_EPISODIC
    return MemoryLayer.L1_IMMEDIATE


def _dt(value: str | None) -> datetime:
    if not value:
        return utcnow()
    return datetime.fromisoformat(value)


def _relationship_stage_from_json(value: Any) -> RelationshipStage:
    try:
        return RelationshipStage(str(value))
    except ValueError:
        return RelationshipStage.INITIATING


def _clamped_float_from_json(value: Any, default: float = 0.0) -> float:
    try:
        return clamp(float(value))
    except (TypeError, ValueError):
        return default


def _nonnegative_int_from_json(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _positive_int_from_json(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _retention_multiplier_from_json(value: Any, stage: RelationshipStage) -> float:
    default = RETENTION_MULTIPLIER_BY_STAGE.get(stage, 1.0)
    try:
        return max(0.4, min(2.5, float(value)))
    except (TypeError, ValueError):
        return default


def _relationship_from_json(raw: dict[str, Any]) -> Relationship:
    relationship = Relationship(
        user_id=raw["user_id"],
        ai_id=raw["ai_id"],
        relationship_id=raw["relationship_id"],
        created_at=_dt(raw.get("created_at")),
    )
    relationship.strength = _clamped_float_from_json(raw.get("strength"), 0.0)
    relationship.stage = _relationship_stage_from_json(raw.get("stage", RelationshipStage.INITIATING.value))
    relationship.trust_level = _clamped_float_from_json(raw.get("trust_level"), 0.5)
    relationship.intimacy_level = _clamped_float_from_json(raw.get("intimacy_level"), 0.0)
    relationship.retention_multiplier = _retention_multiplier_from_json(
        raw.get("retention_multiplier"),
        relationship.stage,
    )
    relationship.schema_version = _positive_int_from_json(raw.get("schema_version"), 1)
    relationship.relationship_age = _nonnegative_int_from_json(raw.get("relationship_age"), 0)
    relationship.last_interaction = _dt(raw["last_interaction"]) if raw.get("last_interaction") else None
    relationship.last_updated = _dt(raw.get("last_updated"))
    relationship.interaction_count = _nonnegative_int_from_json(raw.get("interaction_count"), 0)
    relationship.active_days = set(raw.get("active_days", []))
    relationship.shared_episodes = raw.get("shared_episodes", [])
    relationship.inside_jokes = raw.get("inside_jokes", [])
    relationship.inside_joke_candidates = raw.get("inside_joke_candidates", {})
    relationship.milestones = raw.get("milestones", [])
    relationship.unresolved_threads = raw.get("unresolved_threads", [])
    relationship.core_identity = raw.get("core_identity", [])
    relationship.active_behavior_log = raw.get("active_behavior_log", [])
    relationship.mode_history = raw.get("mode_history", [])
    relationship.stage_history = raw.get("stage_history", [])
    relationship.user_age = raw.get("user_age")
    relationship.daily_interaction_minutes = raw.get("daily_interaction_minutes", {})
    relationship.transparency_acknowledged_at = (
        _dt(raw["transparency_acknowledged_at"]) if raw.get("transparency_acknowledged_at") else None
    )
    relationship.baseline_deviation_state = raw.get("baseline_deviation_state", {})
    relationship.active_feedback_state = raw.get("active_feedback_state", {})
    relationship.retention_calibration_state = raw.get("retention_calibration_state", {})
    relationship.maintenance_signals = raw.get("maintenance_signals", {})
    relationship.trust_decay_state = raw.get("trust_decay_state", {})
    relationship.implicit_topics = raw.get("implicit_topics", [])
    relationship.preferences = _preferences_from_json(raw.get("preferences", {}))
    relationship.decay_curve_type = _decay_curve_from_json(
        raw.get("decay_curve_type"),
        relationship.preferences.reverse_decay_enabled,
    )
    relationship.preferences.reverse_decay_enabled = relationship.decay_curve_type != DecayCurve.STANDARD_POWER_LAW
    relationship.emotional_baseline = _baseline_from_json(raw.get("emotional_baseline", {}))
    relationship.interaction_patterns = _patterns_from_json(raw.get("interaction_patterns", {}))
    relationship.relationship_narrative = _narrative_from_json(raw.get("relationship_narrative", {}))
    return relationship


def _memory_from_json(raw: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        memory_id=raw["memory_id"],
        relationship_id=raw["relationship_id"],
        content=raw["content"],
        memory_type=MemoryType(raw["memory_type"]),
        context_tag=ContextTag(raw["context_tag"]),
        created_at=_dt(raw["created_at"]),
        updated_at=_dt(raw["updated_at"]),
        base_weight=raw["base_weight"],
        importance=raw["importance"],
        emotion_intensity=raw.get("emotion_intensity", 0.0),
        emotional_valence=raw.get("emotional_valence", 0.0),
        mention_count=raw.get("mention_count", 1),
        decay_curve=DecayCurve(raw.get("decay_curve", DecayCurve.STANDARD_POWER_LAW.value)),
        relationship_stage_at_creation=RelationshipStage(raw.get("relationship_stage_at_creation", RelationshipStage.INITIATING.value)),
        relationship_age_at_creation=raw.get("relationship_age_at_creation", 0),
        trust_level_at_creation=raw.get("trust_level_at_creation", 0.5),
        storage_layer=MemoryLayer(raw.get("storage_layer", _legacy_storage_layer(raw).value)),
        tags=set(raw.get("tags", [])),
        metadata=raw.get("metadata", {}),
    )


def _memory_graph_edge_from_json(raw: dict[str, Any]) -> MemoryGraphEdge:
    return MemoryGraphEdge(
        edge_id=raw["edge_id"],
        relationship_id=raw["relationship_id"],
        source_memory_id=raw["source_memory_id"],
        target_memory_id=raw["target_memory_id"],
        relation_type=raw["relation_type"],
        weight=raw.get("weight", 0.0),
        created_at=_dt(raw["created_at"]),
        evidence=raw.get("evidence", {}),
    )


def _emotional_from_json(raw: dict[str, Any]) -> EmotionalMemory:
    return EmotionalMemory(
        emotion_id=raw["emotion_id"],
        relationship_id=raw["relationship_id"],
        source_memory_id=raw["source_memory_id"],
        content=raw["content"],
        timestamp=_dt(raw["timestamp"]),
        relationship_age_at_creation=raw["relationship_age_at_creation"],
        emotions=[EmotionLabel(item["name"], item["intensity"]) for item in raw.get("emotions", [])],
        primary_emotion=raw["primary_emotion"],
        emotional_valence=raw["emotional_valence"],
        emotional_arousal=raw["emotional_arousal"],
        personal_importance=raw["personal_importance"],
        self_disclosure_depth=raw["self_disclosure_depth"],
        context_tag=ContextTag(raw["context_tag"]),
        relationship_stage_at_creation=RelationshipStage(raw["relationship_stage_at_creation"]),
        trust_level_at_creation=raw["trust_level_at_creation"],
        embeddings=raw.get("embeddings", {}),
    )


def _story_from_json(raw: dict[str, Any]) -> SharedStoryNode:
    return SharedStoryNode(
        story_id=raw["story_id"],
        relationship_id=raw["relationship_id"],
        title=raw["title"],
        narrative_level=NarrativeLevel(raw["narrative_level"]),
        core_events=raw.get("core_events", []),
        key_moments=raw.get("key_moments", []),
        recurring_themes=raw.get("recurring_themes", []),
        participants=raw.get("participants", []),
        story_arc_start=_dt(raw["story_arc_start"]),
        story_arc_end=_dt(raw["story_arc_end"]) if raw.get("story_arc_end") else None,
        retell_count=raw.get("retell_count", 1),
        last_retold=_dt(raw.get("last_retold")),
        consensus_version=raw.get("consensus_version", ""),
        conflict_versions=raw.get("conflict_versions", []),
        correction_versions=raw.get("correction_versions", []),
        narrative_versions=raw.get("narrative_versions", []),
        child_inside_jokes=raw.get("child_inside_jokes", []),
        consistency_score=raw.get("consistency_score", 0.7),
        emotional_arc=raw.get("emotional_arc", []),
        user_framing=raw.get("user_framing", "NEUTRAL"),
        ai_framing_confidence=raw.get("ai_framing_confidence", 0.7),
        consensus_status=raw.get("consensus_status", "SIMULATED_FROM_USER_ACCOUNT"),
        consensus_provenance=raw.get(
            "consensus_provenance",
            {
                "source": "legacy_or_unknown",
                "status": raw.get("consensus_status", "SIMULATED_FROM_USER_ACCOUNT"),
                "single_user_account": True,
                "requires_user_confirmation": True,
                "memory_ids": raw.get("core_events", []),
            },
        ),
        consensus_confirmed_at=_dt(raw["consensus_confirmed_at"]) if raw.get("consensus_confirmed_at") else None,
    )


def _trajectory_from_json(raw: dict[str, Any]) -> EmotionalTrajectory:
    return EmotionalTrajectory(
        relationship_id=raw["relationship_id"],
        time_series=[
            EmotionalTrajectoryWindow(
                window_start=_dt(item["window_start"]),
                window_end=_dt(item["window_end"]),
                avg_valence=item.get("avg_valence", 0.0),
                avg_arousal=item.get("avg_arousal", 0.0),
                dominant_emotions=item.get("dominant_emotions", []),
                emotional_diversity=item.get("emotional_diversity", 0.0),
                notable_events=item.get("notable_events", []),
            )
            for item in raw.get("time_series", [])
        ],
        detected_patterns=raw.get("detected_patterns", []),
    )


def _bool_from_json(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on", "开"}
    return bool(value)


def _max_active_from_json(value: Any) -> int:
    return max(0, min(5, int(value)))


def _time_presentation_from_json(value: Any) -> str | None:
    normalized = str(value).upper()
    return normalized if normalized in {"AUTO", "EXACT", "FUZZY"} else None


def _anchor_preference_from_json(value: Any) -> str | None:
    normalized = str(value).upper()
    allowed = {"EMOTION_FIRST", "EVENT_FIRST", "RELATIONSHIP_FIRST", "STORY_FIRST", "SENSORY_FIRST"}
    return normalized if normalized in allowed else None


def _baseline_sensitivity_from_json(value: Any) -> str | None:
    normalized = str(value).upper()
    return normalized if normalized in {"LOW", "MEDIUM", "HIGH"} else None


def _baseline_dimensions_from_json(value: Any) -> list[str]:
    allowed = {
        "sentiment",
        "response_length",
        "emoji",
        "exclamation",
        "question",
        "vocabulary",
        "topic_distribution",
        "interaction_frequency",
        "response_latency",
        "arousal",
        "self_disclosure",
        "personal_importance",
    }
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value).split(",")
    normalized = [str(item).strip() for item in raw_items if str(item).strip() in allowed]
    return normalized or UserPreferences().baseline_detection_dimensions


def _data_export_permission_from_json(value: Any) -> str | None:
    normalized = str(value).upper()
    if normalized == "ANONYMOUS":
        normalized = "ANONYMIZED"
    return normalized if normalized in {"FULL", "ANONYMIZED"} else None


def _custom_profile_from_json(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, Any] = {}
    bool_keys = {
        "active_recall_enabled",
        "trust_bias_enabled",
        "reverse_decay_enabled",
        "emotional_layer_enabled",
        "baseline_detection_enabled",
        "relationship_nature_disclosure_enabled",
        "level3_enabled",
        "uncertainty_expression_enabled",
    }
    float_keys = {"nostalgia_tendency", "surprise_tendency", "depth_tendency"}
    for key, value in raw.items():
        try:
            if key in bool_keys:
                normalized[key] = _bool_from_json(value)
            elif key == "max_active_per_session":
                normalized[key] = _max_active_from_json(value)
            elif key in float_keys:
                normalized[key] = clamp(float(value))
            elif key == "time_presentation_mode":
                parsed = _time_presentation_from_json(value)
                if parsed is not None:
                    normalized[key] = parsed
            elif key == "anchor_preference":
                parsed = _anchor_preference_from_json(value)
                if parsed is not None:
                    normalized[key] = parsed
            elif key == "baseline_sensitivity":
                parsed = _baseline_sensitivity_from_json(value)
                if parsed is not None:
                    normalized[key] = parsed
            elif key == "baseline_detection_dimensions":
                normalized[key] = _baseline_dimensions_from_json(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _decay_curve_from_json(value: Any, reverse_decay_enabled: bool) -> DecayCurve:
    if value is not None:
        try:
            return DecayCurve(str(value))
        except ValueError:
            pass
    return DecayCurve.HYBRID if reverse_decay_enabled else DecayCurve.STANDARD_POWER_LAW


def _preferences_from_json(raw: dict[str, Any]) -> UserPreferences:
    preferences = UserPreferences()
    for key, value in raw.items():
        if not hasattr(preferences, key):
            continue
        try:
            current = getattr(preferences, key)
            if key == "mode":
                parsed: Any = Mode(value)
            elif key == "custom_profile":
                parsed = _custom_profile_from_json(value)
            elif key == "max_active_per_session":
                parsed = _max_active_from_json(value)
            elif key in {"nostalgia_tendency", "surprise_tendency", "depth_tendency"}:
                parsed = clamp(float(value))
            elif key == "time_presentation_mode":
                parsed = _time_presentation_from_json(value)
                if parsed is None:
                    continue
            elif key == "anchor_preference":
                parsed = _anchor_preference_from_json(value)
                if parsed is None:
                    continue
            elif key == "baseline_sensitivity":
                parsed = _baseline_sensitivity_from_json(value)
                if parsed is None:
                    continue
            elif key == "baseline_detection_dimensions":
                parsed = _baseline_dimensions_from_json(value)
            elif key == "data_export_permission":
                parsed = _data_export_permission_from_json(value)
                if parsed is None:
                    continue
            elif isinstance(current, bool):
                parsed = _bool_from_json(value)
            elif isinstance(current, int):
                parsed = int(value)
            elif isinstance(current, float):
                parsed = float(value)
            elif isinstance(current, list):
                parsed = value if isinstance(value, list) else str(value).split(",")
            else:
                parsed = value
        except (TypeError, ValueError):
            continue
        setattr(preferences, key, parsed)
    return preferences


def _baseline_from_json(raw: dict[str, Any]) -> EmotionalBaseline:
    baseline = EmotionalBaseline()
    for key, value in raw.items():
        if key == "last_updated":
            baseline.last_updated = _dt(value)
        elif hasattr(baseline, key):
            setattr(baseline, key, value)
    return baseline


def _patterns_from_json(raw: dict[str, Any]) -> InteractionPatterns:
    patterns = InteractionPatterns()
    for key, value in raw.items():
        if hasattr(patterns, key):
            setattr(patterns, key, value)
    return patterns


def _narrative_from_json(raw: dict[str, Any]) -> RelationshipNarrative:
    return RelationshipNarrative(
        framing=raw.get("framing", "NEUTRAL"),
        core_themes=raw.get("core_themes", []),
        origin_story=raw.get("origin_story"),
    )


def _core_identity_from_json(raw: dict[str, Any]) -> CoreIdentityRecord:
    return CoreIdentityRecord(
        identity_id=raw["identity_id"],
        relationship_id=raw["relationship_id"],
        memory_id=raw["memory_id"],
        title=raw["title"],
        content=raw["content"],
        created_at=_dt(raw["created_at"]),
        updated_at=_dt(raw["updated_at"]),
        protection_level=raw.get("protection_level", "L4"),
        review_status=raw.get("review_status", "UNREVIEWED"),
        review_score=raw.get("review_score", 0.0),
        review_history=raw.get("review_history", []),
        user_confirmed_at=_dt(raw["user_confirmed_at"]) if raw.get("user_confirmed_at") else None,
        change_log=raw.get("change_log", []),
        pending_delete=raw.get("pending_delete", False),
        replicas=raw.get("replicas", []),
    )


def _core_identity_delete_request_from_json(raw: dict[str, Any]) -> CoreIdentityDeleteRequest:
    return CoreIdentityDeleteRequest(
        request_id=raw["request_id"],
        identity_id=raw["identity_id"],
        relationship_id=raw["relationship_id"],
        memory_id=raw["memory_id"],
        requested_at=_dt(raw["requested_at"]),
        execute_after=_dt(raw["execute_after"]),
        reason=raw.get("reason", "user_delete"),
        status=ResetRequestStatus(raw.get("status", ResetRequestStatus.PENDING.value)),
        executed_at=_dt(raw["executed_at"]) if raw.get("executed_at") else None,
    )


def _memory_delete_request_from_json(raw: dict[str, Any]) -> MemoryDeleteRequest:
    return MemoryDeleteRequest(
        request_id=raw["request_id"],
        memory_id=raw["memory_id"],
        relationship_id=raw["relationship_id"],
        requested_at=_dt(raw["requested_at"]),
        execute_after=_dt(raw["execute_after"]),
        reason=raw.get("reason", "user_delete"),
        status=ResetRequestStatus(raw.get("status", ResetRequestStatus.PENDING.value)),
        executed_at=_dt(raw["executed_at"]) if raw.get("executed_at") else None,
    )


def _reset_request_from_json(raw: dict[str, Any]) -> ResetRequest:
    return ResetRequest(
        request_id=raw["request_id"],
        relationship_id=raw["relationship_id"],
        mode=ResetMode(raw["mode"]),
        requested_at=_dt(raw["requested_at"]),
        execute_after=_dt(raw["execute_after"]),
        status=ResetRequestStatus(raw.get("status", ResetRequestStatus.PENDING.value)),
        executed_at=_dt(raw["executed_at"]) if raw.get("executed_at") else None,
    )


def _health_alert_from_json(raw: dict[str, Any]) -> HealthAlert:
    return HealthAlert(
        alert_id=raw["alert_id"],
        relationship_id=raw["relationship_id"],
        risk_type=raw["risk_type"],
        level=HealthRiskLevel(raw["level"]),
        message=raw["message"],
        created_at=_dt(raw["created_at"]),
        source_memory_id=raw.get("source_memory_id"),
        acknowledged=raw.get("acknowledged", False),
        acknowledged_at=_dt(raw["acknowledged_at"]) if raw.get("acknowledged_at") else None,
        acknowledgement_note=raw.get("acknowledgement_note"),
        feedback=raw.get("feedback"),
        feedback_at=_dt(raw["feedback_at"]) if raw.get("feedback_at") else None,
        feedback_note=raw.get("feedback_note"),
        resources=raw.get("resources", []),
    )


def _guardian_summary_from_json(raw: dict[str, Any]) -> GuardianSummary:
    return GuardianSummary(
        summary_id=raw["summary_id"],
        relationship_id=raw["relationship_id"],
        period_start=_dt(raw["period_start"]),
        period_end=_dt(raw["period_end"]),
        generated_at=_dt(raw["generated_at"]),
        user_age=raw.get("user_age"),
        stage=RelationshipStage(raw["stage"]),
        interaction_count=raw.get("interaction_count", 0),
        total_minutes=raw.get("total_minutes", 0),
        memory_count=raw.get("memory_count", 0),
        emotional_memory_count=raw.get("emotional_memory_count", 0),
        active_behavior_count=raw.get("active_behavior_count", 0),
        health_alert_ids=raw.get("health_alert_ids", []),
        milestone_count=raw.get("milestone_count", 0),
        core_identity_count=raw.get("core_identity_count", 0),
        recommendation=raw.get("recommendation", ""),
        guardian_visible=raw.get("guardian_visible", True),
        privacy_boundary=raw.get("privacy_boundary", {}),
        resource_summary=raw.get("resource_summary", []),
    )


def _commitment_reminder_from_json(raw: dict[str, Any]) -> CommitmentReminder:
    return CommitmentReminder(
        reminder_id=raw["reminder_id"],
        relationship_id=raw["relationship_id"],
        memory_id=raw["memory_id"],
        title=raw["title"],
        source_text=raw.get("source_text", raw.get("title", "")),
        due_at=_dt(raw["due_at"]),
        created_at=_dt(raw["created_at"]),
        priority=raw.get("priority", "NORMAL"),
        status=ReminderStatus(raw.get("status", ReminderStatus.PENDING.value)),
        reminder_count=raw.get("reminder_count", 0),
        last_reminded_at=_dt(raw["last_reminded_at"]) if raw.get("last_reminded_at") else None,
        completed_at=_dt(raw["completed_at"]) if raw.get("completed_at") else None,
        archived_at=_dt(raw["archived_at"]) if raw.get("archived_at") else None,
    )
