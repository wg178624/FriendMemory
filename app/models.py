from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RelationshipStage(StrEnum):
    INITIATING = "INITIATING"
    EXPERIMENTING = "EXPERIMENTING"
    INTENSIFYING = "INTENSIFYING"
    INTEGRATING = "INTEGRATING"
    BONDING = "BONDING"
    DIFFERENTIATING = "DIFFERENTIATING"
    CIRCUMSCRIBING = "CIRCUMSCRIBING"
    STAGNATING = "STAGNATING"
    AVOIDING = "AVOIDING"
    TERMINATING = "TERMINATING"


class Mode(StrEnum):
    ASSISTANT = "assistant"
    FRIEND = "friend"
    CUSTOM = "custom"


class DecayCurve(StrEnum):
    STANDARD_POWER_LAW = "STANDARD_POWER_LAW"
    REVERSE_DECAY = "REVERSE_DECAY"
    HYBRID = "HYBRID"
    PERMANENT = "PERMANENT"


class MemoryType(StrEnum):
    IDENTITY = "identity"
    MILESTONE = "milestone"
    COMMITMENT = "commitment"
    EMOTIONAL_MOMENT = "emotional_moment"
    SHARED_EPISODE = "shared_episode"
    INSIDE_JOKE = "inside_joke"
    EMOTIONAL_PREFERENCE = "emotional_preference"
    FACT = "fact"
    CONTEXT_DETAIL = "context_detail"
    CONFLICT = "conflict"


class MemoryLayer(StrEnum):
    L1_IMMEDIATE = "L1_IMMEDIATE"
    L2_EPISODIC = "L2_EPISODIC"
    L3_RELATIONAL = "L3_RELATIONAL"
    L4_CORE_IDENTITY = "L4_CORE_IDENTITY"
    L5_RELATIONSHIP_HISTORY = "L5_RELATIONSHIP_HISTORY"


class ContextTag(StrEnum):
    INSIDE_JOKE = "INSIDE_JOKE"
    VULNERABLE_MOMENT = "VULNERABLE_MOMENT"
    SHARED_CELEBRATION = "SHARED_CELEBRATION"
    TURNING_POINT = "TURNING_POINT"
    COMFORT_MOMENT = "COMFORT_MOMENT"
    CONFLICT = "CONFLICT"
    REVELATION = "REVELATION"
    MILESTONE = "MILESTONE"
    UNRESOLVED_THREAD = "UNRESOLVED_THREAD"
    GENERAL = "GENERAL"


class ResetMode(StrEnum):
    SOFT = "SOFT_RESET"
    MEDIUM = "MEDIUM_RESET"
    HARD = "HARD_RESET"


class NarrativeLevel(StrEnum):
    FRAGMENT = "FRAGMENT"
    EPISODE = "EPISODE"
    CHAPTER = "CHAPTER"
    STORYLINE = "STORYLINE"


class ResetRequestStatus(StrEnum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"


class HealthRiskLevel(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class ReminderStatus(StrEnum):
    PENDING = "PENDING"
    REMINDER_SENT = "REMINDER_SENT"
    COMPLETED = "COMPLETED"
    ARCHIVED = "ARCHIVED"


@dataclass
class InteractionPatterns:
    avg_session_duration: float = 0.0
    avg_emotional_valence: float = 0.0
    self_disclosure_depth: float = 0.0
    response_consistency: float = 0.5
    initiation_ratio: float = 1.0


@dataclass
class RelationshipNarrative:
    framing: str = "NEUTRAL"
    core_themes: list[str] = field(default_factory=list)
    origin_story: str | None = None


@dataclass
class UserPreferences:
    mode: Mode = Mode.FRIEND
    active_recall_enabled: bool = True
    trust_bias_enabled: bool = True
    reverse_decay_enabled: bool = True
    emotional_layer_enabled: bool = True
    baseline_detection_enabled: bool = True
    baseline_sensitivity: str = "MEDIUM"
    baseline_detection_dimensions: list[str] = field(
        default_factory=lambda: [
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
    )
    level3_enabled: bool = True
    nostalgia_tendency: float = 0.5
    surprise_tendency: float = 0.5
    depth_tendency: float = 0.6
    max_active_per_session: int = 2
    time_presentation_mode: str = "AUTO"
    uncertainty_expression_enabled: bool = True
    data_export_permission: str = "FULL"
    anchor_preference: str = "EMOTION_FIRST"
    relationship_nature_disclosure_enabled: bool = True
    guardian_summary_enabled: bool = True
    custom_profile: dict[str, Any] = field(default_factory=dict)
    memory_writes_enabled: bool = True
    memory_pause_reason: str | None = None
    memory_paused_at: str | None = None


@dataclass
class EmotionalBaseline:
    avg_sentiment: float = 0.0
    std_sentiment: float = 0.25
    avg_response_length: float = 0.0
    std_response_length: float = 25.0
    vocabulary_richness: float = 0.0
    emoji_usage_rate: float = 0.0
    exclamation_rate: float = 0.0
    question_rate: float = 0.0
    topic_distribution: dict[str, float] = field(default_factory=dict)
    interaction_frequency: float = 0.0
    avg_response_latency: float = 0.0
    std_response_latency: float = 3600.0
    update_window_days: int = 30
    min_samples: int = 50
    sample_count: int = 0
    baseline_confidence: float = 0.0
    last_updated: datetime = field(default_factory=utcnow)


@dataclass
class Relationship:
    user_id: str
    ai_id: str
    relationship_id: str
    created_at: datetime = field(default_factory=utcnow)
    strength: float = 0.0
    stage: RelationshipStage = RelationshipStage.INITIATING
    trust_level: float = 0.5
    intimacy_level: float = 0.0
    relationship_age: int = 0
    last_interaction: datetime | None = None
    interaction_count: int = 0
    active_days: set[str] = field(default_factory=set)
    shared_episodes: list[str] = field(default_factory=list)
    inside_jokes: list[str] = field(default_factory=list)
    inside_joke_candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    milestones: list[str] = field(default_factory=list)
    unresolved_threads: list[str] = field(default_factory=list)
    interaction_patterns: InteractionPatterns = field(default_factory=InteractionPatterns)
    relationship_narrative: RelationshipNarrative = field(default_factory=RelationshipNarrative)
    decay_curve_type: DecayCurve = DecayCurve.HYBRID
    retention_multiplier: float = 1.0
    schema_version: int = 1
    last_updated: datetime = field(default_factory=utcnow)
    preferences: UserPreferences = field(default_factory=UserPreferences)
    emotional_baseline: EmotionalBaseline = field(default_factory=EmotionalBaseline)
    mode_history: list[dict[str, Any]] = field(default_factory=list)
    stage_history: list[dict[str, Any]] = field(default_factory=list)
    active_behavior_log: list[dict[str, Any]] = field(default_factory=list)
    core_identity: list[str] = field(default_factory=list)
    user_age: int | None = None
    daily_interaction_minutes: dict[str, int] = field(default_factory=dict)
    transparency_acknowledged_at: datetime | None = None
    baseline_deviation_state: dict[str, Any] = field(default_factory=dict)
    active_feedback_state: dict[str, Any] = field(default_factory=dict)
    retention_calibration_state: dict[str, Any] = field(default_factory=dict)
    maintenance_signals: dict[str, Any] = field(default_factory=dict)
    trust_decay_state: dict[str, Any] = field(default_factory=dict)
    implicit_topics: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EmotionLabel:
    name: str
    intensity: float


@dataclass
class EmotionalMemory:
    emotion_id: str
    relationship_id: str
    source_memory_id: str
    content: str
    timestamp: datetime
    relationship_age_at_creation: int
    emotions: list[EmotionLabel]
    primary_emotion: str
    emotional_valence: float
    emotional_arousal: float
    personal_importance: float
    self_disclosure_depth: float
    context_tag: ContextTag
    relationship_stage_at_creation: RelationshipStage
    trust_level_at_creation: float
    decay_curve: DecayCurve = DecayCurve.REVERSE_DECAY
    retention_multiplier: float = 1.0
    source_confidence: float = 0.7
    emotion_detection_confidence: float = 0.7
    human_verified: bool = False
    embeddings: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    memory_id: str
    relationship_id: str
    content: str
    memory_type: MemoryType
    context_tag: ContextTag
    created_at: datetime
    updated_at: datetime
    base_weight: float
    importance: float
    emotion_intensity: float = 0.0
    emotional_valence: float = 0.0
    mention_count: int = 1
    decay_curve: DecayCurve = DecayCurve.STANDARD_POWER_LAW
    relationship_stage_at_creation: RelationshipStage = RelationshipStage.INITIATING
    relationship_age_at_creation: int = 0
    trust_level_at_creation: float = 0.5
    storage_layer: MemoryLayer = MemoryLayer.L1_IMMEDIATE
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryGraphEdge:
    edge_id: str
    relationship_id: str
    source_memory_id: str
    target_memory_id: str
    relation_type: str
    weight: float
    created_at: datetime
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedStoryNode:
    story_id: str
    relationship_id: str
    title: str
    narrative_level: NarrativeLevel
    core_events: list[str]
    key_moments: list[str]
    recurring_themes: list[str]
    participants: list[str]
    story_arc_start: datetime
    story_arc_end: datetime | None = None
    retell_count: int = 1
    last_retold: datetime = field(default_factory=utcnow)
    consensus_version: str = ""
    conflict_versions: list[str] = field(default_factory=list)
    correction_versions: list[dict[str, Any]] = field(default_factory=list)
    narrative_versions: list[dict[str, Any]] = field(default_factory=list)
    child_inside_jokes: list[str] = field(default_factory=list)
    consistency_score: float = 0.7
    emotional_arc: list[dict[str, Any]] = field(default_factory=list)
    user_framing: str = "NEUTRAL"
    ai_framing_confidence: float = 0.7
    consensus_status: str = "SIMULATED_FROM_USER_ACCOUNT"
    consensus_provenance: dict[str, Any] = field(default_factory=dict)
    consensus_confirmed_at: datetime | None = None


@dataclass
class EmotionalTrajectoryWindow:
    window_start: datetime
    window_end: datetime
    avg_valence: float
    avg_arousal: float
    dominant_emotions: list[str]
    emotional_diversity: float
    notable_events: list[str]


@dataclass
class EmotionalTrajectory:
    relationship_id: str
    time_series: list[EmotionalTrajectoryWindow] = field(default_factory=list)
    detected_patterns: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CoreIdentityRecord:
    identity_id: str
    relationship_id: str
    memory_id: str
    title: str
    content: str
    created_at: datetime
    updated_at: datetime
    protection_level: str = "L4"
    review_status: str = "UNREVIEWED"
    review_score: float = 0.0
    review_history: list[dict[str, Any]] = field(default_factory=list)
    user_confirmed_at: datetime | None = None
    change_log: list[dict[str, Any]] = field(default_factory=list)
    pending_delete: bool = False
    replicas: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ResetRequest:
    request_id: str
    relationship_id: str
    mode: ResetMode
    requested_at: datetime
    execute_after: datetime
    status: ResetRequestStatus = ResetRequestStatus.PENDING
    executed_at: datetime | None = None


@dataclass
class CoreIdentityDeleteRequest:
    request_id: str
    identity_id: str
    relationship_id: str
    memory_id: str
    requested_at: datetime
    execute_after: datetime
    reason: str = "user_delete"
    status: ResetRequestStatus = ResetRequestStatus.PENDING
    executed_at: datetime | None = None


@dataclass
class MemoryDeleteRequest:
    request_id: str
    memory_id: str
    relationship_id: str
    requested_at: datetime
    execute_after: datetime
    reason: str = "user_delete"
    status: ResetRequestStatus = ResetRequestStatus.PENDING
    executed_at: datetime | None = None


@dataclass
class HealthAlert:
    alert_id: str
    relationship_id: str
    risk_type: str
    level: HealthRiskLevel
    message: str
    created_at: datetime
    source_memory_id: str | None = None
    acknowledged: bool = False
    acknowledged_at: datetime | None = None
    acknowledgement_note: str | None = None
    feedback: str | None = None
    feedback_at: datetime | None = None
    feedback_note: str | None = None
    resources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GuardianSummary:
    summary_id: str
    relationship_id: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    user_age: int | None
    stage: RelationshipStage
    interaction_count: int
    total_minutes: int
    memory_count: int
    emotional_memory_count: int
    active_behavior_count: int
    health_alert_ids: list[str]
    milestone_count: int
    core_identity_count: int
    recommendation: str
    guardian_visible: bool = True
    privacy_boundary: dict[str, Any] = field(default_factory=dict)
    resource_summary: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CommitmentReminder:
    reminder_id: str
    relationship_id: str
    memory_id: str
    title: str
    source_text: str
    due_at: datetime
    created_at: datetime
    priority: str = "NORMAL"
    status: ReminderStatus = ReminderStatus.PENDING
    reminder_count: int = 0
    last_reminded_at: datetime | None = None
    completed_at: datetime | None = None
    archived_at: datetime | None = None


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"
