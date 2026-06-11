from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime

from models import (
    ContextTag,
    DecayCurve,
    MemoryRecord,
    MemoryType,
    Relationship,
    RelationshipStage,
)


TYPE_MULTIPLIER = {
    MemoryType.IDENTITY: 2.0,
    MemoryType.MILESTONE: 2.0,
    MemoryType.COMMITMENT: 1.8,
    MemoryType.EMOTIONAL_MOMENT: 1.6,
    MemoryType.SHARED_EPISODE: 1.5,
    MemoryType.INSIDE_JOKE: 1.4,
    MemoryType.EMOTIONAL_PREFERENCE: 1.0,
    MemoryType.FACT: 0.8,
    MemoryType.CONTEXT_DETAIL: 0.5,
    MemoryType.CONFLICT: 0.8,
}

STAGE_ALPHA = {
    RelationshipStage.INITIATING: 0.05,
    RelationshipStage.EXPERIMENTING: 0.08,
    RelationshipStage.INTENSIFYING: 0.12,
    RelationshipStage.INTEGRATING: 0.20,
    RelationshipStage.BONDING: 0.30,
    RelationshipStage.DIFFERENTIATING: 0.12,
    RelationshipStage.CIRCUMSCRIBING: 0.08,
    RelationshipStage.STAGNATING: 0.05,
    RelationshipStage.AVOIDING: 0.05,
    RelationshipStage.TERMINATING: 0.05,
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def lexical_similarity(a: str, b: str) -> float:
    ta = Counter(tokenize(a))
    tb = Counter(tokenize(b))
    if not ta or not tb:
        return 0.0
    common = sum((ta & tb).values())
    denom = math.sqrt(sum(v * v for v in ta.values())) * math.sqrt(sum(v * v for v in tb.values()))
    return common / denom if denom else 0.0


def age_days(created_at: datetime, now: datetime) -> float:
    return max(0.0, (now - created_at).total_seconds() / 86400)


def standard_power_law_weight(memory: MemoryRecord, now: datetime) -> float:
    if memory.memory_type in {MemoryType.IDENTITY, MemoryType.MILESTONE}:
        return 1.0
    days = age_days(memory.created_at, now)
    return max(_cold_review_weight_floor(memory, now), memory.base_weight * ((1 + days) ** -0.65))


def reverse_decay_weight(memory: MemoryRecord, relationship: Relationship, now: datetime) -> float:
    if memory.decay_curve == DecayCurve.PERMANENT or memory.memory_type in {MemoryType.IDENTITY, MemoryType.MILESTONE}:
        return 1.0
    days = age_days(memory.created_at, now)
    alpha = STAGE_ALPHA.get(relationship.stage, 0.15)
    mentions = math.log1p(min(memory.mention_count, 100))
    type_multiplier = relationship_type_multiplier(memory, relationship)
    emotion_multiplier = 1 + 1.5 * (memory.emotion_intensity**2)
    retention_multiplier = relationship.retention_multiplier * retention_calibration_multiplier(relationship)
    cooldown = memory.metadata.get("trust_soft_cooldown")
    if isinstance(cooldown, dict) and relationship.preferences.trust_bias_enabled and not is_trust_bias_protected(memory):
        try:
            retention_multiplier = min(retention_multiplier, float(cooldown.get("retention_multiplier", retention_multiplier)))
        except (TypeError, ValueError):
            pass
    value = (
        memory.base_weight
        * (1 + alpha * mentions * math.log1p(days))
        * max(0.1, relationship.strength)
        * type_multiplier
        * emotion_multiplier
        * retention_multiplier
    )
    return max(_cold_review_weight_floor(memory, now), value)


def _cold_review_weight_floor(memory: MemoryRecord, now: datetime) -> float:
    review = memory.metadata.get("cold_information_review")
    if not isinstance(review, dict):
        return 0.0
    protected_until = review.get("protected_until")
    if not protected_until:
        return 0.0
    try:
        until = datetime.fromisoformat(str(protected_until))
    except ValueError:
        return 0.0
    if until < now:
        return 0.0
    status = str(review.get("status") or "").upper()
    criticality = str(memory.metadata.get("criticality") or "").upper()
    if status.startswith("CONFIRMED") and criticality in {"CRITICAL", "SAFETY", "MEDICAL", "MAJOR_COMMITMENT"}:
        return 0.95
    if status == "CONFIRMED_HIGH":
        return 0.80
    return 0.0


def retention_calibration_multiplier(relationship: Relationship) -> float:
    state = relationship.retention_calibration_state
    if not isinstance(state, dict):
        return 1.0
    try:
        offset = float(state.get("multiplier_offset", 0.0) or 0.0)
    except (TypeError, ValueError):
        offset = 0.0
    return max(0.70, min(1.30, 1.0 + offset))


def relationship_type_multiplier(memory: MemoryRecord, relationship: Relationship) -> float:
    harmful = memory.context_tag == ContextTag.CONFLICT or memory.emotional_valence <= -0.5 or memory.memory_type == MemoryType.CONFLICT
    if (
        harmful
        and trust_bias_stage_enabled(relationship)
        and relationship.preferences.trust_bias_enabled
        and relationship.trust_level < 0.4
        and not is_trust_bias_protected(memory)
    ):
        return max(TYPE_MULTIPLIER.get(memory.memory_type, 1.0), 1.2)
    return TYPE_MULTIPLIER.get(memory.memory_type, 1.0)


def is_trust_bias_protected(memory: MemoryRecord) -> bool:
    if memory.importance >= 0.95:
        return True
    protection = str(memory.metadata.get("criticality") or memory.metadata.get("severity") or "").upper()
    if protection in {"CRITICAL", "SAFETY", "MEDICAL", "MAJOR_COMMITMENT"}:
        return True
    return bool(memory.metadata.get("trust_bias_protected"))


def trust_bias_stage_enabled(relationship: Relationship) -> bool:
    return relationship.stage in {
        RelationshipStage.INTENSIFYING,
        RelationshipStage.INTEGRATING,
        RelationshipStage.BONDING,
        RelationshipStage.DIFFERENTIATING,
    }


def reverse_decay_stage_enabled(relationship: Relationship) -> bool:
    return relationship.stage in {
        RelationshipStage.INTENSIFYING,
        RelationshipStage.INTEGRATING,
        RelationshipStage.BONDING,
        RelationshipStage.DIFFERENTIATING,
        RelationshipStage.CIRCUMSCRIBING,
    }


def memory_weight(memory: MemoryRecord, relationship: Relationship, now: datetime) -> float:
    if (
        relationship.preferences.mode.value == "assistant"
        or not relationship.preferences.reverse_decay_enabled
        or not reverse_decay_stage_enabled(relationship)
    ):
        return standard_power_law_weight(memory, now)
    relationship_curve = relationship.decay_curve_type
    if relationship_curve == DecayCurve.STANDARD_POWER_LAW:
        return standard_power_law_weight(memory, now)
    if relationship_curve == DecayCurve.PERMANENT:
        if memory.decay_curve in {DecayCurve.REVERSE_DECAY, DecayCurve.PERMANENT} or memory.memory_type in {
            MemoryType.IDENTITY,
            MemoryType.MILESTONE,
            MemoryType.COMMITMENT,
            MemoryType.EMOTIONAL_MOMENT,
            MemoryType.SHARED_EPISODE,
            MemoryType.INSIDE_JOKE,
        }:
            return 1.0
        return standard_power_law_weight(memory, now)
    if relationship_curve == DecayCurve.REVERSE_DECAY:
        if memory.decay_curve in {DecayCurve.REVERSE_DECAY, DecayCurve.PERMANENT, DecayCurve.HYBRID}:
            return reverse_decay_weight(memory, relationship, now)
        return standard_power_law_weight(memory, now)
    if memory.decay_curve in {DecayCurve.REVERSE_DECAY, DecayCurve.PERMANENT}:
        return reverse_decay_weight(memory, relationship, now)
    if memory.decay_curve == DecayCurve.HYBRID:
        return max(standard_power_law_weight(memory, now), reverse_decay_weight(memory, relationship, now) * 0.7)
    return standard_power_law_weight(memory, now)


def apply_trust_bias(score: float, memory: MemoryRecord, relationship: Relationship) -> float:
    if not relationship.preferences.trust_bias_enabled or not trust_bias_stage_enabled(relationship):
        return score
    harmful = memory.context_tag == ContextTag.CONFLICT or memory.emotional_valence <= -0.5 or memory.memory_type == MemoryType.CONFLICT
    if not harmful:
        return score
    if is_trust_bias_protected(memory):
        return score
    trust = relationship.trust_level
    if trust >= 0.5:
        return score * (1 - 0.6 * max(0.0, trust - 0.5))
    return score * (1 + (0.4 - trust))


def temporal_fuzz(created_at: datetime, now: datetime, emotion_intensity: float) -> dict[str, float | str]:
    days = age_days(created_at, now)
    fuzz_level = min(1.0, math.log2(max(days, 7) / 7) * 0.20)
    detail_retention = clamp(emotion_intensity * (1 - fuzz_level * 0.5))
    if days < 7:
        grain = "day"
        phrase = "最近几天"
    elif days < 30:
        grain = "week"
        phrase = "前阵子"
    elif days < 180:
        grain = "month_or_season"
        phrase = "几个月前"
    elif days < 730:
        grain = "season_or_year"
        phrase = "去年那阵"
    else:
        grain = "very_fuzzy"
        phrase = "很久以前"
    return {"fuzz_level": fuzz_level, "detail_retention": detail_retention, "grain": grain, "phrase": phrase}
