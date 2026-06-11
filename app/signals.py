from __future__ import annotations

import re
from dataclasses import dataclass

from models import ContextTag, MemoryType
from scoring import clamp


SELF_DISCLOSURE = ["其实", "说实话", "只有你", "从来没告诉", "私下", "我害怕", "我崩溃", "我哭"]
RELATION_WORDS = ["我们", "咱们", "一起", "咱俩", "认识", "陪我"]
COMMITMENT_WORDS = ["答应", "约定", "以后", "下次", "决定", "承诺", "deadline", "截止"]
CELEBRATION_WORDS = ["太好了", "开心", "成功", "终于", "拿到了", "通过了", "庆祝"]
CONFLICT_WORDS = ["你不懂", "算了", "失望", "生气", "吵", "冲突", "别提"]
NEGATIVE_WORDS = ["难过", "焦虑", "害怕", "崩溃", "哭", "失眠", "痛苦", "压力", "抑郁"]
POSITIVE_WORDS = ["开心", "感谢", "谢谢", "喜欢", "安心", "温暖", "庆祝", "骄傲"]


@dataclass(frozen=True)
class TurnSignals:
    relationship_depth: float
    emotion_intensity: float
    personal_importance: float
    time_preciousness: float
    self_disclosure_depth: float
    sentiment: float
    arousal: float
    context_tag: ContextTag
    memory_type: MemoryType
    unresolved_thread: bool
    milestone_candidate: bool
    inside_joke_candidate: str | None
    trust_delta: float


def _contains_any(text: str, words: list[str]) -> bool:
    return any(word.lower() in text.lower() for word in words)


def _emoji_count(text: str) -> int:
    return len(re.findall(r"[\U0001f300-\U0001faff]", text))


def detect_turn_signals(text: str, relationship_strength: float, trust_level: float, intimacy_level: float) -> TurnSignals:
    self_disclosure = 0.7 if _contains_any(text, SELF_DISCLOSURE) else 0.0
    if "只有你" in text or "从来没告诉" in text:
        self_disclosure = 0.9

    punctuation = min(1.0, (text.count("!") + text.count("！") + text.count("…") + text.count("。") * 0.1) / 4)
    emoji = min(0.5, _emoji_count(text) * 0.15)
    negative_hits = sum(1 for word in NEGATIVE_WORDS if word in text)
    positive_hits = sum(1 for word in POSITIVE_WORDS if word in text)
    emotion_intensity = clamp(0.15 + punctuation + emoji + 0.18 * (negative_hits + positive_hits) + 0.3 * self_disclosure)
    sentiment = clamp((positive_hits - negative_hits) * 0.25, -1.0, 1.0)
    arousal = clamp(emotion_intensity)

    relation_bonus = 0.2 if _contains_any(text, RELATION_WORDS) else 0.0
    relationship_depth = clamp(0.5 * relationship_strength + 0.3 * trust_level + 0.2 * intimacy_level + relation_bonus)

    commitment = _contains_any(text, COMMITMENT_WORDS)
    unresolved = commitment or "还没决定" in text or "想去" in text
    celebration = _contains_any(text, CELEBRATION_WORDS)
    conflict = _contains_any(text, CONFLICT_WORDS)
    milestone = "第一次" in text or celebration or "纪念日" in text

    personal_importance = clamp(0.4 * self_disclosure + (0.25 if commitment else 0) + (0.25 if milestone else 0) + 0.1)
    time_preciousness = clamp((0.5 if milestone else 0) + (0.3 if unresolved else 0) + min(0.2, relationship_strength * 0.2))

    if conflict:
        context_tag = ContextTag.CONFLICT
        memory_type = MemoryType.CONFLICT
        trust_delta = -0.06
    elif milestone:
        context_tag = ContextTag.MILESTONE if "第一次" in text or "纪念日" in text else ContextTag.SHARED_CELEBRATION
        memory_type = MemoryType.MILESTONE if context_tag == ContextTag.MILESTONE else MemoryType.SHARED_EPISODE
        trust_delta = 0.03
    elif self_disclosure >= 0.7:
        context_tag = ContextTag.VULNERABLE_MOMENT
        memory_type = MemoryType.EMOTIONAL_MOMENT
        trust_delta = 0.05
    elif commitment:
        context_tag = ContextTag.UNRESOLVED_THREAD
        memory_type = MemoryType.COMMITMENT
        trust_delta = 0.01
    elif emotion_intensity >= 0.55:
        context_tag = ContextTag.COMFORT_MOMENT if sentiment < 0 else ContextTag.SHARED_CELEBRATION
        memory_type = MemoryType.EMOTIONAL_MOMENT
        trust_delta = 0.01 if sentiment >= 0 else 0.0
    else:
        context_tag = ContextTag.GENERAL
        memory_type = MemoryType.FACT
        trust_delta = 0.0

    inside_joke = None
    quoted = re.findall(r"[“\"]([^”\"]{2,16})[”\"]", text)
    if quoted:
        inside_joke = quoted[0]

    return TurnSignals(
        relationship_depth=relationship_depth,
        emotion_intensity=emotion_intensity,
        personal_importance=personal_importance,
        time_preciousness=time_preciousness,
        self_disclosure_depth=self_disclosure,
        sentiment=sentiment,
        arousal=arousal,
        context_tag=context_tag,
        memory_type=memory_type,
        unresolved_thread=unresolved,
        milestone_candidate=milestone,
        inside_joke_candidate=inside_joke,
        trust_delta=trust_delta,
    )
