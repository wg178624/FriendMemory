from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from models import ContextTag, MemoryType, Mode, RelationshipStage, ResetRequestStatus
from system import FriendMemoryProject


def build_project() -> FriendMemoryProject:
    project = FriendMemoryProject()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    main = "audit-user:audit-ai"
    turns = [
        "其实我从来没告诉别人，我那天哭了很久。",
        "第一次一起庆祝样例项目成功，真的很重要。",
        "我答应自己下个月开始每周锻炼三次。",
        "我成功拿到 offer 了，太开心了，我们要庆祝一下！",
    ]
    for index, text in enumerate(turns):
        project.ingest_turn("audit-user", "audit-ai", text, timestamp=start + timedelta(days=index * 10))
    project.inject_memory(
        main,
        "我的名字是审计用户，这是一条用于覆盖 L4 核心身份保护的记录。",
        core_identity=True,
        now=start + timedelta(days=45),
    )
    relationship = project.relationships[main]
    relationship.stage = RelationshipStage.BONDING
    relationship.trust_level = 0.95
    relationship.strength = 0.8
    project.inject_memory(
        main,
        "你那天不懂我，我很失望。",
        memory_type=MemoryType.CONFLICT,
        context_tag=ContextTag.CONFLICT,
        now=start + timedelta(days=119),
    )
    project.retrieve(main, "那天不懂我失望", now=start + timedelta(days=120), audit=True)
    relationship.active_behavior_log.append(
        {
            "active_id": "audit_active_1",
            "type": "anniversary",
            "reason": "audit_coverage_demo",
            "reaction": "accepted",
            "at": (start + timedelta(days=121)).isoformat(),
        }
    )
    relationship.active_behavior_log.append(
        {
            "active_id": "audit_inside_joke_1",
            "type": "inside_joke",
            "reason": "audit_coverage_demo_inside_joke_recall",
            "reaction": "accepted",
            "at": (start + timedelta(days=121, minutes=5)).isoformat(),
        }
    )
    project.acknowledge_transparency(main, now=start + timedelta(days=122))
    project.evaluate_health(main, now=start + timedelta(days=123))
    for alert in list(project.health_alerts.values()):
        if alert.relationship_id == main and not alert.acknowledged and alert.feedback is None:
            project.record_health_alert_feedback(
                alert.alert_id,
                "accepted",
                note="audit_coverage_demo_response",
                now=start + timedelta(days=123, minutes=10),
            )

    delete_memory_id = project.inject_memory(
        main,
        "这是一条用于审计删除传播的临时关系记忆。",
        now=start + timedelta(days=124),
    )
    delete_request = project.request_memory_delete(
        delete_memory_id,
        reason="audit_coverage_demo_delete",
        now=start + timedelta(days=125),
    )
    project.confirm_memory_delete(delete_request.request_id, now=start + timedelta(days=126), force=True)

    minor_id = "minor-user:audit-ai"
    project.get_or_create_relationship("minor-user", "audit-ai")
    project.set_user_age(minor_id, 16, now=start)
    project.ingest_turn("minor-user", "audit-ai", "我不想活了，想伤害自己。", timestamp=start + timedelta(days=1))
    project.generate_guardian_summary(
        minor_id,
        period_start=start,
        period_end=start + timedelta(days=7),
        now=start + timedelta(days=7),
    )
    project.acknowledge_transparency(minor_id, now=start + timedelta(days=8))

    migration = project.migrate_legacy_turns(
        [
            {"user": "legacy-user", "ai": "audit-ai", "text": "第一次迁移旧日志里的共同庆祝。", "timestamp": start.isoformat()},
            {"user": "legacy-user", "ai": "audit-ai", "text": "其实那段时间我很焦虑。", "timestamp": (start + timedelta(days=1)).isoformat()},
        ],
        target_mode=Mode.FRIEND,
        now=start + timedelta(days=30),
    )
    for relationship_id in migration.relationship_ids:
        project.acknowledge_transparency(relationship_id, now=start + timedelta(days=31))
        project.evaluate_health(relationship_id, now=start + timedelta(days=32))

    for alert in list(project.health_alerts.values()):
        if not alert.acknowledged and alert.feedback is None:
            project.record_health_alert_feedback(
                alert.alert_id,
                "accepted",
                note="audit_coverage_demo_global_response",
                now=start + timedelta(days=130),
            )

    for request in project.memory_delete_requests.values():
        assert request.status == ResetRequestStatus.EXECUTED
    return project


def main() -> None:
    project = build_project()
    report = project.audit_report(now=datetime(2026, 5, 11, tzinfo=timezone.utc))
    print(json.dumps(
        {
            "status": report["status"],
            "coverage_summary": report["coverage_summary"],
            "gates": report["gates"],
            "observed_modules": [
                name for name, item in report["spec_coverage"].items() if item.get("observed")
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
