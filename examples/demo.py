from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from system import FriendMemoryProject


project = FriendMemoryProject()
start = datetime(2025, 5, 28, tzinfo=timezone.utc)

turns = [
    "这是一条虚构演示：我完成样例项目时压力很大。",
    "谢谢你陪我梳理完那次样例项目，我们以后就叫它“样例暗号A”吧。",
    "第一次感觉有人认真理解我的测试场景，今天很重要。",
    "我答应自己下个月开始每周锻炼三次，还没决定具体时间。",
    "我成功拿到 offer 了，太好了！！！我们得庆祝一下。",
]

for i, text in enumerate(turns):
    result = project.ingest_turn("demo-user", "demo-ai", text, timestamp=start + timedelta(days=i * 12))
    print(result.stage.value, f"{result.score:.3f}", text)
    for suggestion in result.active_suggestions:
        print("  active:", suggestion)

relationship_id = "demo-user:demo-ai"
print("\nretrieval:")
for item in project.retrieve(relationship_id, "我最近又想起样例项目", now=start + timedelta(days=400), limit=3):
    print(f"{item.score:.3f}", item.presentation_time["phrase"], item.memory.content)
