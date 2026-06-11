from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from env_config import load_dotenv
from models import ContextTag, DecayCurve, MemoryLayer, MemoryType, Mode, ResetMode
from system import FriendMemoryProject

load_dotenv()
STATE_PATH = Path(os.environ.get("FRIEND_MEMORY_STATE", "data/friend_memory_state.json"))
EVALUATION_TASKS = [
    "stage_detection",
    "inside_joke_detection",
    "emotional_resonance_retrieval",
    "self_disclosure_capture",
    "story_quality",
    "friend_mode_ab",
    "production_telemetry",
]
EVIDENCE_FILENAMES = {
    "stage_detection": "stage-labels.json",
    "self_disclosure_capture": "self-disclosure-labels.json",
    "inside_joke_detection": "inside-joke-labels.json",
    "emotional_resonance_retrieval": "emotional-labels.json",
    "story_quality": "story-quality-labels.json",
    "friend_mode_ab": "friend-mode-ab.json",
    "production_telemetry": "production-telemetry.json",
}
EVIDENCE_DATASET_SCHEMA = "friend-memory-evidence-v1"
AI_OBSERVATION_FILENAME = "ai-observation.json"
AI_OBSERVATION_MAX_AGE_DAYS = 30
AI_OBSERVATION_FUTURE_SKEW_MINUTES = 5
EXTERNAL_AI_PARTICIPATION_KINDS = {"external_http_worker", "external_model"}
FORMAL_EVIDENCE_REQUIREMENTS = {
    "stage_detection": {"min_samples": 200, "kind": "labelled_dataset"},
    "self_disclosure_capture": {"min_samples": 200, "kind": "labelled_dataset"},
    "inside_joke_detection": {"min_samples": 200, "kind": "labelled_dataset"},
    "emotional_resonance_retrieval": {"min_samples": 200, "kind": "labelled_dataset"},
    "story_quality": {"min_samples": 200, "kind": "human_review_sample"},
    "friend_mode_ab": {"min_users_per_cohort": 1000, "min_duration_weeks": 12, "kind": "ab_experiment"},
    "production_telemetry": {"min_active_users": 1000, "min_duration_days": 30, "kind": "production_telemetry"},
}
FORMAL_PROVENANCE_REQUIREMENTS = {
    "labelled_dataset": [
        "dataset_id",
        "collected_at",
        "owner",
        "review_protocol",
        "privacy_redaction",
    ],
    "human_review_sample": [
        "dataset_id",
        "collected_at",
        "owner",
        "review_protocol",
        "privacy_redaction",
    ],
    "ab_experiment": [
        "experiment_id",
        "start_date",
        "end_date",
        "analysis_owner",
        "assignment_method",
        "primary_metrics",
        "statistical_test",
        "privacy_redaction",
    ],
    "production_telemetry": [
        "telemetry_id",
        "collected_at",
        "owner",
        "source_system",
        "aggregation_window",
        "privacy_redaction",
    ],
}
EVIDENCE_MANIFEST_FILENAME = "evidence-manifest.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def shell_arg(value: object) -> str:
    return shlex.quote(str(value))


def _doctor_item(name: str, ok: bool, detail: str, *, severity: str = "required") -> dict:
    return {"name": name, "ok": ok, "severity": severity, "detail": detail}


def _html_meta_value(html_text: str, name: str) -> str | None:
    match = re.search(rf'<meta name="{re.escape(name)}" content="([^"]*)">', html_text)
    return match.group(1) if match else None


def friend_memory_html_source_check(root: Path = PROJECT_ROOT) -> dict:
    source_relative = "docs/[重要]朋友类记忆系统改进方案.md"
    html_relative = "docs/朋友类记忆系统改进方案.html"
    source_path = root / source_relative
    html_path = root / html_relative
    if not source_path.exists() or not html_path.exists():
        missing = [relative for relative, path in [(source_relative, source_path), (html_relative, html_path)] if not path.exists()]
        return {"ok": False, "detail": f"missing={','.join(missing)}", "metadata": {}}
    source = source_path.read_text(encoding="utf-8")
    html_text = html_path.read_text(encoding="utf-8")
    source_lines = len(source.splitlines()) + (1 if source.endswith("\n") else 0)
    source_headings = len([line for line in source.splitlines() if re.match(r"^#{1,4}\s+", line)])
    expected = {
        "source-path": source_relative,
        "source-lines": str(source_lines),
        "source-headings": str(source_headings),
        "source-chars": str(len(source)),
        "source-sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
    actual = {name: _html_meta_value(html_text, name) for name in expected}
    mismatches = [
        f"{name}:expected={expected[name]} actual={actual[name]}"
        for name in expected
        if actual[name] != expected[name]
    ]
    return {
        "ok": not mismatches,
        "detail": "metadata matches current source markdown" if not mismatches else "; ".join(mismatches),
        "metadata": actual,
    }


def http_route_catalog_source_check(root: Path = PROJECT_ROOT) -> dict:
    server_path = root / "app" / "server.py"
    if not server_path.exists():
        return {"ok": False, "detail": f"missing={server_path}"}
    source = server_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(server_path))
    except SyntaxError as exc:
        return {"ok": False, "detail": f"server.py syntax error: {exc}"}

    routes: list[dict] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "HTTP_ROUTES" for target in node.targets):
            try:
                routes = ast.literal_eval(node.value)
            except (ValueError, SyntaxError) as exc:
                return {"ok": False, "detail": f"HTTP_ROUTES is not literal data: {exc}"}
            break
    if routes is None:
        return {"ok": False, "detail": "HTTP_ROUTES missing"}

    route_keys = {
        (str(item.get("method")), str(item.get("path")))
        for item in routes
        if isinstance(item, dict)
    }

    def implemented_paths(method_name: str) -> set[str]:
        match = re.search(
            rf"def _do_{method_name}_unlocked\(self\) -> None:\n(?P<body>.*?)(?:\n    def |\n\ndef )",
            source,
            re.S,
        )
        if not match:
            return set()
        return set(re.findall(r'parsed\.path == "([^"]+)"', match.group("body")))

    implemented = {
        *{("GET", path) for path in implemented_paths("GET")},
        *{("POST", path) for path in implemented_paths("POST")},
    }
    missing = sorted(implemented - route_keys)
    extra = sorted(route_keys - implemented)
    malformed = [
        item
        for item in routes
        if not isinstance(item, dict)
        or not {"method", "path", "category", "writes_state", "description"}.issubset(item)
    ]
    if missing or extra or malformed:
        return {
            "ok": False,
            "detail": f"missing={missing} extra={extra} malformed={len(malformed)}",
        }
    return {
        "ok": True,
        "detail": f"routes={len(route_keys)} methods={','.join(sorted({method for method, _ in route_keys}))}",
    }


def readme_cli_catalog_check(root: Path = PROJECT_ROOT) -> dict:
    readme_path = root / "README.md"
    if not readme_path.exists():
        return {"ok": False, "detail": "README.md missing"}
    readme = readme_path.read_text(encoding="utf-8")
    documented = set(
        re.findall(r"python app/main\.py(?:\s+--state\s+\S+)?\s+([a-z0-9][a-z0-9-]*)", readme)
    )
    catalog = cli_command_catalog()
    registered = {item["name"] for item in catalog["commands"]}
    missing = sorted(documented - registered)
    undocumented = sorted(registered - documented)
    if missing or undocumented:
        return {"ok": False, "detail": f"missing={missing} undocumented={undocumented}"}
    return {"ok": True, "detail": f"documented={len(documented)} registered={len(registered)}"}


def readme_http_route_catalog_check(root: Path = PROJECT_ROOT) -> dict:
    readme_path = root / "README.md"
    server_check = http_route_catalog_source_check(root)
    if not server_check["ok"]:
        return {"ok": False, "detail": f"server_route_catalog_invalid:{server_check['detail']}"}
    if not readme_path.exists():
        return {"ok": False, "detail": "README.md missing"}
    readme = readme_path.read_text(encoding="utf-8")
    documented: set[tuple[str, str]] = set()
    for line in readme.splitlines():
        match = re.search(r"curl(?:\s+-X\s+(?P<method>[A-Z]+))?\s+'?http://127\.0\.0\.1:8765(?P<path>/[^'?\s]+)", line)
        if match:
            documented.add((match.group("method") or "GET", match.group("path")))

    server_path = root / "app" / "server.py"
    source = server_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(server_path))
    routes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "HTTP_ROUTES" for target in node.targets):
            routes = ast.literal_eval(node.value)
            break
    route_keys = {
        (str(item.get("method")), str(item.get("path")))
        for item in routes
        if isinstance(item, dict)
    }
    missing = sorted(documented - route_keys)
    if missing:
        return {"ok": False, "detail": f"missing={missing}"}
    return {"ok": True, "detail": f"documented={len(documented)} registered={len(route_keys)}"}


def readme_local_file_references_check(root: Path = PROJECT_ROOT) -> dict:
    readme_path = root / "README.md"
    if not readme_path.exists():
        return {"ok": False, "detail": "README.md missing"}
    readme = readme_path.read_text(encoding="utf-8")
    referenced_files = {
        item.rstrip(".,;:")
        for item in re.findall(r"\b(?:app|examples|scripts|docs)/[^\s)'\"`]+", readme)
        if Path(item.rstrip(".,;:")).suffix
    }
    missing = sorted(path for path in referenced_files if not (root / path).is_file())
    if missing:
        return {"ok": False, "detail": f"missing={missing}"}
    return {"ok": True, "detail": f"referenced_files={len(referenced_files)}"}


def readme_project_structure_examples_check(root: Path = PROJECT_ROOT) -> dict:
    readme_path = root / "README.md"
    if not readme_path.exists():
        return {"ok": False, "detail": "README.md missing"}
    readme = readme_path.read_text(encoding="utf-8")
    structure_match = re.search(r"## 项目结构\n\n```text\n(?P<body>.*?)\n```", readme, re.S)
    if not structure_match:
        return {"ok": False, "detail": "project structure block missing"}
    structure = structure_match.group("body")
    runnable_examples = {
        Path(match).name
        for match in re.findall(r"python (examples/[a-zA-Z0-9_./-]+\.py)", readme)
    }
    missing = sorted(name for name in runnable_examples if name not in structure)
    if missing:
        return {"ok": False, "detail": f"missing_examples={missing}"}
    return {"ok": True, "detail": f"runnable_examples={len(runnable_examples)}"}


def readme_project_structure_docs_check(root: Path = PROJECT_ROOT) -> dict:
    readme_path = root / "README.md"
    if not readme_path.exists():
        return {"ok": False, "detail": "README.md missing"}
    readme = readme_path.read_text(encoding="utf-8")
    structure_match = re.search(r"## 项目结构\n\n```text\n(?P<body>.*?)\n```", readme, re.S)
    if not structure_match:
        return {"ok": False, "detail": "project structure block missing"}
    structure = structure_match.group("body")
    required_docs = [
        "[重要]朋友类记忆系统改进方案.md",
        "朋友类记忆系统改进方案.html",
        "实现验收总览.md",
    ]
    missing = [name for name in required_docs if name not in structure]
    if missing:
        return {"ok": False, "detail": f"missing_docs={missing}"}
    return {"ok": True, "detail": f"docs={len(required_docs)}"}


def readme_http_curl_json_bodies_check(root: Path = PROJECT_ROOT) -> dict:
    readme_path = root / "README.md"
    if not readme_path.exists():
        return {"ok": False, "detail": "README.md missing"}
    readme = readme_path.read_text(encoding="utf-8")
    payloads = re.findall(r"-d '([^']*)'", readme)
    invalid: list[str] = []
    for payload in payloads:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            invalid.append(f"{payload[:80]}: {exc.msg}")
            continue
        if not isinstance(parsed, dict):
            invalid.append(f"{payload[:80]}: not_object")
    if invalid:
        return {"ok": False, "detail": f"invalid={invalid[:5]} count={len(invalid)}"}
    return {"ok": True, "detail": f"json_bodies={len(payloads)}"}


def readme_quickstart_state_path_check(root: Path = PROJECT_ROOT) -> dict:
    readme_path = root / "README.md"
    if not readme_path.exists():
        return {"ok": False, "detail": "README.md missing"}
    readme = readme_path.read_text(encoding="utf-8")
    quickstart_match = re.search(r"## 运行\n\n```bash\n(?P<body>.*?)\n```", readme, re.S)
    if not quickstart_match:
        return {"ok": False, "detail": "quickstart block missing"}
    command_lines = [
        line
        for line in quickstart_match.group("body").splitlines()
        if "python app/main.py" in line
    ]
    unsafe = [line for line in command_lines if "--state /tmp/friend-memory-demo.json" not in line]
    if unsafe:
        return {"ok": False, "detail": f"unsafe={unsafe[:5]} count={len(unsafe)}"}
    return {"ok": True, "detail": f"quickstart_commands={len(command_lines)}"}


def readme_ingest_state_path_check(root: Path = PROJECT_ROOT) -> dict:
    readme_path = root / "README.md"
    if not readme_path.exists():
        return {"ok": False, "detail": "README.md missing"}
    readme = readme_path.read_text(encoding="utf-8")
    ingest_lines = [
        line
        for line in readme.splitlines()
        if "python app/main.py" in line and " ingest " in line
    ]
    unsafe = [
        line
        for line in ingest_lines
        if "--state " not in line and "FRIEND_MEMORY_STATE=" not in line
    ]
    if unsafe:
        return {"ok": False, "detail": f"unsafe={unsafe[:5]} count={len(unsafe)}"}
    return {"ok": True, "detail": f"ingest_examples={len(ingest_lines)}"}


def examples_static_runnability_check(root: Path = PROJECT_ROOT) -> dict:
    examples_dir = root / "examples"
    if not examples_dir.exists():
        return {"ok": False, "detail": "examples directory missing"}
    example_paths = sorted(examples_dir.glob("*.py"))
    if not example_paths:
        return {"ok": False, "detail": "no example scripts found"}

    parse_errors: list[str] = []
    main_guard_count = 0
    for path in example_paths:
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            parse_errors.append(f"{path.name}: {exc.msg}")
            continue
        if '__name__ == "__main__"' in source or "__name__ == '__main__'" in source:
            main_guard_count += 1

    readme_path = root / "README.md"
    missing_readme_examples: list[str] = []
    readme_example_count = 0
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        readme_examples = {
            Path(match).name
            for match in re.findall(r"python (examples/[a-zA-Z0-9_./-]+\.py)", readme)
        }
        readme_example_count = len(readme_examples)
        available = {path.name for path in example_paths}
        missing_readme_examples = sorted(readme_examples - available)

    if parse_errors or missing_readme_examples:
        return {
            "ok": False,
            "detail": f"parse_errors={parse_errors} missing_readme_examples={missing_readme_examples}",
        }
    return {
        "ok": True,
        "detail": f"examples={len(example_paths)} main_guard={main_guard_count} readme_examples={readme_example_count}",
    }


def external_ai_strict_acceptance_surface_check(root: Path = PROJECT_ROOT) -> dict:
    issues: list[str] = []
    try:
        catalog = cli_command_catalog()
        ai_probe = next((item for item in catalog["commands"] if item["name"] == "ai-probe"), None)
        if not ai_probe:
            issues.append("cli_ai_probe_missing")
        elif "--require-external-ai" not in ai_probe["options"]:
            issues.append("cli_require_external_ai_option_missing")
    except Exception as exc:
        issues.append(f"cli_catalog_error:{type(exc).__name__}")

    server_path = root / "app" / "server.py"
    if not server_path.exists():
        issues.append("server_py_missing")
    else:
        server_source = server_path.read_text(encoding="utf-8")
        required_fragments = [
            'parsed.path == "/ai-probe"',
            'payload.get("require_external_ai"',
            '"external_ai_requirement"',
            "HTTPStatus.PRECONDITION_FAILED",
        ]
        missing_fragments = [fragment for fragment in required_fragments if fragment not in server_source]
        issues.extend(f"http_ai_probe_strict_fragment_missing:{fragment}" for fragment in missing_fragments)

    readme_path = root / "README.md"
    if not readme_path.exists():
        issues.append("readme_missing")
    else:
        readme = readme_path.read_text(encoding="utf-8")
        readme_fragments = [
            "--require-external-ai",
            '"require_external_ai":true',
            "412 Precondition Failed",
        ]
        missing_readme = [fragment for fragment in readme_fragments if fragment not in readme]
        issues.extend(f"readme_ai_strict_fragment_missing:{fragment}" for fragment in missing_readme)

    if issues:
        return {"ok": False, "detail": "; ".join(issues[:8])}
    return {"ok": True, "detail": "cli=--require-external-ai http=require_external_ai readme=documented"}


def implementation_acceptance_overview_check(root: Path = PROJECT_ROOT) -> dict:
    overview_path = root / "docs" / "实现验收总览.md"
    if not overview_path.exists():
        return {"ok": False, "detail": "docs/实现验收总览.md missing"}
    text = overview_path.read_text(encoding="utf-8")
    required_fragments = [
        "在当前项目根目录实现系统",
        "使用 Python 和 uv 管理依赖",
        "不做可导入包，做成项目",
        "文档移动到 `docs/`",
        "将朋友类记忆方案做成人类可读 HTML",
        "保证 HTML 信息不丢失",
        "不是纯计算，要有 AI 参与",
        "可验证外部 AI 是否真实参与",
        "uv --cache-dir .uv-cache run python app/main.py doctor --json --strict",
        "uv --cache-dir .uv-cache run python -m unittest discover -s tests",
        "--require-external-ai",
        '"require_external_ai":true',
        "412 Precondition Failed",
        "正式生产效果未被样本替代证明",
        "至少 200 条有效且唯一的样本",
        "每组至少 1000 用户，持续 12 周",
        "至少 30 天窗口和 1000 活跃用户",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        return {"ok": False, "detail": f"missing_fragments={missing[:6]} count={len(missing)}"}
    return {"ok": True, "detail": f"fragments={len(required_fragments)}"}


def project_doctor_report(root: Path = PROJECT_ROOT) -> dict:
    checks: list[dict] = []
    pyproject_path = root / "pyproject.toml"
    pyproject_data: dict = {}
    if pyproject_path.exists():
        try:
            pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            checks.append(_doctor_item("pyproject_toml_parseable", True, str(pyproject_path)))
        except tomllib.TOMLDecodeError as exc:
            checks.append(_doctor_item("pyproject_toml_parseable", False, f"{pyproject_path}: {exc}"))
    else:
        checks.append(_doctor_item("pyproject_toml_parseable", False, f"missing {pyproject_path}"))

    checks.append(
        _doctor_item(
            "uv_project_not_package",
            pyproject_data.get("tool", {}).get("uv", {}).get("package") is False,
            "[tool.uv] package = false",
        )
    )
    checks.append(_doctor_item("no_importable_package_init", not (root / "app" / "__init__.py").exists(), "app/__init__.py absent"))
    checks.append(_doctor_item("no_setup_py", not (root / "setup.py").exists(), "setup.py absent"))

    required_files = [
        "app/main.py",
        "app/system.py",
        "app/models.py",
        "app/ai.py",
        "app/server.py",
        "tests/test_project.py",
        "scripts/build-friend-memory-report.js",
        "docs/[重要]朋友类记忆系统改进方案.md",
        "docs/朋友类记忆系统改进方案.html",
        "docs/实现验收总览.md",
        "data/friend_memory_state.json",
        "README.md",
        "uv.lock",
    ]
    for relative in required_files:
        checks.append(_doctor_item(f"file:{relative}", (root / relative).is_file(), relative))

    docs_dir = root / "docs"
    markdown_docs = sorted(path.name for path in docs_dir.glob("*.md")) if docs_dir.exists() else []
    root_markdown_docs = [path for path in root.glob("*.md") if path.name != "README.md"]
    checks.append(
        _doctor_item(
            "docs_moved_to_docs_dir",
            bool(markdown_docs) and not root_markdown_docs,
            f"docs={len(markdown_docs)} root_non_readme_md={len(root_markdown_docs)}",
        )
    )
    html_source_check = friend_memory_html_source_check(root)
    checks.append(
        _doctor_item(
            "friend_memory_html_matches_markdown_source",
            bool(html_source_check["ok"]),
            html_source_check["detail"],
        )
    )

    state_path = root / "data" / "friend_memory_state.json"
    if state_path.exists():
        try:
            raw = json.loads(state_path.read_text(encoding="utf-8"))
            checks.append(
                _doctor_item(
                    "default_state_json_loads",
                    isinstance(raw.get("relationships"), dict) and isinstance(raw.get("memories"), dict),
                    "default state has relationships and memories",
                )
            )
        except json.JSONDecodeError as exc:
            checks.append(_doctor_item("default_state_json_loads", False, f"{state_path}: {exc}"))
    else:
        checks.append(_doctor_item("default_state_json_loads", False, f"missing {state_path}"))

    try:
        cli_catalog = cli_command_catalog()
        cli_names = {item["name"] for item in cli_catalog["commands"]}
        cli_ok = cli_catalog["schema"] == "friend-memory-cli-commands-v1" and {"doctor", "commands", "ingest"}.issubset(cli_names)
        cli_detail = f"commands={cli_catalog['command_count']}"
    except Exception as exc:
        cli_ok = False
        cli_detail = f"{type(exc).__name__}: {exc}"
    checks.append(_doctor_item("cli_command_catalog_generates", cli_ok, cli_detail))

    http_route_check = http_route_catalog_source_check(root)
    checks.append(
        _doctor_item(
            "http_route_catalog_matches_server_paths",
            bool(http_route_check["ok"]),
            http_route_check["detail"],
        )
    )

    try:
        readme_cli_check = readme_cli_catalog_check(root)
    except Exception as exc:
        readme_cli_check = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    checks.append(
        _doctor_item(
            "readme_cli_examples_match_command_catalog",
            bool(readme_cli_check["ok"]),
            readme_cli_check["detail"],
        )
    )

    try:
        readme_http_check = readme_http_route_catalog_check(root)
    except Exception as exc:
        readme_http_check = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    checks.append(
        _doctor_item(
            "readme_http_examples_match_route_catalog",
            bool(readme_http_check["ok"]),
            readme_http_check["detail"],
        )
    )

    try:
        readme_files_check = readme_local_file_references_check(root)
    except Exception as exc:
        readme_files_check = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    checks.append(
        _doctor_item(
            "readme_local_file_references_exist",
            bool(readme_files_check["ok"]),
            readme_files_check["detail"],
        )
    )

    try:
        readme_examples_check = readme_project_structure_examples_check(root)
    except Exception as exc:
        readme_examples_check = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    checks.append(
        _doctor_item(
            "readme_project_structure_lists_examples",
            bool(readme_examples_check["ok"]),
            readme_examples_check["detail"],
        )
    )

    try:
        readme_docs_check = readme_project_structure_docs_check(root)
    except Exception as exc:
        readme_docs_check = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    checks.append(
        _doctor_item(
            "readme_project_structure_lists_required_docs",
            bool(readme_docs_check["ok"]),
            readme_docs_check["detail"],
        )
    )

    try:
        readme_json_body_check = readme_http_curl_json_bodies_check(root)
    except Exception as exc:
        readme_json_body_check = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    checks.append(
        _doctor_item(
            "readme_http_curl_json_bodies_valid",
            bool(readme_json_body_check["ok"]),
            readme_json_body_check["detail"],
        )
    )

    try:
        readme_quickstart_check = readme_quickstart_state_path_check(root)
    except Exception as exc:
        readme_quickstart_check = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    checks.append(
        _doctor_item(
            "readme_quickstart_uses_temporary_state",
            bool(readme_quickstart_check["ok"]),
            readme_quickstart_check["detail"],
        )
    )

    try:
        readme_ingest_check = readme_ingest_state_path_check(root)
    except Exception as exc:
        readme_ingest_check = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    checks.append(
        _doctor_item(
            "readme_ingest_examples_use_nondefault_state",
            bool(readme_ingest_check["ok"]),
            readme_ingest_check["detail"],
        )
    )

    try:
        examples_check = examples_static_runnability_check(root)
    except Exception as exc:
        examples_check = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    checks.append(
        _doctor_item(
            "examples_static_runnability",
            bool(examples_check["ok"]),
            examples_check["detail"],
        )
    )

    try:
        ai_strict_check = external_ai_strict_acceptance_surface_check(root)
    except Exception as exc:
        ai_strict_check = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    checks.append(
        _doctor_item(
            "external_ai_strict_acceptance_surface",
            bool(ai_strict_check["ok"]),
            ai_strict_check["detail"],
        )
    )

    try:
        acceptance_overview_check = implementation_acceptance_overview_check(root)
    except Exception as exc:
        acceptance_overview_check = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    checks.append(
        _doctor_item(
            "implementation_acceptance_overview_complete",
            bool(acceptance_overview_check["ok"]),
            acceptance_overview_check["detail"],
        )
    )

    commands = [
        "ingest",
        "retrieve",
        "browser",
        "ai-status",
        "ai-probe",
        "ai-observation",
        "evidence-validate",
        "decision-report",
        "release-gate",
        "release-bundle",
        "doctor",
        "commands",
    ]
    checks.append(_doctor_item("cli_surface_declared", True, ",".join(commands), severity="info"))

    failed = [item for item in checks if item["severity"] == "required" and not item["ok"]]
    warnings = [item for item in checks if item["severity"] == "warning" and not item["ok"]]
    return {
        "status": "PASS" if not failed else "FAIL",
        "project_root": str(root),
        "summary": {
            "required_passed": len([item for item in checks if item["severity"] == "required" and item["ok"]]),
            "required_failed": len(failed),
            "warnings": len(warnings),
            "total_checks": len(checks),
        },
        "checks": checks,
        "next_commands": [
            "uv --cache-dir .uv-cache run python -m unittest discover -s tests",
            "node scripts/build-friend-memory-report.js",
            "uv --cache-dir .uv-cache run python app/main.py commands --json",
            "uv --cache-dir .uv-cache run python app/main.py ai-probe --json",
            "uv --cache-dir .uv-cache run python app/main.py ai-observation --output /tmp/friend-memory-evidence/ai-observation.json --json",
        ],
    }


def provenance_template(kind: str) -> dict:
    if kind == "ab_experiment":
        return {
            "experiment_id": "replace-with-experiment-id",
            "start_date": "2026-01-01",
            "end_date": "2026-03-26",
            "analysis_owner": "replace-with-owner",
            "assignment_method": "stable random assignment by user id with holdout cohort",
            "primary_metrics": ["nps", "retention_rate", "avg_session_minutes", "avg_intimacy_delta"],
            "statistical_test": "pre-registered two-sided test with confidence intervals",
            "privacy_redaction": "raw user text removed or anonymized before export",
        }
    if kind == "production_telemetry":
        return {
            "telemetry_id": "replace-with-telemetry-id",
            "collected_at": "2026-04-01T00:00:00+00:00",
            "owner": "replace-with-owner",
            "source_system": "production metrics warehouse",
            "aggregation_window": "30d rolling window",
            "privacy_redaction": "aggregated counts only, no raw user text",
        }
    return {
        "dataset_id": "replace-with-dataset-id",
        "collected_at": "2026-01-01T00:00:00+00:00",
        "owner": "replace-with-owner",
        "review_protocol": "two-pass human review with disagreement resolution",
        "privacy_redaction": "raw identifiers removed or anonymized before export",
    }


def provenance_has_placeholder(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower().startswith("replace-with-")
    if isinstance(value, list):
        return any(provenance_has_placeholder(item) for item in value)
    if isinstance(value, dict):
        return any(provenance_has_placeholder(item) for item in value.values())
    return False


def privacy_redaction_is_unsafe(value: object) -> bool:
    if not isinstance(value, str):
        return True
    normalized = " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())
    unsafe_phrases = {
        "none",
        "no",
        "no redaction",
        "not redacted",
        "unredacted",
        "raw data retained",
        "raw text retained",
        "raw user text retained",
        "raw identifiers retained",
    }
    return normalized in unsafe_phrases or "unredacted" in normalized


def parse_provenance_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def evidence_sample_signature(task: str, item: dict) -> str:
    if task == "stage_detection":
        payload = {"text": item.get("text"), "expected_stage": item.get("expected_stage")}
    elif task == "self_disclosure_capture":
        payload = {"text": item.get("text"), "expected_self_disclosure": item.get("expected_self_disclosure")}
    elif task == "inside_joke_detection":
        payload = {
            "turns": item.get("turns"),
            "expected_detected": item.get("expected_detected"),
            "expected_phrase": item.get("expected_phrase"),
        }
    elif task == "emotional_resonance_retrieval":
        payload = {
            "memories": item.get("memories"),
            "query": item.get("query"),
            "expected_relevant_indices": item.get("expected_relevant_indices"),
        }
    elif task == "story_quality":
        payload = {
            "story_id": item.get("story_id"),
            "title": item.get("title"),
            "score": item.get("score"),
        }
    else:
        payload = item
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def unique_evidence_sample_count(task: str, dataset: dict) -> int:
    examples = dataset.get("examples") if isinstance(dataset, dict) else None
    if not isinstance(examples, list):
        return 0
    signatures = {
        evidence_sample_signature(task, item)
        for item in examples
        if isinstance(item, dict)
    }
    return len(signatures)


def with_evidence_schema(task: str, payload: dict) -> dict:
    return {"schema": EVIDENCE_DATASET_SCHEMA, "task": task, **payload}


def evaluation_template(task: str) -> dict:
    requirement = FORMAL_EVIDENCE_REQUIREMENTS[task]
    provenance = provenance_template(requirement["kind"])
    if task == "stage_detection":
        return with_evidence_schema(task, {
            "config": {
                "required_samples": 200,
                "required_unique_samples": 200,
                "target_accuracy": 0.75,
                "provenance": provenance,
            },
            "examples": [
                {"text": "你不懂我，我很失望。", "expected_stage": "DIFFERENTIATING"},
                {"text": "第一次一起庆祝成功，太开心了！", "expected_stage": "INTENSIFYING"},
            ],
        })
    if task == "self_disclosure_capture":
        return with_evidence_schema(task, {
            "config": {
                "required_samples": 200,
                "required_unique_samples": 200,
                "target_recall": 0.90,
                "provenance": provenance,
            },
            "examples": [
                {"text": "其实我从来没告诉别人，我那天哭了很久。", "expected_self_disclosure": True},
                {"text": "今天只是普通聊项目进展。", "expected_self_disclosure": False},
            ],
        })
    if task == "inside_joke_detection":
        return with_evidence_schema(task, {
            "config": {
                "required_samples": 200,
                "required_unique_samples": 200,
                "target_accuracy": 0.70,
                "provenance": provenance,
            },
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
        })
    if task == "emotional_resonance_retrieval":
        return with_evidence_schema(task, {
            "config": {
                "required_samples": 200,
                "required_unique_samples": 200,
                "target_p5": 0.65,
                "provenance": provenance,
            },
            "examples": [
                {
                    "memories": ["我那天崩溃哭了很久，压力特别大。", "第一次一起庆祝项目成功，真的开心。"],
                    "query": "我今天也崩溃难过，压力很大。",
                    "expected_relevant_indices": [0],
                }
            ],
        })
    if task == "story_quality":
        return with_evidence_schema(task, {
            "config": {
                "required_samples": 200,
                "required_unique_samples": 200,
                "target_average_score": 4.0,
                "provenance": provenance,
            },
            "examples": [
                {
                    "story_id": "story_xxx",
                    "title": "样例暗号A",
                    "score": 4.5,
                    "note": "来源清晰，叙事无明显编造",
                }
            ],
        })
    if task == "friend_mode_ab":
        return with_evidence_schema(task, {
            "config": {"duration_weeks": 12, "provenance": provenance},
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
        })
    if task == "production_telemetry":
        return with_evidence_schema(task, {
            "config": {
                "duration_days": 30,
                "target_active_complaint_rate": 0.05,
                "target_hard_delete_success_rate": 0.99,
                "target_transparency_ack_rate": 0.30,
                "target_crisis_review_rate": 0.95,
                "provenance": provenance,
            },
            "metrics": {
                "active_users": 1000,
                "active_complaint_rate": 0.04,
                "hard_delete_success_rate": 1.0,
                "transparency_ack_rate": 0.35,
                "crisis_escalation_review_rate": 1.0,
            },
        })
    raise ValueError(f"unknown evaluation task: {task}")


def load_project() -> FriendMemoryProject:
    if STATE_PATH.exists():
        return FriendMemoryProject.load(STATE_PATH)
    return FriendMemoryProject()


def save_project(project: FriendMemoryProject) -> None:
    project.save(STATE_PATH)


def cmd_ingest(args: argparse.Namespace) -> None:
    project = load_project()
    result = project.ingest_turn(args.user, args.ai, args.text, timestamp=datetime.now(timezone.utc))
    save_project(project)
    print(f"relationship={result.relationship_id}")
    if result.memory_paused:
        print(f"memory=paused stage={result.stage.value}")
        return
    print(f"memory={result.memory_id} score={result.score:.3f} stage={result.stage.value}")
    ai_decision = next(
        (
            item
            for item in reversed(project.ai_decision_log)
            if item.get("relationship_id") == result.relationship_id and item.get("task") == "analyze_turn"
        ),
        None,
    )
    if result.memory_id and ai_decision:
        summary = project.ai_decision_summary(ai_decision)
        fallback = " fallback" if summary["fallback_used"] else ""
        sanitized = " sanitized" if summary["sanitized"] else ""
        print(
            f"ai[{summary['used_provider']}:{summary['used_participation_kind']}:{summary['task']}"
            f"{fallback}{sanitized}]: {summary['reason'] or 'n/a'}"
        )
    if result.emotional_memory_id:
        print(f"emotional_memory={result.emotional_memory_id}")
    for event in result.active_events:
        print(f"active[{event['active_id']}]: {event['reason']}")


def cmd_ingest_exchange(args: argparse.Namespace) -> None:
    project = load_project()
    result = project.ingest_exchange(
        args.user,
        args.ai,
        args.user_text,
        args.assistant_text,
        timestamp=datetime.now(timezone.utc),
    )
    save_project(project)
    ai_decision = next(
        (
            item
            for item in reversed(project.ai_decision_log)
            if item.get("relationship_id") == result.relationship_id and item.get("task") == "analyze_turn"
        ),
        None,
    )
    payload = {
        "relationship_id": result.relationship_id,
        "memory_id": result.memory_id,
        "emotional_memory_id": result.emotional_memory_id,
        "score": result.score,
        "stage": result.stage.value,
        "active_suggestions": result.active_suggestions,
        "active_events": result.active_events,
        "memory_paused": result.memory_paused,
        "ai_decision_summary": project.ai_decision_summary(ai_decision),
        "ai_decision": ai_decision,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(f"relationship={result.relationship_id}")
    if result.memory_paused:
        print(f"memory=paused stage={result.stage.value}")
        return
    print(f"memory={result.memory_id} score={result.score:.3f} stage={result.stage.value}")
    summary = payload["ai_decision_summary"]
    fallback = " fallback" if summary["fallback_used"] else ""
    sanitized = " sanitized" if summary["sanitized"] else ""
    print(
        f"ai[{summary['used_provider']}:{summary['used_participation_kind']}:{summary['task']}"
        f"{fallback}{sanitized}]: {summary['reason'] or 'n/a'}"
    )


def cmd_retrieve(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        if args.json:
            print(json.dumps({"error": "relationship not found", "relationship_id": relationship_id}, ensure_ascii=False, indent=2))
            return
        print("relationship not found")
        return
    results = project.retrieve(relationship_id, args.query, limit=args.limit, include_archived=args.include_archived)
    query_meta = (
        results[0].explanation.get("query_metacognition")
        if results
        else project.retrieval_audit_log[-1].get("query_metacognition")
        if project.retrieval_audit_log
        else None
    )
    association_expansions = results[0].explanation.get("association_expansions", []) if results else []
    if args.json:
        payload = {
            "relationship_id": relationship_id,
            "query": args.query,
            "results": [
                {
                    "memory_id": item.memory.memory_id,
                    "type": item.memory.memory_type.value,
                    "score": item.score,
                    "content": item.explanation.get("display_content", item.memory.content),
                    "raw_content_available": item.explanation.get("trust_presentation", {}).get("original_preserved", True),
                    "time": item.presentation_time,
                    "explanation": item.explanation,
                }
                for item in results
            ],
            "query_metacognition": query_meta,
            "association_expansions": association_expansions,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        save_project(project)
        return
    for item in results:
        phrase = item.presentation_time["phrase"]
        relationship_phrase = item.presentation_time["relationship_since_phrase"]
        content = item.explanation.get("display_content", item.memory.content)
        print(f"{item.score:.3f} [{item.memory.memory_type.value}] {phrase} / {relationship_phrase}: {content}")
        if args.explain:
            weights = item.explanation["weights"]
            print(
                "  explain "
                f"semantic={item.explanation['semantic']:.3f} "
                f"emotional={item.explanation['emotional_resonance']:.3f} "
                f"relation={item.explanation['relationship_relevance']:.3f} "
                f"precious={item.explanation['time_preciousness']:.3f} "
                f"weight={item.explanation['memory_weight']:.3f} "
                f"trust_bias={item.explanation['trust_bias_applied']}"
            )
            print(f"  presentation={item.explanation['trust_presentation']['mode']}")
            meta = item.explanation["metacognition"]
            print(
                "  metacognition "
                f"confidence={meta['confidence']:.3f} "
                f"action={meta['uncertainty_action']} "
                f"verified={meta['human_verified']}"
            )
            print(
                "  weights "
                f"semantic={weights['semantic']:.3f} "
                f"emotional={weights['emotional_resonance']:.3f} "
                f"relation={weights['relationship_relevance']:.3f} "
                f"precious={weights['time_preciousness']:.3f} "
                f"reasons={','.join(weights['reasons'])}"
            )
    if args.explain and query_meta:
        print(
            "query_metacognition "
            f"coverage={query_meta['coverage']:.3f} "
            f"confidence={query_meta['confidence']:.3f} "
            f"action={query_meta['action']} "
            f"reason={query_meta['reason']}"
        )
    if args.explain and results:
        for expansion in association_expansions:
            print(
                "association "
                f"type={expansion['type']} "
                f"confidence={expansion['confidence']:.3f} "
                f"text={expansion['text']}"
            )
    save_project(project)


def cmd_status(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    relationship = project.relationships.get(relationship_id)
    if not relationship:
        print("relationship not found")
        return
    print(f"relationship={relationship.relationship_id}")
    print(f"stage={relationship.stage.value} strength={relationship.strength:.3f} trust={relationship.trust_level:.3f}")
    print(f"interactions={relationship.interaction_count} age_days={relationship.relationship_age}")
    print(f"shared={len(relationship.shared_episodes)} milestones={len(relationship.milestones)} unresolved={len(relationship.unresolved_threads)}")
    print(f"stories={len([s for s in project.story_nodes.values() if s.relationship_id == relationship_id])}")
    for suggestion in project.mode_suggestions(relationship_id):
        print(f"mode_suggestion={suggestion['recommended_mode']} inactive_days={suggestion['inactive_days']} reason={suggestion['reason']}")


def cmd_reset(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        if args.json:
            print(json.dumps({"error": "relationship not found", "relationship_id": relationship_id}, ensure_ascii=False, indent=2))
            return
        print("relationship not found")
        return
    request = project.request_reset(relationship_id, ResetMode(args.mode))
    save_project(project)
    if args.json:
        print(
            json.dumps(
                {
                    "request_id": request.request_id,
                    "relationship_id": request.relationship_id,
                    "mode": request.mode.value,
                    "status": request.status.value,
                    "requested_at": request.requested_at.isoformat(),
                    "execute_after": request.execute_after.isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"reset_request={request.request_id} mode={request.mode.value} execute_after={request.execute_after.isoformat()}")


def cmd_reset_request(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        if args.json:
            print(json.dumps({"error": "relationship not found", "relationship_id": relationship_id}, ensure_ascii=False, indent=2))
            return
        print("relationship not found")
        return
    request = project.request_reset(relationship_id, ResetMode(args.mode))
    save_project(project)
    if args.json:
        print(
            json.dumps(
                {
                    "request_id": request.request_id,
                    "relationship_id": request.relationship_id,
                    "mode": request.mode.value,
                    "status": request.status.value,
                    "requested_at": request.requested_at.isoformat(),
                    "execute_after": request.execute_after.isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"reset_request={request.request_id} mode={request.mode.value} execute_after={request.execute_after.isoformat()}")


def cmd_reset_confirm(args: argparse.Namespace) -> None:
    project = load_project()
    ok = project.confirm_reset(args.request_id, force=args.force)
    support = project.relationship_ending_support(args.request_id)
    save_project(project)
    if args.json:
        latest_support = support[-1] if support else None
        print(
            json.dumps(
                {
                    "request_id": args.request_id,
                    "confirmed": ok,
                    "support_generated": bool(latest_support),
                    "relationship_ending_support": latest_support,
                    "cooldown_until": latest_support.get("cooldown_until") if latest_support else None,
                    "restart_policy": latest_support.get("restart_policy") if latest_support else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"confirmed={ok}")
    if support:
        print(f"relationship_ending_support={support[-1]['request_id']}")
        print(f"cooldown_until={support[-1].get('cooldown_until')}")
        restart_policy = support[-1].get("restart_policy", {})
        print(f"old_memory_recovery_allowed={restart_policy.get('old_memory_recovery_allowed')}")
        print(support[-1]["message"])
        for item in support[-1].get("soft_landing_plan", [])[:3]:
            print(f"- {item['step']}: {item['label']}")


def cmd_reset_cancel(args: argparse.Namespace) -> None:
    project = load_project()
    project.cancel_reset(args.request_id)
    save_project(project)
    if args.json:
        request = project.reset_requests[args.request_id]
        print(
            json.dumps(
                {
                    "request_id": args.request_id,
                    "relationship_id": request.relationship_id,
                    "mode": request.mode.value,
                    "status": request.status.value,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"cancelled={args.request_id}")


def cmd_browser(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        if args.json:
            print(json.dumps({"error": "relationship not found", "relationship_id": relationship_id}, ensure_ascii=False, indent=2))
            return
        print("relationship not found")
        return
    snapshot = project.browser_snapshot(relationship_id)
    if args.json:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        return
    dashboard = snapshot["dashboard"]
    counts = snapshot["counts"]
    print(f"relationship={dashboard['relationship_id']}")
    print(
        f"stage={dashboard['stage']} mode={dashboard['mode']} "
        f"strength={dashboard['strength']:.3f} trust={dashboard['trust_level']:.3f} intimacy={dashboard['intimacy_level']:.3f}"
    )
    print(
        f"memories={counts['memories']} emotional={counts['emotional_memories']} "
        f"stories={counts['stories']} milestones={counts['milestones']} unresolved={counts['unresolved_threads']}"
    )
    for story in snapshot["stories"][: args.limit]:
        print(f"story[{story['level']}] {story['title']} events={story['events']} themes={','.join(story['themes'])}")
    for item in snapshot["active_behavior_log"][-args.limit :]:
        print(f"active_log {item['at']}: {item['reason']}")


def cmd_export(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    anonymized = args.anonymize or args.format == "anonymous-json" or project.export_requires_anonymization(relationship_id)
    if args.format not in {"json", "anonymous-json"} and relationship_id not in project.relationships:
        print("relationship not found")
        return
    exported = project.generate_export(
        relationship_id=relationship_id if relationship_id in project.relationships else None,
        export_format=args.format,
        anonymized=anonymized,
        destination=args.output or "stdout",
        purpose=args.purpose,
    )
    payload = exported if isinstance(exported, str) else json.dumps(exported, ensure_ascii=False, indent=2)
    save_project(project)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload, encoding="utf-8")
        print(args.output)
    else:
        print(payload)


def cmd_mode(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    custom_profile = json.loads(args.profile_json) if args.profile_json else None
    event = project.set_mode(relationship_id, Mode(args.mode), custom_profile=custom_profile, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
        return
    print(f"mode={args.mode}")
    if custom_profile is not None:
        print(json.dumps(project.relationships[relationship_id].preferences.custom_profile, ensure_ascii=False))


def cmd_pref(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    event = project.set_preference(relationship_id, args.key, args.value, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"{args.key}={event['new_value']}")


def cmd_decay_curve(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    event = project.set_decay_curve_type(relationship_id, args.curve, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"decay_curve_type={event['to']}")


def cmd_custom_profile(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    event = project.update_custom_mode_profile(
        relationship_id,
        json.loads(args.profile_json),
        reason=args.reason,
    )
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"custom_profile_updated applied={event['applied']}")
        print(json.dumps(event["profile"], ensure_ascii=False))


def cmd_active_feedback(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    result = project.record_active_feedback(relationship_id, args.active_id, args.reaction)
    save_project(project)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"active_feedback={result['active_id']} reaction={result['reaction']}")
        if result["adjustment"]:
            print(json.dumps(result["adjustment"], ensure_ascii=False))


def cmd_active_type_mute(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    event = project.mute_active_type(relationship_id, args.active_type, days=args.days, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"active_type_muted={event['active_type']} until={event['until']}")


def cmd_active_type_unmute(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    event = project.unmute_active_type(relationship_id, args.active_type, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"active_type_unmuted={event['active_type']}")


def cmd_implicit_topic_feedback(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    result = project.record_implicit_topic_feedback(relationship_id, args.topic_id, args.reaction)
    save_project(project)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"implicit_topic_feedback={args.topic_id} reaction={args.reaction}")


def cmd_transparency(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    if args.ack:
        project.acknowledge_transparency(relationship_id)
        save_project(project)
    panel = project.transparency_panel(relationship_id)
    if args.json:
        print(json.dumps(panel, ensure_ascii=False, indent=2))
        return
    print(panel["statement"])
    print(f"acknowledged_at={panel['acknowledged_at'] or 'None'}")
    print("Mandatory disclosures:")
    for item in panel["mandatory_disclosures"]:
        print(f"- {item['title']}: {item['text']}")
    print("AI participation:")
    for item in panel["ai_participation"]:
        print(f"- {item}")
    print("Controls:")
    for key, value in panel["user_controls"].items():
        print(f"- {key}={value}")


def cmd_ai_status(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = None if args.all else f"{args.user}:{args.ai}"
    status = project.ai_status(relationship_id)
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    print(f"provider={status['provider']}")
    print(status["runtime_note"])
    print(f"readiness={status['readiness_status']} label={status['readiness_label']}")
    print(f"external_ai_configured={status['external_ai_configured']}")
    print(f"external_ai_used_recently={status['external_ai_used_recently']}")
    readiness = status["readiness"]
    print(
        f"decision_count={status['decision_count']} "
        f"fallback_event_count={status['fallback_event_count']} "
        f"external_success_count={readiness['external_success_count']}"
    )
    print(f"tasks={','.join(status['tasks']) if status['tasks'] else 'none'}")
    for summary in status["recent_summaries"][-3:]:
        fallback = " fallback" if summary["fallback_used"] else ""
        sanitized = " sanitized" if summary["sanitized"] else ""
        print(
            f"recent_ai[{summary['used_provider']}:{summary['used_participation_kind']}:{summary['task']}"
            f"{fallback}{sanitized}]: {summary['reason'] or 'n/a'}"
        )
    configuration = status["configuration"]
    if configuration.get("primary"):
        primary = configuration["primary"]
        print(f"primary={primary['provider']} endpoint={primary.get('endpoint', 'n/a')}")
        print(f"fallback={configuration['fallback']['provider']}")
    else:
        print(f"configured={configuration['provider']}")


def cmd_ai_probe(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    probe = project.probe_ai(relationship_id, args.text)
    external = probe["external_ai_participation"]
    require_external = bool(getattr(args, "require_external_ai", False))
    requirement_met = bool(external["external_ai_participated"])
    probe["external_ai_requirement"] = {
        "required": require_external,
        "met": requirement_met,
        "verdict": external["verdict"],
        "explanation": external["explanation"],
    }
    if args.json:
        print(json.dumps(probe, ensure_ascii=False, indent=2))
        if require_external and not requirement_met:
            raise SystemExit(1)
        return
    print(f"ok={probe['ok']}")
    print(f"provider={probe['provider']} participation_kind={probe['participation_kind']}")
    print(f"used_provider={probe['used_provider']} used_participation_kind={probe['used_participation_kind']}")
    print(f"fallback_used={probe['fallback_used']}")
    print(
        f"external_ai_participated={external['external_ai_participated']} "
        f"verdict={external['verdict']}"
    )
    print(f"external_ai_explanation={external['explanation']}")
    if require_external:
        print(f"external_ai_requirement_met={requirement_met}")
    print(f"writes_memory={probe['writes_memory']} appends_ai_decision_log={probe['appends_ai_decision_log']}")
    if probe.get("error"):
        print(f"error={probe['error_type']}: {probe['error']}")
    else:
        output = probe.get("sanitized_output", {})
        print(
            "analysis="
            f"importance={output.get('importance')} "
            f"memory_type={output.get('memory_type')} "
            f"context_tag={output.get('context_tag')} "
            f"reason={output.get('reason') or 'n/a'}"
        )
    if require_external and not requirement_met:
        raise SystemExit(1)


def ai_observation_report_for_project(project: FriendMemoryProject, relationship_id: str | None = None) -> dict:
    status = project.ai_status(relationship_id)
    readiness = status["readiness"]
    observed = readiness["status"] == "external_observed"
    return {
        "schema": "friend-memory-ai-observation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "relationship_id": relationship_id,
        "formal_ready": observed,
        "issues": [] if observed else [f"external_ai_not_observed:{readiness['status']}"],
        "readiness_status": readiness["status"],
        "readiness_label": readiness["label"],
        "external_success_count": readiness["external_success_count"],
        "fallback_event_count": readiness["fallback_event_count"],
        "tasks": readiness["tasks"],
        "configuration": readiness["configuration"],
        "recent_summaries": readiness["recent_summaries"],
        "note": (
            "formal_ready requires at least one recent AI decision whose used_participation_kind "
            "is external_http_worker or external_model."
        ),
    }


def validate_ai_observation(evidence_dir: Path, relationship_id: str | None = None, *, now: datetime | None = None) -> dict:
    path = evidence_dir / AI_OBSERVATION_FILENAME
    result = {
        "filename": AI_OBSERVATION_FILENAME,
        "path": str(path),
        "exists": path.exists(),
        "formal_ready": False,
        "issues": [],
        "observation": None,
    }
    if not path.exists():
        result["issues"].append("missing_file")
        return result
    try:
        observation = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["issues"].append(f"invalid_json:{exc.msg}")
        return result
    result["observation"] = observation
    if observation.get("schema") != "friend-memory-ai-observation-v1":
        result["issues"].append("schema_mismatch")
    if observation.get("formal_ready") is not True:
        result["issues"].append("formal_ready_not_true")
    if relationship_id and observation.get("relationship_id") not in {None, relationship_id}:
        result["issues"].append("relationship_id_mismatch")
    generated_at = parse_provenance_datetime(observation.get("generated_at"))
    if generated_at is None:
        result["issues"].append("generated_at_invalid_or_missing")
    else:
        reference = now or datetime.now(timezone.utc)
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        if generated_at > reference + timedelta(minutes=AI_OBSERVATION_FUTURE_SKEW_MINUTES):
            result["issues"].append("generated_at_in_future")
        age = reference - generated_at
        if age > timedelta(days=AI_OBSERVATION_MAX_AGE_DAYS):
            result["issues"].append(f"generated_at_stale:{age.days}>{AI_OBSERVATION_MAX_AGE_DAYS}d")
    summaries = observation.get("recent_summaries") or []
    external_summaries = [
        item
        for item in summaries
        if isinstance(item, dict)
        and item.get("used_participation_kind") in EXTERNAL_AI_PARTICIPATION_KINDS
        and not item.get("fallback_used")
    ]
    external_event_times: list[datetime] = []
    invalid_external_event_times = 0
    reference_for_events = now or datetime.now(timezone.utc)
    if reference_for_events.tzinfo is None:
        reference_for_events = reference_for_events.replace(tzinfo=timezone.utc)
    for summary in external_summaries:
        event_at = parse_provenance_datetime(summary.get("at"))
        if event_at is None:
            invalid_external_event_times += 1
            continue
        if event_at.tzinfo is None:
            event_at = event_at.replace(tzinfo=timezone.utc)
        external_event_times.append(event_at)
        if event_at > reference_for_events + timedelta(minutes=AI_OBSERVATION_FUTURE_SKEW_MINUTES):
            result["issues"].append("external_summary_at_in_future")
    if observation.get("readiness_status") != "external_observed":
        result["issues"].append(f"readiness_status:{observation.get('readiness_status')}")
    if int(observation.get("external_success_count", 0) or 0) < 1:
        result["issues"].append("external_success_count_below_min:0<1")
    if not external_summaries:
        result["issues"].append("external_summary_missing")
    elif not external_event_times:
        result["issues"].append("external_summary_at_missing_or_invalid")
    else:
        latest_event_age = reference_for_events - max(external_event_times)
        if latest_event_age > timedelta(days=AI_OBSERVATION_MAX_AGE_DAYS):
            result["issues"].append(f"external_summary_at_stale:{latest_event_age.days}>{AI_OBSERVATION_MAX_AGE_DAYS}d")
    if invalid_external_event_times:
        result["issues"].append(f"external_summary_at_invalid_count:{invalid_external_event_times}")
    result["formal_ready"] = not result["issues"]
    return result


def cmd_ai_observation(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = args.relationship_id or (f"{args.user}:{args.ai}" if args.relationship else None)
    report = ai_observation_report_for_project(project, relationship_id)
    if args.output:
        report["output_path"] = str(Path(args.output))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        if args.json:
            print(text)
            return
        print(f"wrote={args.output}")
        return
    print(text)


def cmd_memory_writes(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    event = project.set_memory_writes(relationship_id, args.enabled, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"memory_writes_enabled={event['enabled']} reason={event['reason']}")


def cmd_health(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    project.evaluate_health(relationship_id)
    alerts = [item for item in project.health_alerts.values() if item.relationship_id == relationship_id]
    save_project(project)
    payload = [
        {
            "alert_id": item.alert_id,
            "risk_type": item.risk_type,
            "level": item.level.value,
            "message": item.message,
            "acknowledged": item.acknowledged,
            "acknowledged_at": item.acknowledged_at.isoformat() if item.acknowledged_at else None,
            "acknowledgement_note": item.acknowledgement_note,
            "resources": item.resources,
        }
        for item in alerts
    ]
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif not payload:
        print("health=ok")
    else:
        for item in payload:
            print(f"{item['level']} {item['risk_type']} {item['alert_id']}: {item['message']}")
            for resource in item.get("resources", []):
                contact = resource.get("phone") or resource.get("chat_url") or resource.get("label")
                print(f"  resource[{resource.get('type')}:{resource.get('region')}]: {contact}")


def cmd_guardian_summary(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    period_start = datetime.fromisoformat(args.start) if args.start else None
    period_end = datetime.fromisoformat(args.end) if args.end else None
    try:
        summary = project.generate_guardian_summary(relationship_id, period_start=period_start, period_end=period_end)
    except (PermissionError, ValueError) as exc:
        save_project(project)
        if args.json:
            print(json.dumps({"error": str(exc), "relationship_id": relationship_id}, ensure_ascii=False, indent=2))
        else:
            print(f"guardian_summary_error={exc}")
        return
    save_project(project)
    payload = {
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
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"guardian_summary={summary.summary_id} stage={summary.stage.value} minutes={summary.total_minutes}")
        print(f"memory_text_included={summary.privacy_boundary.get('memory_text_included')}")
        print(summary.recommendation)
        for resource in summary.resource_summary:
            contact = resource.get("phone") or resource.get("chat_url") or resource.get("label")
            print(f"resource[{resource.get('type')}:{resource.get('region')}]: {contact}")


def cmd_consolidate(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        if args.json:
            print(json.dumps({"error": "relationship not found", "relationship_id": relationship_id}, ensure_ascii=False, indent=2))
            return
        print("relationship not found")
        return
    report = project.consolidate_relationship(relationship_id)
    save_project(project)
    payload = {
        "relationship_id": report.relationship_id,
        "replayed_memories": report.replayed_memories,
        "upgraded_stories": report.upgraded_stories,
        "downgraded_memories": report.downgraded_memories,
        "archived_memories": report.archived_memories,
        "compressed_stories": report.compressed_stories,
        "health_alerts": report.health_alerts,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"consolidated={relationship_id} replayed={report.replayed_memories} "
            f"upgraded_stories={report.upgraded_stories} downgraded={len(report.downgraded_memories)} "
            f"archived={len(report.archived_memories)}"
        )


def cmd_audit(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = args.relationship_id or (f"{args.user}:{args.ai}" if args.relationship else None)
    report = project.audit_report(relationship_id)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        metrics = report["metrics"]
        print(f"audit_status={report['status']} scope={report['scope']}")
        print(
            f"relationships={metrics['relationships']} memories={metrics['memories']} "
            f"ai_decisions={metrics['ai_decision_events']} orphan_refs={len(report['integrity']['orphan_references'])}"
        )
        complaint_rate = metrics.get("active_complaint_rate")
        if complaint_rate is not None:
            print(
                f"active_acceptance_rate={metrics.get('active_acceptance_rate')} "
                f"active_complaint_rate={complaint_rate:.3f}"
            )
        if "trust_bias_monthly_audit_ready" in metrics:
            print(
                f"trust_bias_monthly_audit_ready={metrics['trust_bias_monthly_audit_ready']} "
                f"trust_bias_adjusted_samples={metrics.get('trust_bias_adjusted_samples', 0)} "
                f"trust_bias_critical_exemptions={metrics.get('trust_bias_critical_exemption_samples', 0)}"
            )
        if report["integrity"]["warnings"]:
            print(f"warnings={len(report['integrity']['warnings'])}")


def cmd_decision_report(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = args.relationship_id or (f"{args.user}:{args.ai}" if args.relationship else None)
    report = decision_report_with_evidence(
        project,
        relationship_id=relationship_id,
        run_benchmarks=args.run_benchmarks,
        benchmark_iterations=args.benchmark_iterations,
        evidence_dir=Path(args.evidence_dir) if args.evidence_dir else None,
        evaluation_files=[Path(item) for item in args.evaluation_file] if args.evaluation_file else None,
        evaluation_tasks=list(args.evaluation_task or []),
        manifest_path=Path(args.manifest) if args.manifest else None,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"decision_report scope={report['scope']} completion_claim={report['summary']['completion_claim']}")
        for phase_id, decision in report["summary"]["phase_decisions"].items():
            print(f"{phase_id}={decision}")
        print(f"status_counts={json.dumps(report['summary']['status_counts'], ensure_ascii=False)}")


def decision_report_with_evidence(
    project: FriendMemoryProject,
    *,
    relationship_id: str | None = None,
    run_benchmarks: bool = False,
    benchmark_iterations: int = 20,
    evidence_dir: Path | None = None,
    evaluation_files: list[Path] | None = None,
    evaluation_tasks: list[str] | None = None,
    manifest_path: Path | None = None,
) -> dict:
    evaluation_results = collect_evaluation_results(
        project,
        evidence_dir=evidence_dir,
        evaluation_files=evaluation_files,
        evaluation_tasks=evaluation_tasks or [],
    )
    report = project.decision_report(
        relationship_id,
        run_benchmarks=run_benchmarks,
        benchmark_iterations=benchmark_iterations,
        evaluation_results=evaluation_results,
    )
    formal_validation = None
    if evidence_dir:
        formal_validation = validate_evidence_dir_for_project(project, evidence_dir, manifest_path=manifest_path)
        manifest_verification = formal_validation.get("manifest_verification")
        report["evidence_inputs"] = {
            "evidence_dir": str(evidence_dir),
            "tasks_loaded": sorted(evaluation_results or []),
            "formal_ready": formal_validation["formal_ready"],
        }
        report["formal_evidence_validation"] = {
            "status": formal_validation["status"],
            "formal_ready": formal_validation["formal_ready"],
            "ready_count": formal_validation["ready_count"],
            "required_count": formal_validation["required_count"],
            "missing_count": formal_validation["missing_count"],
            "manifest_verified": (
                None if manifest_verification is None else bool(manifest_verification.get("verified"))
            ),
            "manifest_issues": [] if manifest_verification is None else manifest_verification.get("issues", []),
            "items": [
                {
                    "task": item["task"],
                    "formal_ready": item["formal_ready"],
                    "issues": item["issues"],
                    "sample_count": (item.get("evaluation") or {}).get("sample_count"),
                    "unique_sample_count": item.get("unique_sample_count"),
                }
                for item in formal_validation["items"]
            ],
            "note": formal_validation["note"],
        }
        report["summary"]["formal_evidence_claim"] = "formal_ready" if formal_validation["formal_ready"] else "not_formal"
        if not formal_validation["formal_ready"] and report["summary"]["completion_claim"] == "locally_proven":
            report["summary"]["completion_claim"] = "not_proven_formal_evidence"
            report["summary"]["formal_evidence_note"] = (
                "All local decision criteria may have passed, but evidence_dir did not pass formal "
                "scale, uniqueness, provenance, and methodology validation."
            )
    if evidence_dir and manifest_path:
        report["evidence_integrity"] = (
            formal_validation["manifest_verification"] if formal_validation else verify_evidence_manifest(evidence_dir, manifest_path)
        )
    return report


def collect_evaluation_results(
    project: FriendMemoryProject,
    *,
    evidence_dir: Path | None = None,
    evaluation_files: list[Path] | None = None,
    evaluation_tasks: list[str] | None = None,
) -> dict[str, dict] | None:
    evaluation_pairs: list[tuple[Path, str]] = []
    if evidence_dir:
        for task, filename in EVIDENCE_FILENAMES.items():
            path = evidence_dir / filename
            if path.exists():
                evaluation_pairs.append((path, task))
    if evaluation_files:
        tasks = list(evaluation_tasks or [])
        if not tasks:
            tasks = ["stage_detection"] * len(evaluation_files)
        if len(tasks) == 1 and len(evaluation_files) > 1:
            tasks = tasks * len(evaluation_files)
        if len(tasks) != len(evaluation_files):
            raise SystemExit("--evaluation-task must be provided once per --evaluation-file, or once to reuse for all files")
        evaluation_pairs.extend(zip(evaluation_files, tasks))
    evaluation_results = {}
    for path, task in evaluation_pairs:
        dataset = json.loads(path.read_text(encoding="utf-8"))
        evaluation = project.evaluate_labeled_dataset(dataset, task=task)
        warnings = evidence_input_warnings(dataset, task)
        if warnings:
            evaluation["input_warnings"] = warnings
            evaluation["input_file"] = str(path)
        evaluation_results[evaluation["task"]] = evaluation
    return evaluation_results or None


def cmd_evaluate_labels(args: argparse.Namespace) -> None:
    project = load_project()
    dataset = json.loads(Path(args.file).read_text(encoding="utf-8"))
    result = project.evaluate_labeled_dataset(dataset, task=args.task)
    warnings = evidence_input_warnings(dataset, args.task)
    if warnings:
        result["input_warnings"] = warnings
        result["input_file"] = str(Path(args.file))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(evaluation_text_summary(result))


def evidence_input_warnings(dataset: object, expected_task: str) -> list[str]:
    if not isinstance(dataset, dict):
        return ["evidence_schema_invalid:expected_object"]
    warnings: list[str] = []
    schema = dataset.get("schema")
    declared_task = dataset.get("task")
    if schema is not None and schema != EVIDENCE_DATASET_SCHEMA:
        warnings.append(f"evidence_schema_mismatch:{schema}")
    if declared_task is not None and declared_task != expected_task:
        warnings.append(f"evidence_task_mismatch:{declared_task}!={expected_task}")
    return warnings


def evaluation_text_summary(result: dict) -> str:
    parts = [f"task={result.get('task')}", f"status={result.get('status')}"]
    if result.get("input_warnings"):
        parts.append(f"input_warnings={','.join(result['input_warnings'])}")
    for key in (
        "sample_count",
        "accuracy",
        "recall",
        "average_p_at_5",
        "average_score",
        "control_users",
        "friend_users",
        "duration_weeks",
        "active_users",
        "duration_days",
    ):
        if key in result:
            parts.append(f"{key}={result[key]}")
    return " ".join(parts)


def formal_evidence_item(project: FriendMemoryProject, evidence_dir: Path, task: str) -> dict:
    path = evidence_dir / EVIDENCE_FILENAMES[task]
    requirement = FORMAL_EVIDENCE_REQUIREMENTS[task]
    item = {
        "task": task,
        "filename": EVIDENCE_FILENAMES[task],
        "path": str(path),
        "required": requirement,
        "exists": path.exists(),
        "formal_ready": False,
        "issues": [],
        "evaluation": None,
    }
    if not path.exists():
        item["issues"].append("missing_file")
        return item
    try:
        dataset = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        item["issues"].append(f"invalid_json:{exc.msg}")
        return item
    if not isinstance(dataset, dict):
        item["issues"].append("evidence_schema_invalid:expected_object")
    else:
        if dataset.get("schema") != EVIDENCE_DATASET_SCHEMA:
            item["issues"].append(f"evidence_schema_mismatch:{dataset.get('schema')}")
        item["issues"].extend([issue for issue in evidence_input_warnings(dataset, task) if issue.startswith("evidence_task_mismatch")])
    evaluation = project.evaluate_labeled_dataset(dataset, task=task)
    item["evaluation"] = evaluation
    provenance = (dataset.get("config") or {}).get("provenance") if isinstance(dataset, dict) else None
    if not isinstance(provenance, dict):
        provenance = {}
    item["provenance"] = provenance
    for field in FORMAL_PROVENANCE_REQUIREMENTS[requirement["kind"]]:
        value = provenance.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            item["issues"].append(f"provenance_missing:{field}")
        elif field == "primary_metrics" and (not isinstance(value, list) or not value):
            item["issues"].append("provenance_invalid:primary_metrics")
        elif provenance_has_placeholder(value):
            item["issues"].append(f"provenance_placeholder:{field}")
        elif field == "privacy_redaction" and privacy_redaction_is_unsafe(value):
            item["issues"].append("provenance_unsafe:privacy_redaction")
    if requirement["kind"] in {"labelled_dataset", "human_review_sample", "production_telemetry"} and provenance.get("collected_at"):
        if parse_provenance_datetime(provenance.get("collected_at")) is None:
            item["issues"].append("provenance_invalid_date:collected_at")
    if requirement["kind"] == "ab_experiment":
        start_at = parse_provenance_datetime(provenance.get("start_date"))
        end_at = parse_provenance_datetime(provenance.get("end_date"))
        if provenance.get("start_date") and start_at is None:
            item["issues"].append("provenance_invalid_date:start_date")
        if provenance.get("end_date") and end_at is None:
            item["issues"].append("provenance_invalid_date:end_date")
        if start_at is not None and end_at is not None:
            if end_at <= start_at:
                item["issues"].append("provenance_date_order:end_date_not_after_start_date")
            else:
                provenance_duration_weeks = (end_at - start_at).total_seconds() / (7 * 24 * 60 * 60)
                item["provenance_duration_weeks"] = provenance_duration_weeks
                min_duration = float(requirement["min_duration_weeks"])
                if provenance_duration_weeks < min_duration:
                    item["issues"].append(
                        f"provenance_duration_weeks_below_formal_min:{provenance_duration_weeks:.2f}<{min_duration:g}"
                    )
    if evaluation.get("status") != "pass":
        item["issues"].append(f"evaluation_status:{evaluation.get('status')}")

    if "min_samples" in requirement:
        sample_count = int(evaluation.get("sample_count", 0) or 0)
        unique_sample_count = unique_evidence_sample_count(task, dataset)
        item["unique_sample_count"] = unique_sample_count
        if sample_count < int(requirement["min_samples"]):
            item["issues"].append(f"sample_count_below_formal_min:{sample_count}<{requirement['min_samples']}")
        if unique_sample_count < int(requirement["min_samples"]):
            item["issues"].append(
                f"unique_sample_count_below_formal_min:{unique_sample_count}<{requirement['min_samples']}"
            )
    elif requirement["kind"] == "production_telemetry":
        active_users = int(evaluation.get("active_users", 0) or 0)
        duration_days = float(evaluation.get("duration_days", 0.0) or 0.0)
        if active_users < int(requirement["min_active_users"]):
            item["issues"].append(f"active_users_below_formal_min:{active_users}<{requirement['min_active_users']}")
        if duration_days < float(requirement["min_duration_days"]):
            item["issues"].append(f"duration_days_below_formal_min:{duration_days}<{requirement['min_duration_days']}")
    else:
        control = (evaluation.get("cohorts") or {}).get("control") or {}
        friend = (evaluation.get("cohorts") or {}).get("friend") or {}
        control_users = int(control.get("users", 0) or 0)
        friend_users = int(friend.get("users", 0) or 0)
        duration_weeks = float(evaluation.get("duration_weeks", 0.0) or 0.0)
        if control_users < int(requirement["min_users_per_cohort"]):
            item["issues"].append(
                f"control_users_below_formal_min:{control_users}<{requirement['min_users_per_cohort']}"
            )
        if friend_users < int(requirement["min_users_per_cohort"]):
            item["issues"].append(f"friend_users_below_formal_min:{friend_users}<{requirement['min_users_per_cohort']}")
        if duration_weeks < float(requirement["min_duration_weeks"]):
            item["issues"].append(f"duration_weeks_below_formal_min:{duration_weeks}<{requirement['min_duration_weeks']}")

    item["formal_ready"] = not item["issues"]
    return item


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_manifest(evidence_dir: Path) -> dict:
    files = []
    for task, filename in EVIDENCE_FILENAMES.items():
        path = evidence_dir / filename
        exists = path.exists()
        stat = path.stat() if exists else None
        files.append(
            {
                "task": task,
                "filename": filename,
                "path": str(path),
                "exists": exists,
                "sha256": _file_sha256(path) if exists else None,
                "size_bytes": stat.st_size if stat else None,
                "mtime_ns": stat.st_mtime_ns if stat else None,
                "required": FORMAL_EVIDENCE_REQUIREMENTS[task],
            }
        )
    ai_observation_path = evidence_dir / AI_OBSERVATION_FILENAME
    ai_observation_stat = ai_observation_path.stat() if ai_observation_path.exists() else None
    return {
        "schema": "friend-memory-evidence-manifest-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_dir": str(evidence_dir),
        "files": files,
        "ai_observation": {
            "filename": AI_OBSERVATION_FILENAME,
            "path": str(ai_observation_path),
            "exists": ai_observation_path.exists(),
            "sha256": _file_sha256(ai_observation_path) if ai_observation_path.exists() else None,
            "size_bytes": ai_observation_stat.st_size if ai_observation_stat else None,
            "mtime_ns": ai_observation_stat.st_mtime_ns if ai_observation_stat else None,
            "required_when": "release-gate --require-external-ai and no runtime external AI decision is available",
        },
    }


def verify_evidence_manifest(evidence_dir: Path, manifest_path: Path) -> dict:
    result = {
        "manifest_path": str(manifest_path),
        "verified": False,
        "issues": [],
        "file_results": [],
    }
    if not manifest_path.exists():
        result["issues"].append("manifest_missing")
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["issues"].append(f"manifest_invalid_json:{exc.msg}")
        return result
    if manifest.get("schema") != "friend-memory-evidence-manifest-v1":
        result["issues"].append("manifest_schema_mismatch")
    by_task = {item.get("task"): item for item in manifest.get("files", []) if isinstance(item, dict)}
    for task, filename in EVIDENCE_FILENAMES.items():
        expected = by_task.get(task)
        path = evidence_dir / filename
        file_result = {"task": task, "filename": filename, "ok": False, "issues": []}
        if not expected:
            file_result["issues"].append("manifest_entry_missing")
        elif not path.exists():
            file_result["issues"].append("file_missing")
        else:
            current_hash = _file_sha256(path)
            file_result["sha256"] = current_hash
            if expected.get("filename") != filename:
                file_result["issues"].append("filename_mismatch")
            if expected.get("sha256") != current_hash:
                file_result["issues"].append("sha256_mismatch")
            current_size = path.stat().st_size
            file_result["size_bytes"] = current_size
            if expected.get("size_bytes") != current_size:
                file_result["issues"].append("size_mismatch")
        file_result["ok"] = not file_result["issues"]
        if file_result["issues"]:
            result["issues"].append(f"{task}:{','.join(file_result['issues'])}")
        result["file_results"].append(file_result)
    ai_expected = manifest.get("ai_observation") or {}
    path = evidence_dir / AI_OBSERVATION_FILENAME
    if ai_expected.get("exists"):
        file_result = {"task": "ai_observation", "filename": AI_OBSERVATION_FILENAME, "ok": False, "issues": []}
        if not path.exists():
            file_result["issues"].append("file_missing")
        else:
            current_hash = _file_sha256(path)
            file_result["sha256"] = current_hash
            if ai_expected.get("filename") != AI_OBSERVATION_FILENAME:
                file_result["issues"].append("filename_mismatch")
            if ai_expected.get("sha256") != current_hash:
                file_result["issues"].append("sha256_mismatch")
            current_size = path.stat().st_size
            file_result["size_bytes"] = current_size
            if ai_expected.get("size_bytes") != current_size:
                file_result["issues"].append("size_mismatch")
        file_result["ok"] = not file_result["issues"]
        if file_result["issues"]:
            result["issues"].append(f"ai_observation:{','.join(file_result['issues'])}")
        result["file_results"].append(file_result)
    elif path.exists():
        file_result = {
            "task": "ai_observation",
            "filename": AI_OBSERVATION_FILENAME,
            "ok": False,
            "issues": ["unexpected_file_not_in_manifest"],
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        result["issues"].append("ai_observation:unexpected_file_not_in_manifest")
        result["file_results"].append(file_result)
    result["verified"] = not result["issues"]
    return result


def validate_evidence_dir_for_project(
    project: FriendMemoryProject,
    evidence_dir: Path,
    *,
    manifest_path: Path | None = None,
) -> dict:
    items = [formal_evidence_item(project, evidence_dir, task) for task in EVALUATION_TASKS]
    ready_items = [item for item in items if item["formal_ready"]]
    missing_items = [item for item in items if not item["exists"]]
    remediation = [formal_evidence_remediation(item, evidence_dir) for item in items if not item["formal_ready"]]
    manifest_verification = verify_evidence_manifest(evidence_dir, manifest_path) if manifest_path else None
    manifest_ready = manifest_verification is None or manifest_verification["verified"]
    ready = len(ready_items) == len(items) and manifest_ready
    return {
        "evidence_dir": str(evidence_dir),
        "status": "formal_ready" if ready else "incomplete",
        "formal_ready": ready,
        "ready_count": len(ready_items),
        "required_count": len(items),
        "missing_count": len(missing_items),
        "manifest_verification": manifest_verification,
        "items": items,
        "remediation": remediation,
        "note": (
            "This validates formal evidence scale and provenance. Small demo fixtures can pass metric evaluators "
            "but remain formal_ready=false until they meet sample/user/duration and provenance requirements."
        ),
    }


def formal_evidence_remediation(item: dict, evidence_dir: Path) -> dict:
    task = item["task"]
    filename = item["filename"]
    path = evidence_dir / filename
    return {
        "task": task,
        "filename": filename,
        "path": str(path),
        "issues": item["issues"],
        "template_command": (
            "uv --cache-dir .uv-cache run python app/main.py evidence-template "
            f"--task {shell_arg(task)} --output {shell_arg(path)}"
        ),
        "evaluate_command": (
            "uv --cache-dir .uv-cache run python app/main.py evaluate-labels "
            f"{shell_arg(path)} --task {shell_arg(task)} --json"
        ),
        "validate_command": (
            "uv --cache-dir .uv-cache run python app/main.py evidence-validate "
            f"--evidence-dir {shell_arg(evidence_dir)} --json"
        ),
    }


def validate_evidence_dir(evidence_dir: Path, manifest_path: Path | None = None) -> dict:
    return validate_evidence_dir_for_project(load_project(), evidence_dir, manifest_path=manifest_path)


def cmd_evidence_validate(args: argparse.Namespace) -> None:
    report = validate_evidence_dir(Path(args.evidence_dir), manifest_path=Path(args.manifest) if args.manifest else None)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"evidence_validate status={report['status']} "
            f"ready={report['ready_count']}/{report['required_count']} missing={report['missing_count']}"
        )
        for item in report["items"]:
            issue_text = ",".join(item["issues"]) if item["issues"] else "ok"
            print(f"{item['task']} formal_ready={item['formal_ready']} issues={issue_text}")
    if args.strict and not report["formal_ready"]:
        raise SystemExit(1)


def cmd_evidence_manifest(args: argparse.Namespace) -> None:
    manifest = evidence_manifest(Path(args.evidence_dir))
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote={args.output}")
        return
    print(text)


def cmd_doctor(args: argparse.Namespace) -> None:
    report = project_doctor_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"doctor status={report['status']} root={report['project_root']}")
        summary = report["summary"]
        print(
            f"required_passed={summary['required_passed']} "
            f"required_failed={summary['required_failed']} total_checks={summary['total_checks']}"
        )
        for item in report["checks"]:
            marker = "ok" if item["ok"] else "fail"
            print(f"{marker} {item['name']}: {item['detail']}")
        print("next_commands:")
        for command in report["next_commands"]:
            print(f"- {command}")
    if args.strict and report["status"] != "PASS":
        raise SystemExit(1)


def cli_command_catalog(parser: argparse.ArgumentParser | None = None) -> dict:
    parser = parser or build_parser()
    subparser_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    help_by_command = {action.dest: action.help for action in subparser_action._choices_actions}
    commands = []
    for name, command_parser in sorted(subparser_action.choices.items()):
        option_strings = {
            option
            for action in command_parser._actions
            for option in action.option_strings
        }
        positionals = [
            action.dest
            for action in command_parser._actions
            if not action.option_strings and action.dest != argparse.SUPPRESS
        ]
        commands.append(
            {
                "name": name,
                "help": help_by_command.get(name),
                "supports_json": "--json" in option_strings,
                "options": sorted(option_strings),
                "positionals": positionals,
            }
        )
    return {
        "schema": "friend-memory-cli-commands-v1",
        "command_count": len(commands),
        "commands": commands,
    }


def cmd_commands(args: argparse.Namespace) -> None:
    catalog = cli_command_catalog()
    if args.json:
        print(json.dumps(catalog, ensure_ascii=False, indent=2))
        return
    print(f"commands count={catalog['command_count']}")
    for command in catalog["commands"]:
        json_marker = " json" if command["supports_json"] else ""
        print(f"{command['name']}{json_marker}: {command['help'] or ''}")


def release_gate_report_for_project(
    project: FriendMemoryProject,
    *,
    evidence_dir: Path,
    manifest_path: Path | None = None,
    relationship_id: str | None = None,
    run_benchmarks: bool = False,
    benchmark_iterations: int = 20,
    require_external_ai: bool = False,
) -> dict:
    evidence_report = validate_evidence_dir_for_project(project, evidence_dir, manifest_path=manifest_path)
    manifest_verification = evidence_report.get("manifest_verification")
    evaluation_results = collect_evaluation_results(project, evidence_dir=evidence_dir)
    audit = project.audit_report(relationship_id)
    decision = project.decision_report(
        relationship_id,
        run_benchmarks=run_benchmarks,
        benchmark_iterations=benchmark_iterations,
        evaluation_results=evaluation_results,
    )
    gates = {
        "runtime_audit_pass": audit["status"] == "PASS",
        "formal_evidence_ready": evidence_report["formal_ready"],
        "decision_locally_proven": decision["summary"]["completion_claim"] == "locally_proven",
        "benchmarks_included": bool(run_benchmarks),
        "no_failed_decision_criteria": not bool(decision["summary"]["status_counts"].get("fail")),
        "external_ai_observed": decision["ai_readiness"]["status"] == "external_observed",
    }
    ai_observation = validate_ai_observation(evidence_dir, relationship_id)
    external_ai_source = "runtime_ai_readiness" if gates["external_ai_observed"] else "not_observed"
    if require_external_ai and not gates["external_ai_observed"] and ai_observation["formal_ready"]:
        gates["external_ai_observed"] = True
        external_ai_source = "ai_observation_file"
    if not require_external_ai:
        gates["external_ai_observed"] = True
        external_ai_source = "not_required"
    blocked = [name for name, passed in gates.items() if not passed]
    status = "release_ready" if not blocked else "blocked"
    return {
        "status": status,
        "release_ready": status == "release_ready",
        "blocked_gates": blocked,
        "gates": gates,
        "scope": decision["scope"],
        "relationship_id": relationship_id,
        "require_external_ai": require_external_ai,
        "run_benchmarks": run_benchmarks,
        "benchmark_iterations": benchmark_iterations,
        "external_ai_source": external_ai_source,
        "audit_status": audit["status"],
        "decision_summary": decision["summary"],
        "ai_readiness": decision["ai_readiness"],
        "ai_observation": ai_observation,
        "formal_evidence": {
            "status": evidence_report["status"],
            "formal_ready": evidence_report["formal_ready"],
            "ready_count": evidence_report["ready_count"],
            "required_count": evidence_report["required_count"],
            "missing_count": evidence_report["missing_count"],
            "manifest_verified": (
                None if manifest_verification is None else bool(manifest_verification.get("verified"))
            ),
            "manifest_issues": [] if manifest_verification is None else manifest_verification.get("issues", []),
            "not_ready_items": [
                {
                    "task": item["task"],
                    "filename": item["filename"],
                    "issues": item["issues"],
                }
                for item in evidence_report["items"]
                if not item["formal_ready"]
            ],
            "remediation": evidence_report["remediation"],
        },
        "note": (
            "release_ready requires runtime audit PASS, formal evidence scale, locally_proven decision report, "
            "and benchmark inclusion. Use --require-external-ai to additionally require observed external model/worker usage."
        ),
    }


def release_gate_report(args: argparse.Namespace) -> dict:
    relationship_id = args.relationship_id or (f"{args.user}:{args.ai}" if args.relationship else None)
    return release_gate_report_for_project(
        load_project(),
        evidence_dir=Path(args.evidence_dir),
        manifest_path=Path(args.manifest) if args.manifest else None,
        relationship_id=relationship_id,
        run_benchmarks=args.run_benchmarks,
        benchmark_iterations=args.benchmark_iterations,
        require_external_ai=args.require_external_ai,
    )


def cmd_release_gate(args: argparse.Namespace) -> None:
    report = release_gate_report(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"release_gate status={report['status']} scope={report['scope']}")
        print(f"blocked_gates={','.join(report['blocked_gates']) if report['blocked_gates'] else 'none'}")
        print(f"audit_status={report['audit_status']}")
        print(f"completion_claim={report['decision_summary']['completion_claim']}")
        print(
            f"formal_evidence={report['formal_evidence']['ready_count']}/"
            f"{report['formal_evidence']['required_count']}"
        )
    if args.strict and not report["release_ready"]:
        raise SystemExit(1)


def _write_json_file(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def release_bundle_for_project(
    project: FriendMemoryProject,
    *,
    evidence_dir: Path,
    output_dir: Path,
    relationship_id: str | None = None,
    run_benchmarks: bool = False,
    benchmark_iterations: int = 20,
    require_external_ai: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_manifest_path = output_dir / EVIDENCE_MANIFEST_FILENAME
    validation_path = output_dir / "evidence-validate.json"
    decision_path = output_dir / "decision-report.json"
    release_gate_path = output_dir / "release-gate.json"
    bundle_path = output_dir / "release-bundle.json"

    evidence_manifest_payload = evidence_manifest(evidence_dir)
    _write_json_file(evidence_manifest_path, evidence_manifest_payload)
    validation = validate_evidence_dir_for_project(project, evidence_dir, manifest_path=evidence_manifest_path)
    decision = decision_report_with_evidence(
        project,
        relationship_id=relationship_id,
        run_benchmarks=run_benchmarks,
        benchmark_iterations=benchmark_iterations,
        evidence_dir=evidence_dir,
        manifest_path=evidence_manifest_path,
    )
    gate = release_gate_report_for_project(
        project,
        evidence_dir=evidence_dir,
        manifest_path=evidence_manifest_path,
        relationship_id=relationship_id,
        run_benchmarks=run_benchmarks,
        benchmark_iterations=benchmark_iterations,
        require_external_ai=require_external_ai,
    )

    _write_json_file(validation_path, validation)
    _write_json_file(decision_path, decision)
    _write_json_file(release_gate_path, gate)

    reports = []
    for label, path in [
        ("evidence_manifest", evidence_manifest_path),
        ("evidence_validate", validation_path),
        ("decision_report", decision_path),
        ("release_gate", release_gate_path),
    ]:
        reports.append(
            {
                "name": label,
                "filename": path.name,
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    next_actions = release_bundle_next_actions(gate, evidence_dir, output_dir)
    bundle = {
        "schema": "friend-memory-release-bundle-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_dir": str(evidence_dir),
        "output_dir": str(output_dir),
        "relationship_id": relationship_id,
        "run_benchmarks": run_benchmarks,
        "benchmark_iterations": benchmark_iterations,
        "require_external_ai": require_external_ai,
        "release_ready": gate["release_ready"],
        "status": gate["status"],
        "blocked_gates": gate["blocked_gates"],
        "next_actions": next_actions,
        "reports": reports,
        "note": "Bundle stores report hashes and an evidence manifest; raw evidence files remain in evidence_dir.",
    }
    _write_json_file(bundle_path, bundle)
    return bundle


def release_bundle_next_actions(gate: dict, evidence_dir: Path, output_dir: Path) -> list[dict]:
    actions: list[dict] = []
    external_ai_flag = " --require-external-ai" if gate.get("require_external_ai") else ""
    relationship_flag = f" --relationship-id {shell_arg(gate['relationship_id'])}" if gate.get("relationship_id") else ""
    benchmark_iterations = int(gate.get("benchmark_iterations", 20) or 20)
    benchmark_flag = (
        f" --run-benchmarks --benchmark-iterations {benchmark_iterations}"
        if gate.get("run_benchmarks") or (gate.get("gates") or {}).get("benchmarks_included")
        else ""
    )
    for gate_name in gate["blocked_gates"]:
        if gate_name == "formal_evidence_ready":
            manifest_issues = gate.get("formal_evidence", {}).get("manifest_issues", [])
            if gate.get("formal_evidence", {}).get("manifest_verified") is False:
                actions.append(
                    {
                        "type": "manifest_integrity",
                        "gate": gate_name,
                        "issues": manifest_issues,
                        "command": (
                            "uv --cache-dir .uv-cache run python app/main.py evidence-manifest "
                            f"--evidence-dir {shell_arg(evidence_dir)} "
                            f"--output {shell_arg(output_dir / EVIDENCE_MANIFEST_FILENAME)}"
                        ),
                        "follow_up": (
                            "uv --cache-dir .uv-cache run python app/main.py release-bundle "
                            f"--evidence-dir {shell_arg(evidence_dir)} --output-dir {shell_arg(output_dir)}"
                            f"{relationship_flag}{benchmark_flag}{external_ai_flag} --json"
                        ),
                    }
                )
            for item in gate["formal_evidence"].get("remediation", []):
                actions.append(
                    {
                        "type": "formal_evidence",
                        "gate": gate_name,
                        "task": item["task"],
                        "filename": item["filename"],
                        "issues": item["issues"],
                        "command": item["template_command"],
                        "validation_command": item["validate_command"],
                        "follow_up": (
                            "uv --cache-dir .uv-cache run python app/main.py release-bundle "
                            f"--evidence-dir {shell_arg(evidence_dir)} --output-dir {shell_arg(output_dir)}"
                            f"{relationship_flag}{benchmark_flag}{external_ai_flag} --json"
                        ),
                    }
                )
        elif gate_name == "benchmarks_included":
            actions.append(
                {
                    "type": "benchmark",
                    "gate": gate_name,
                    "command": (
                        "uv --cache-dir .uv-cache run python app/main.py release-bundle "
                        f"--evidence-dir {shell_arg(evidence_dir)} --output-dir {shell_arg(output_dir)} "
                        f"{relationship_flag} --run-benchmarks --benchmark-iterations {benchmark_iterations}"
                        f"{external_ai_flag} --json"
                    ),
                }
            )
        elif gate_name == "external_ai_observed":
            actions.append(
                {
                    "type": "external_ai",
                    "gate": gate_name,
                    "issues": gate.get("ai_observation", {}).get("issues", []),
                    "command": (
                        "uv --cache-dir .uv-cache run python app/main.py ai-observation "
                        f"--output {shell_arg(evidence_dir / AI_OBSERVATION_FILENAME)}{relationship_flag}"
                    ),
                    "follow_up": (
                        "uv --cache-dir .uv-cache run python app/main.py release-bundle "
                        f"--evidence-dir {shell_arg(evidence_dir)} --output-dir {shell_arg(output_dir)}"
                        f"{relationship_flag}{benchmark_flag} --require-external-ai --json"
                    ),
                }
            )
        else:
            actions.append(
                {
                    "type": "gate",
                    "gate": gate_name,
                    "command": (
                        "uv --cache-dir .uv-cache run python app/main.py release-gate "
                        f"--evidence-dir {shell_arg(evidence_dir)}"
                        f"{relationship_flag}{benchmark_flag}{external_ai_flag} --json"
                    ),
                }
            )
    return actions


def cmd_release_bundle(args: argparse.Namespace) -> None:
    relationship_id = args.relationship_id or (f"{args.user}:{args.ai}" if args.relationship else None)
    bundle = release_bundle_for_project(
        load_project(),
        evidence_dir=Path(args.evidence_dir),
        output_dir=Path(args.output_dir),
        relationship_id=relationship_id,
        run_benchmarks=args.run_benchmarks,
        benchmark_iterations=args.benchmark_iterations,
        require_external_ai=args.require_external_ai,
    )
    if args.json:
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
    else:
        print(f"release_bundle status={bundle['status']} output_dir={bundle['output_dir']}")
        print(f"release_ready={bundle['release_ready']} blocked_gates={','.join(bundle['blocked_gates']) or 'none'}")


def cmd_evidence_template(args: argparse.Namespace) -> None:
    payload = {task: evaluation_template(task) for task in EVALUATION_TASKS} if args.all else evaluation_template(args.task)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote={args.output}")
        return
    print(text)


def cmd_stage_rollback(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = args.relationship_id or f"{args.user}:{args.ai}"
    result = project.rollback_stage(relationship_id, history_index=args.history_index, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"stage_rollback={relationship_id} from={result['from']} to={result['to']}")


def cmd_story_correct(args: argparse.Namespace) -> None:
    project = load_project()
    correction = project.correct_story_consensus(args.story_id, args.consensus, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(correction, ensure_ascii=False, indent=2))
    else:
        print(f"story_corrected={args.story_id}")


def cmd_story_confirm(args: argparse.Namespace) -> None:
    project = load_project()
    event = project.confirm_story_consensus(args.story_id, note=args.note)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"story_confirmed={args.story_id} status={event['status']}")


def cmd_story_rollback(args: argparse.Namespace) -> None:
    project = load_project()
    event = project.rollback_story_narrative(args.story_id, version_index=args.version_index, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"story_rollback={args.story_id} from={event['from_level']} to={event['to_level']}")


def cmd_migrate(args: argparse.Namespace) -> None:
    project = load_project()
    raw = Path(args.input).read_text(encoding="utf-8")
    if args.format == "jsonl":
        turns = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        payload = json.loads(raw)
        turns = payload.get("turns", payload) if isinstance(payload, dict) else payload
    if not isinstance(turns, list):
        raise ValueError("migration input must be a JSON array, JSON object with `turns`, or JSONL records")
    certificate = json.loads(Path(args.certificate).read_text(encoding="utf-8")) if args.certificate else None
    report = project.migrate_legacy_turns(
        turns,
        default_user=args.user,
        default_ai=args.ai,
        relationship_certificate=certificate,
        require_certificate=args.require_certificate,
        target_mode=Mode(args.target_mode) if args.target_mode else None,
    )
    save_project(project)
    payload = {
        "migration_id": report.migration_id,
        "imported_turns": report.imported_turns,
        "relationship_ids": report.relationship_ids,
        "created_memories": report.created_memories,
        "created_emotional_memories": report.created_emotional_memories,
        "recognized_milestones": report.recognized_milestones,
        "rollback_expires_at": report.rollback_expires_at.isoformat(),
        "relationship_certificate": project.migration_batches[report.migration_id].get("relationship_certificate"),
        "target_mode": project.migration_batches[report.migration_id].get("target_mode"),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"migration={report.migration_id} imported={report.imported_turns} "
            f"relationships={','.join(report.relationship_ids)} rollback_expires_at={report.rollback_expires_at.isoformat()}"
        )


def cmd_migration_cert(args: argparse.Namespace) -> None:
    project = load_project()
    raw = Path(args.input).read_text(encoding="utf-8")
    if args.format == "jsonl":
        turns = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        payload = json.loads(raw)
        turns = payload.get("turns", payload) if isinstance(payload, dict) else payload
    if not isinstance(turns, list):
        raise ValueError("migration input must be a JSON array, JSON object with `turns`, or JSONL records")
    certificate = project.build_migration_certificate(turns, default_user=args.user, default_ai=args.ai)
    print(json.dumps(certificate, ensure_ascii=False, indent=2))


def cmd_migrate_rollback(args: argparse.Namespace) -> None:
    project = load_project()
    ok = project.rollback_migration(args.migration_id)
    save_project(project)
    print(f"rolled_back={ok}")


def cmd_health_ack(args: argparse.Namespace) -> None:
    project = load_project()
    alert = project.acknowledge_health_alert(args.alert_id, note=args.note)
    save_project(project)
    if args.json:
        print(
            json.dumps(
                {
                    "alert_id": alert.alert_id,
                    "acknowledged": alert.acknowledged,
                    "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    "acknowledgement_note": alert.acknowledgement_note,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"acknowledged={args.alert_id} at={alert.acknowledged_at.isoformat() if alert.acknowledged_at else 'None'}")


def cmd_health_feedback(args: argparse.Namespace) -> None:
    project = load_project()
    event = project.record_health_alert_feedback(args.alert_id, args.feedback, note=args.note)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
        return
    print(f"health_feedback={args.alert_id} feedback={event['feedback']}")


def cmd_age(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    project.set_user_age(relationship_id, args.age)
    save_project(project)
    print(f"user_age={args.age}")


def cmd_minutes(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    project.record_interaction_minutes(relationship_id, args.date, args.minutes)
    save_project(project)
    print(f"minutes[{args.date}]={args.minutes}")


def cmd_inject(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = f"{args.user}:{args.ai}"
    if relationship_id not in project.relationships:
        project.get_or_create_relationship(args.user, args.ai)
    try:
        memory_id = project.inject_memory(
            relationship_id,
            args.text,
            memory_type=MemoryType(args.memory_type),
            context_tag=ContextTag(args.context_tag),
            milestone=args.milestone,
            core_identity=args.core_identity,
            force_memory_write=args.force_memory_write,
        )
    except PermissionError as exc:
        save_project(project)
        if args.json:
            print(
                json.dumps(
                    {
                        "error": str(exc),
                        "error_type": "PermissionError",
                        "relationship_id": relationship_id,
                        "forced": args.force_memory_write,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(1) from None
        print(f"error={exc}")
        raise SystemExit(1) from None
    save_project(project)
    if args.json:
        memory = project.memories[memory_id]
        print(
            json.dumps(
                {
                    "memory_id": memory_id,
                    "relationship_id": relationship_id,
                    "memory_type": memory.memory_type.value,
                    "context_tag": memory.context_tag.value,
                    "storage_layer": memory.storage_layer.value,
                    "forced": args.force_memory_write,
                    "memory_pause_override": memory.metadata.get("memory_pause_override", False),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"memory={memory_id}")


def cmd_edit(args: argparse.Namespace) -> None:
    project = load_project()
    project.edit_memory(args.memory_id, args.text, reason=args.reason)
    save_project(project)
    print(f"edited={args.memory_id}")


def cmd_retag(args: argparse.Namespace) -> None:
    project = load_project()
    project.retag_memory(
        args.memory_id,
        memory_type=MemoryType(args.memory_type) if args.memory_type else None,
        context_tag=ContextTag(args.context_tag) if args.context_tag else None,
        reason=args.reason,
    )
    save_project(project)
    memory = project.memories[args.memory_id]
    if args.json:
        print(
            json.dumps(
                {
                    "memory_id": memory.memory_id,
                    "memory_type": memory.memory_type.value,
                    "context_tag": memory.context_tag.value,
                    "storage_layer": memory.storage_layer.value,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"retagged={args.memory_id} type={memory.memory_type.value} context={memory.context_tag.value}")


def cmd_memory_suppress(args: argparse.Namespace) -> None:
    project = load_project()
    event = project.suppress_memory(args.memory_id, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"suppressed={args.memory_id} reason={args.reason}")


def cmd_memory_unsuppress(args: argparse.Namespace) -> None:
    project = load_project()
    event = project.unsuppress_memory(args.memory_id, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"unsuppressed={args.memory_id} reason={args.reason}")


def cmd_memory_restore_archive(args: argparse.Namespace) -> None:
    project = load_project()
    event = project.restore_archived_memory(args.memory_id, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"restored_archive={args.memory_id} reason={args.reason}")


def cmd_memory_verify(args: argparse.Namespace) -> None:
    project = load_project()
    event = project.verify_memory(args.memory_id, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"verified={args.memory_id} confidence={event['confidence']:.3f}")


def cmd_memory_calibrate(args: argparse.Namespace) -> None:
    project = load_project()
    event = project.calibrate_memory(args.memory_id, args.outcome, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(
            f"calibrated={args.memory_id} outcome={event['outcome']} "
            f"confidence={event['confidence_before']:.3f}->{event['confidence_after']:.3f}"
        )


def cmd_retention_feedback(args: argparse.Namespace) -> None:
    project = load_project()
    event = project.record_retention_feedback(args.memory_id, args.outcome, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(
            f"retention_feedback={args.memory_id} outcome={event['outcome']} "
            f"multiplier={event['effective_multiplier']:.3f}"
        )


def cmd_time_conflict_resolve(args: argparse.Namespace) -> None:
    project = load_project()
    event = project.resolve_time_conflict(
        args.conflict_id,
        resolution=args.resolution,
        preferred_memory_id=args.preferred_memory_id,
        note=args.note,
    )
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        preferred = f" preferred={event['preferred_memory_id']}" if event.get("preferred_memory_id") else ""
        print(f"time_conflict_resolved={args.conflict_id} resolution={event['resolution']}{preferred}")


def cmd_inside_joke_status(args: argparse.Namespace) -> None:
    project = load_project()
    event = project.set_inside_joke_status(
        args.memory_id,
        active=args.action == "reactivate",
        reason=args.reason,
    )
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"{event['type']}={args.memory_id} inactive={event['inactive_after']}")


def cmd_thread_resolve(args: argparse.Namespace) -> None:
    project = load_project()
    event = project.resolve_unresolved_thread(
        args.memory_id,
        resolution=args.resolution,
        reason=args.reason,
    )
    save_project(project)
    if args.json:
        print(json.dumps(event, ensure_ascii=False, indent=2))
    else:
        print(f"thread_resolved={args.memory_id} resolution={event['resolution']}")


def cmd_mark_milestone(args: argparse.Namespace) -> None:
    project = load_project()
    project.mark_milestone(args.memory_id)
    save_project(project)
    print(f"milestone={args.memory_id}")


def milestone_payload(project: FriendMemoryProject, memory_id: str, action: str) -> dict[str, Any]:
    memory = project.memories[memory_id]
    relationship = project.relationships[memory.relationship_id]
    confirmation = memory.metadata.get("milestone_confirmation", {})
    return {
        "type": action,
        "memory_id": memory_id,
        "relationship_id": memory.relationship_id,
        "memory_type": memory.memory_type.value,
        "context_tag": memory.context_tag.value,
        "is_milestone": memory_id in relationship.milestones,
        "confirmation": confirmation,
        "confirmation_status": confirmation.get("status"),
    }


def cmd_confirm_milestone(args: argparse.Namespace) -> None:
    project = load_project()
    project.confirm_milestone(args.memory_id, title=args.title, description=args.description)
    save_project(project)
    if args.json:
        print(json.dumps(milestone_payload(project, args.memory_id, "milestone_confirmed"), ensure_ascii=False, indent=2))
        return
    print(f"milestone_confirmed={args.memory_id}")


def cmd_edit_milestone(args: argparse.Namespace) -> None:
    project = load_project()
    event = project.edit_milestone(args.memory_id, title=args.title, description=args.description)
    save_project(project)
    if args.json:
        print_json(event)
    else:
        print(f"milestone_edited={args.memory_id}")


def cmd_reject_milestone(args: argparse.Namespace) -> None:
    project = load_project()
    project.reject_milestone(args.memory_id, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(milestone_payload(project, args.memory_id, "milestone_rejected"), ensure_ascii=False, indent=2))
        return
    print(f"milestone_rejected={args.memory_id}")


def cmd_downgrade(args: argparse.Namespace) -> None:
    project = load_project()
    project.downgrade_memory(args.memory_id, reason=args.reason)
    save_project(project)
    print(f"downgraded={args.memory_id}")


def cmd_batch_downgrade(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = args.relationship_id or f"{args.user}:{args.ai}"
    memory_ids = args.memory_ids or None
    if memory_ids:
        relationship_id = args.relationship_id
    downgraded = project.batch_downgrade_memories(
        relationship_id,
        memory_ids=memory_ids,
        memory_type=MemoryType(args.memory_type) if args.memory_type else None,
        context_tag=ContextTag(args.context_tag) if args.context_tag else None,
        storage_layer=MemoryLayer(args.storage_layer) if args.storage_layer else None,
        reason=args.reason,
    )
    save_project(project)
    if args.json:
        print(json.dumps({"downgraded": downgraded, "count": len(downgraded)}, ensure_ascii=False, indent=2))
    else:
        print(f"downgraded={len(downgraded)}")
        for memory_id in downgraded:
            print(memory_id)


def cmd_reminders(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = args.relationship_id or f"{args.user}:{args.ai}"
    reminders = project.check_commitment_reminders(relationship_id, window_days=args.window_days)
    save_project(project)
    if args.json:
        print(json.dumps({"reminders": reminders, "count": len(reminders)}, ensure_ascii=False, indent=2))
    else:
        for reminder in reminders:
            print(f"{reminder['reminder_id']} {reminder['due_phrase']} {reminder['title']}")


def cmd_reminder_complete(args: argparse.Namespace) -> None:
    project = load_project()
    project.complete_commitment_reminder(args.reminder_id)
    save_project(project)
    print(f"completed={args.reminder_id}")


def cmd_memory_delete_request(args: argparse.Namespace) -> None:
    project = load_project()
    request = project.request_memory_delete(args.memory_id, reason=args.reason)
    save_project(project)
    if args.json:
        print(
            json.dumps(
                {
                    "request_id": request.request_id,
                    "memory_id": request.memory_id,
                    "relationship_id": request.relationship_id,
                    "status": request.status.value,
                    "requested_at": request.requested_at.isoformat(),
                    "execute_after": request.execute_after.isoformat(),
                    "reason": request.reason,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"memory_delete_request={request.request_id} execute_after={request.execute_after.isoformat()}")


def cmd_memory_delete_confirm(args: argparse.Namespace) -> None:
    project = load_project()
    ok = project.confirm_memory_delete(args.request_id, force=args.force)
    request = project.memory_delete_requests[args.request_id]
    save_project(project)
    if args.json:
        print(
            json.dumps(
                {
                    "request_id": args.request_id,
                    "memory_id": request.memory_id,
                    "relationship_id": request.relationship_id,
                    "confirmed": ok,
                    "status": request.status.value,
                    "executed_at": request.executed_at.isoformat() if request.executed_at else None,
                    "memory_present": request.memory_id in project.memories,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"confirmed={ok}")


def cmd_memory_delete_cancel(args: argparse.Namespace) -> None:
    project = load_project()
    project.cancel_memory_delete(args.request_id)
    request = project.memory_delete_requests[args.request_id]
    save_project(project)
    if args.json:
        print(
            json.dumps(
                {
                    "request_id": args.request_id,
                    "memory_id": request.memory_id,
                    "relationship_id": request.relationship_id,
                    "status": request.status.value,
                    "memory_present": request.memory_id in project.memories,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"cancelled={args.request_id}")


def cmd_deletion_compliance(args: argparse.Namespace) -> None:
    project = load_project()
    relationship_id = args.relationship_id or f"{args.user}:{args.ai}"
    report = project.deletion_compliance_audit(relationship_id, auditor_token=args.auditor_token)
    save_project(project)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"deletion_compliance relationship={relationship_id} records={len(report['records'])}")


def cmd_l4_delete_request(args: argparse.Namespace) -> None:
    project = load_project()
    request = project.request_core_identity_delete(args.identity_id, reason=args.reason)
    save_project(project)
    if args.json:
        print(
            json.dumps(
                {
                    "request_id": request.request_id,
                    "identity_id": request.identity_id,
                    "memory_id": request.memory_id,
                    "relationship_id": request.relationship_id,
                    "status": request.status.value,
                    "requested_at": request.requested_at.isoformat(),
                    "execute_after": request.execute_after.isoformat(),
                    "reason": request.reason,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"l4_delete_request={request.request_id} execute_after={request.execute_after.isoformat()}")


def cmd_l4_delete_confirm(args: argparse.Namespace) -> None:
    project = load_project()
    ok = project.confirm_core_identity_delete(args.request_id, force=args.force)
    request = project.core_identity_delete_requests[args.request_id]
    save_project(project)
    if args.json:
        print(
            json.dumps(
                {
                    "request_id": args.request_id,
                    "identity_id": request.identity_id,
                    "memory_id": request.memory_id,
                    "relationship_id": request.relationship_id,
                    "confirmed": ok,
                    "status": request.status.value,
                    "executed_at": request.executed_at.isoformat() if request.executed_at else None,
                    "identity_present": request.identity_id in project.core_identity,
                    "memory_present": request.memory_id in project.memories,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"confirmed={ok}")


def cmd_l4_delete_cancel(args: argparse.Namespace) -> None:
    project = load_project()
    project.cancel_core_identity_delete(args.request_id)
    request = project.core_identity_delete_requests[args.request_id]
    save_project(project)
    if args.json:
        print(
            json.dumps(
                {
                    "request_id": args.request_id,
                    "identity_id": request.identity_id,
                    "memory_id": request.memory_id,
                    "relationship_id": request.relationship_id,
                    "status": request.status.value,
                    "identity_present": request.identity_id in project.core_identity,
                    "memory_present": request.memory_id in project.memories,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(f"cancelled={args.request_id}")


def cmd_l4_review(args: argparse.Namespace) -> None:
    project = load_project()
    entry = project.confirm_core_identity_review(args.identity_id, decision=args.decision, reason=args.reason)
    save_project(project)
    if args.json:
        print(json.dumps(entry, ensure_ascii=False, indent=2))
    else:
        print(f"l4_review={args.identity_id} status={entry['review_status']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Friend-style relational memory project")
    parser.add_argument(
        "--state",
        help=(
            "state JSON path for this invocation; overrides FRIEND_MEMORY_STATE "
            "without changing the default project state file"
        ),
    )
    sub = parser.add_subparsers(required=True)

    ingest = sub.add_parser("ingest", help="record a user turn")
    ingest.add_argument("text")
    ingest.add_argument("--user", default="user")
    ingest.add_argument("--ai", default="companion")
    ingest.set_defaults(func=cmd_ingest)

    ingest_exchange = sub.add_parser("ingest-exchange", help="record a complete user/assistant exchange as one memory")
    ingest_exchange.add_argument("user_text")
    ingest_exchange.add_argument("assistant_text")
    ingest_exchange.add_argument("--user", default="user")
    ingest_exchange.add_argument("--ai", default="companion")
    ingest_exchange.add_argument("--json", action="store_true")
    ingest_exchange.set_defaults(func=cmd_ingest_exchange)

    retrieve = sub.add_parser("retrieve", help="retrieve memories for a query")
    retrieve.add_argument("query")
    retrieve.add_argument("--user", default="user")
    retrieve.add_argument("--ai", default="companion")
    retrieve.add_argument("--limit", type=int, default=5)
    retrieve.add_argument("--explain", action="store_true")
    retrieve.add_argument("--include-archived", action="store_true")
    retrieve.add_argument("--json", action="store_true")
    retrieve.set_defaults(func=cmd_retrieve)

    status = sub.add_parser("status", help="show relationship state")
    status.add_argument("--user", default="user")
    status.add_argument("--ai", default="companion")
    status.set_defaults(func=cmd_status)

    doctor = sub.add_parser("doctor", help="run a read-only project structure and readiness self-check")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--strict", action="store_true", help="exit non-zero if required checks fail")
    doctor.set_defaults(func=cmd_doctor)

    commands = sub.add_parser("commands", help="list machine-readable CLI command metadata")
    commands.add_argument("--json", action="store_true")
    commands.set_defaults(func=cmd_commands)

    reset = sub.add_parser("reset", help="request relationship reset with a 24-hour confirmation window")
    reset.add_argument("mode", choices=[item.value for item in ResetMode])
    reset.add_argument("--user", default="user")
    reset.add_argument("--ai", default="companion")
    reset.add_argument("--json", action="store_true")
    reset.set_defaults(func=cmd_reset)

    reset_request = sub.add_parser("reset-request", help="request reset with a 24-hour confirmation window")
    reset_request.add_argument("mode", choices=[item.value for item in ResetMode])
    reset_request.add_argument("--user", default="user")
    reset_request.add_argument("--ai", default="companion")
    reset_request.add_argument("--json", action="store_true")
    reset_request.set_defaults(func=cmd_reset_request)

    reset_confirm = sub.add_parser("reset-confirm", help="confirm a pending reset request")
    reset_confirm.add_argument("request_id")
    reset_confirm.add_argument("--force", action="store_true")
    reset_confirm.add_argument("--json", action="store_true")
    reset_confirm.set_defaults(func=cmd_reset_confirm)

    reset_cancel = sub.add_parser("reset-cancel", help="cancel a pending reset request")
    reset_cancel.add_argument("request_id")
    reset_cancel.add_argument("--json", action="store_true")
    reset_cancel.set_defaults(func=cmd_reset_cancel)

    browser = sub.add_parser("browser", help="show transparency memory browser snapshot")
    browser.add_argument("--user", default="user")
    browser.add_argument("--ai", default="companion")
    browser.add_argument("--limit", type=int, default=5)
    browser.add_argument("--json", action="store_true")
    browser.set_defaults(func=cmd_browser)

    export = sub.add_parser("export", help="export project state or user-facing relationship archives")
    export.add_argument("--output")
    export.add_argument("--format", choices=["json", "anonymous-json", "narrative", "milestones", "timeline"], default="json")
    export.add_argument("--anonymize", action="store_true")
    export.add_argument("--purpose", default="user_archive")
    export.add_argument("--user", default="user")
    export.add_argument("--ai", default="companion")
    export.set_defaults(func=cmd_export)

    mode = sub.add_parser("mode", help="switch assistant/friend/custom mode")
    mode.add_argument("mode", choices=[item.value for item in Mode])
    mode.add_argument("--user", default="user")
    mode.add_argument("--ai", default="companion")
    mode.add_argument("--profile-json", help="custom mode preference overrides as JSON")
    mode.add_argument("--reason", default="user_mode_switch")
    mode.add_argument("--json", action="store_true")
    mode.set_defaults(func=cmd_mode)

    pref = sub.add_parser("pref", help="set a relationship preference")
    pref.add_argument("key")
    pref.add_argument("value")
    pref.add_argument("--user", default="user")
    pref.add_argument("--ai", default="companion")
    pref.add_argument("--reason", default="user_preference")
    pref.add_argument("--json", action="store_true")
    pref.set_defaults(func=cmd_pref)

    decay_curve = sub.add_parser("decay-curve", help="set relationship-level memory decay curve")
    decay_curve.add_argument("curve", choices=[item.value for item in DecayCurve])
    decay_curve.add_argument("--user", default="user")
    decay_curve.add_argument("--ai", default="companion")
    decay_curve.add_argument("--reason", default="privacy_panel")
    decay_curve.add_argument("--json", action="store_true")
    decay_curve.set_defaults(func=cmd_decay_curve)

    custom_profile = sub.add_parser("custom-profile", help="update saved custom mode profile without switching modes")
    custom_profile.add_argument("profile_json")
    custom_profile.add_argument("--user", default="user")
    custom_profile.add_argument("--ai", default="companion")
    custom_profile.add_argument("--reason", default="user_custom_profile")
    custom_profile.add_argument("--json", action="store_true")
    custom_profile.set_defaults(func=cmd_custom_profile)

    active_feedback = sub.add_parser("active-feedback", help="record user feedback for an active recall event")
    active_feedback.add_argument("active_id")
    active_feedback.add_argument("reaction", choices=["accepted", "neutral", "ignored", "denied"])
    active_feedback.add_argument("--user", default="user")
    active_feedback.add_argument("--ai", default="companion")
    active_feedback.add_argument("--json", action="store_true")
    active_feedback.set_defaults(func=cmd_active_feedback)

    active_type_mute = sub.add_parser("active-type-mute", help="temporarily mute one active recall type")
    active_type_mute.add_argument("active_type", choices=sorted(FriendMemoryProject.USER_MUTABLE_ACTIVE_TYPES))
    active_type_mute.add_argument("--days", type=int, default=90)
    active_type_mute.add_argument("--reason", default="user_active_type_mute")
    active_type_mute.add_argument("--user", default="user")
    active_type_mute.add_argument("--ai", default="companion")
    active_type_mute.add_argument("--json", action="store_true")
    active_type_mute.set_defaults(func=cmd_active_type_mute)

    active_type_unmute = sub.add_parser("active-type-unmute", help="restore one muted active recall type")
    active_type_unmute.add_argument("active_type", choices=sorted(FriendMemoryProject.USER_MUTABLE_ACTIVE_TYPES))
    active_type_unmute.add_argument("--reason", default="user_active_type_unmute")
    active_type_unmute.add_argument("--user", default="user")
    active_type_unmute.add_argument("--ai", default="companion")
    active_type_unmute.add_argument("--json", action="store_true")
    active_type_unmute.set_defaults(func=cmd_active_type_unmute)

    implicit_topic_feedback = sub.add_parser("implicit-topic-feedback", help="record feedback for an inferred implicit topic")
    implicit_topic_feedback.add_argument("topic_id")
    implicit_topic_feedback.add_argument("reaction", choices=["accepted", "neutral", "ignored", "denied"])
    implicit_topic_feedback.add_argument("--user", default="user")
    implicit_topic_feedback.add_argument("--ai", default="companion")
    implicit_topic_feedback.add_argument("--json", action="store_true")
    implicit_topic_feedback.set_defaults(func=cmd_implicit_topic_feedback)

    transparency = sub.add_parser("transparency", help="show or acknowledge the AI relationship transparency notice")
    transparency.add_argument("--user", default="user")
    transparency.add_argument("--ai", default="companion")
    transparency.add_argument("--ack", action="store_true")
    transparency.add_argument("--json", action="store_true")
    transparency.set_defaults(func=cmd_transparency)

    ai_status = sub.add_parser("ai-status", help="show configured AI provider and recent decision trace")
    ai_status.add_argument("--user", default="user")
    ai_status.add_argument("--ai", default="companion")
    ai_status.add_argument("--all", action="store_true")
    ai_status.add_argument("--json", action="store_true")
    ai_status.set_defaults(func=cmd_ai_status)

    ai_probe = sub.add_parser("ai-probe", help="call the configured MemoryAI once without writing memory")
    ai_probe.add_argument("text", nargs="?", default="第一次一起庆祝成功，太开心了！")
    ai_probe.add_argument("--user", default="user")
    ai_probe.add_argument("--ai", default="companion")
    ai_probe.add_argument(
        "--require-external-ai",
        action="store_true",
        help="exit non-zero unless this probe observes an external HTTP worker or model",
    )
    ai_probe.add_argument("--json", action="store_true")
    ai_probe.set_defaults(func=cmd_ai_probe)

    ai_observation = sub.add_parser("ai-observation", help="export auditable external AI participation evidence")
    ai_observation.add_argument("--relationship", action="store_true", help="scope to the default --user/--ai relationship")
    ai_observation.add_argument("--relationship-id")
    ai_observation.add_argument("--user", default="user")
    ai_observation.add_argument("--ai", default="companion")
    ai_observation.add_argument("--output", help=f"write {AI_OBSERVATION_FILENAME} JSON to a file")
    ai_observation.add_argument("--json", action="store_true")
    ai_observation.set_defaults(func=cmd_ai_observation)

    memory_writes = sub.add_parser("memory-writes", help="pause or resume persistent relationship memory writes")
    memory_writes.add_argument(
        "enabled",
        type=lambda value: value.lower() in {"1", "true", "yes", "on", "enable", "enabled", "resume"},
    )
    memory_writes.add_argument("--reason", default="user_control")
    memory_writes.add_argument("--user", default="user")
    memory_writes.add_argument("--ai", default="companion")
    memory_writes.add_argument("--json", action="store_true")
    memory_writes.set_defaults(func=cmd_memory_writes)

    health = sub.add_parser("health", help="run relationship health and safety evaluation")
    health.add_argument("--user", default="user")
    health.add_argument("--ai", default="companion")
    health.add_argument("--json", action="store_true")
    health.set_defaults(func=cmd_health)

    guardian_summary = sub.add_parser("guardian-summary", help="generate a weekly guardian summary for a minor user")
    guardian_summary.add_argument("--user", default="user")
    guardian_summary.add_argument("--ai", default="companion")
    guardian_summary.add_argument("--start")
    guardian_summary.add_argument("--end")
    guardian_summary.add_argument("--json", action="store_true")
    guardian_summary.set_defaults(func=cmd_guardian_summary)

    consolidate = sub.add_parser("consolidate", help="run offline consolidation pipeline")
    consolidate.add_argument("--user", default="user")
    consolidate.add_argument("--ai", default="companion")
    consolidate.add_argument("--json", action="store_true")
    consolidate.set_defaults(func=cmd_consolidate)

    audit = sub.add_parser("audit", help="run project or relationship implementation integrity audit")
    audit.add_argument("--relationship", action="store_true", help="audit the default --user/--ai relationship instead of the whole project")
    audit.add_argument("--relationship-id")
    audit.add_argument("--user", default="user")
    audit.add_argument("--ai", default="companion")
    audit.add_argument("--json", action="store_true")
    audit.set_defaults(func=cmd_audit)

    decision_report = sub.add_parser("decision-report", help="show F-1/F-2/F-3 phase decision evidence")
    decision_report.add_argument(
        "--relationship",
        action="store_true",
        help="report the default --user/--ai relationship instead of the whole project",
    )
    decision_report.add_argument("--relationship-id")
    decision_report.add_argument("--user", default="user")
    decision_report.add_argument("--ai", default="companion")
    decision_report.add_argument("--run-benchmarks", action="store_true")
    decision_report.add_argument("--benchmark-iterations", type=int, default=20)
    decision_report.add_argument("--evidence-dir", help="load standard evidence JSON files from a directory")
    decision_report.add_argument("--manifest", help="verify evidence manifest JSON and attach integrity results")
    decision_report.add_argument("--evaluation-file", action="append")
    decision_report.add_argument(
        "--evaluation-task",
        action="append",
        default=None,
        choices=EVALUATION_TASKS,
    )
    decision_report.add_argument("--json", action="store_true")
    decision_report.set_defaults(func=cmd_decision_report)

    evaluate_labels = sub.add_parser("evaluate-labels", help="evaluate labelled validation data")
    evaluate_labels.add_argument("file")
    evaluate_labels.add_argument(
        "--task",
        default="stage_detection",
        choices=EVALUATION_TASKS,
    )
    evaluate_labels.add_argument("--json", action="store_true")
    evaluate_labels.set_defaults(func=cmd_evaluate_labels)

    evidence_template = sub.add_parser("evidence-template", help="print labelled evidence JSON template")
    evidence_template.add_argument("--task", default="stage_detection", choices=EVALUATION_TASKS)
    evidence_template.add_argument("--all", action="store_true", help="print templates for every evaluation task")
    evidence_template.add_argument("--output", help="write template JSON to a file")
    evidence_template.set_defaults(func=cmd_evidence_template)

    evidence_manifest_parser = sub.add_parser("evidence-manifest", help="create a sha256 manifest for an evidence directory")
    evidence_manifest_parser.add_argument("--evidence-dir", required=True, help="directory containing standard evidence JSON files")
    evidence_manifest_parser.add_argument("--output", help="write manifest JSON to a file")
    evidence_manifest_parser.set_defaults(func=cmd_evidence_manifest)

    evidence_validate = sub.add_parser("evidence-validate", help="validate a formal evidence directory")
    evidence_validate.add_argument("--evidence-dir", required=True, help="directory containing standard evidence JSON files")
    evidence_validate.add_argument("--manifest", help="verify an evidence manifest JSON while validating")
    evidence_validate.add_argument("--strict", action="store_true", help="exit non-zero unless every formal evidence item is ready")
    evidence_validate.add_argument("--json", action="store_true")
    evidence_validate.set_defaults(func=cmd_evidence_validate)

    release_gate = sub.add_parser("release-gate", help="run the formal release readiness gate")
    release_gate.add_argument("--evidence-dir", required=True, help="directory containing formal evidence JSON files")
    release_gate.add_argument("--manifest", help="verify an evidence manifest JSON before release readiness")
    release_gate.add_argument("--relationship", action="store_true", help="gate the default --user/--ai relationship instead of project scope")
    release_gate.add_argument("--relationship-id")
    release_gate.add_argument("--user", default="user")
    release_gate.add_argument("--ai", default="companion")
    release_gate.add_argument("--run-benchmarks", action="store_true")
    release_gate.add_argument("--benchmark-iterations", type=int, default=20)
    release_gate.add_argument("--require-external-ai", action="store_true")
    release_gate.add_argument("--strict", action="store_true", help="exit non-zero unless the release gate is ready")
    release_gate.add_argument("--json", action="store_true")
    release_gate.set_defaults(func=cmd_release_gate)

    release_bundle = sub.add_parser("release-bundle", help="write an auditable release evidence bundle")
    release_bundle.add_argument("--evidence-dir", required=True, help="directory containing formal evidence JSON files")
    release_bundle.add_argument("--output-dir", required=True, help="directory to write release bundle JSON reports")
    release_bundle.add_argument("--relationship", action="store_true", help="bundle the default --user/--ai relationship instead of project scope")
    release_bundle.add_argument("--relationship-id")
    release_bundle.add_argument("--user", default="user")
    release_bundle.add_argument("--ai", default="companion")
    release_bundle.add_argument("--run-benchmarks", action="store_true")
    release_bundle.add_argument("--benchmark-iterations", type=int, default=20)
    release_bundle.add_argument("--require-external-ai", action="store_true")
    release_bundle.add_argument("--json", action="store_true")
    release_bundle.set_defaults(func=cmd_release_bundle)

    stage_rollback = sub.add_parser("stage-rollback", help="rollback relationship stage to the previous or selected transition source")
    stage_rollback.add_argument("--relationship-id")
    stage_rollback.add_argument("--user", default="user")
    stage_rollback.add_argument("--ai", default="companion")
    stage_rollback.add_argument("--history-index", type=int)
    stage_rollback.add_argument("--reason", default="user_stage_rollback")
    stage_rollback.add_argument("--json", action="store_true")
    stage_rollback.set_defaults(func=cmd_stage_rollback)

    story_correct = sub.add_parser("story-correct", help="correct a shared story consensus while preserving the old version")
    story_correct.add_argument("story_id")
    story_correct.add_argument("consensus")
    story_correct.add_argument("--reason", default="user_correction")
    story_correct.add_argument("--json", action="store_true")
    story_correct.set_defaults(func=cmd_story_correct)

    story_confirm = sub.add_parser("story-confirm", help="mark a shared story consensus as user-confirmed")
    story_confirm.add_argument("story_id")
    story_confirm.add_argument("--note")
    story_confirm.add_argument("--json", action="store_true")
    story_confirm.set_defaults(func=cmd_story_confirm)

    story_rollback = sub.add_parser("story-rollback", help="roll back a shared story to a recorded narrative version")
    story_rollback.add_argument("story_id")
    story_rollback.add_argument("--version-index", type=int)
    story_rollback.add_argument("--reason", default="user_story_rollback")
    story_rollback.add_argument("--json", action="store_true")
    story_rollback.set_defaults(func=cmd_story_rollback)

    migrate = sub.add_parser("migrate", help="import legacy raw conversation logs into relationship memory")
    migrate.add_argument("input")
    migrate.add_argument("--format", choices=["json", "jsonl"], default="json")
    migrate.add_argument("--user", default="user")
    migrate.add_argument("--ai", default="companion")
    migrate.add_argument("--certificate", help="relationship migration certificate JSON")
    migrate.add_argument("--require-certificate", action="store_true")
    migrate.add_argument("--target-mode", choices=[item.value for item in Mode], help="mode to apply after migration")
    migrate.add_argument("--json", action="store_true")
    migrate.set_defaults(func=cmd_migrate)

    migration_cert = sub.add_parser("migration-cert", help="create a relationship migration certificate for legacy turns")
    migration_cert.add_argument("input")
    migration_cert.add_argument("--format", choices=["json", "jsonl"], default="json")
    migration_cert.add_argument("--user", default="user")
    migration_cert.add_argument("--ai", default="companion")
    migration_cert.set_defaults(func=cmd_migration_cert)

    migrate_rollback = sub.add_parser("migrate-rollback", help="rollback a migration batch within its 30-day window")
    migrate_rollback.add_argument("migration_id")
    migrate_rollback.set_defaults(func=cmd_migrate_rollback)

    health_ack = sub.add_parser("health-ack", help="acknowledge a health alert")
    health_ack.add_argument("alert_id")
    health_ack.add_argument("--note")
    health_ack.add_argument("--json", action="store_true")
    health_ack.set_defaults(func=cmd_health_ack)

    health_feedback = sub.add_parser("health-feedback", help="record accepted/ignored/rejected feedback for a health alert")
    health_feedback.add_argument("alert_id")
    health_feedback.add_argument("feedback", choices=["accepted", "ignored", "rejected"])
    health_feedback.add_argument("--note")
    health_feedback.add_argument("--json", action="store_true")
    health_feedback.set_defaults(func=cmd_health_feedback)

    age = sub.add_parser("age", help="set user age for minor protection rules")
    age.add_argument("age", type=int)
    age.add_argument("--user", default="user")
    age.add_argument("--ai", default="companion")
    age.set_defaults(func=cmd_age)

    minutes = sub.add_parser("minutes", help="record daily interaction minutes")
    minutes.add_argument("date")
    minutes.add_argument("minutes", type=int)
    minutes.add_argument("--user", default="user")
    minutes.add_argument("--ai", default="companion")
    minutes.set_defaults(func=cmd_minutes)

    inject = sub.add_parser("inject", help="manually inject a memory")
    inject.add_argument("text")
    inject.add_argument("--user", default="user")
    inject.add_argument("--ai", default="companion")
    inject.add_argument("--memory-type", default=MemoryType.SHARED_EPISODE.value, choices=[item.value for item in MemoryType])
    inject.add_argument("--context-tag", default=ContextTag.GENERAL.value, choices=[item.value for item in ContextTag])
    inject.add_argument("--milestone", action="store_true")
    inject.add_argument("--core-identity", action="store_true")
    inject.add_argument("--force-memory-write", action="store_true")
    inject.add_argument("--json", action="store_true")
    inject.set_defaults(func=cmd_inject)

    edit = sub.add_parser("edit", help="edit a memory while preserving version history")
    edit.add_argument("memory_id")
    edit.add_argument("text")
    edit.add_argument("--reason", default="user_edit")
    edit.set_defaults(func=cmd_edit)

    retag = sub.add_parser("retag", help="change a memory type/context tag while preserving tag history")
    retag.add_argument("memory_id")
    retag.add_argument("--memory-type", choices=[item.value for item in MemoryType])
    retag.add_argument("--context-tag", choices=[item.value for item in ContextTag])
    retag.add_argument("--reason", default="user_retag")
    retag.add_argument("--json", action="store_true")
    retag.set_defaults(func=cmd_retag)

    memory_suppress = sub.add_parser("memory-suppress", help="hide a memory from default retrieval and active recall without deleting it")
    memory_suppress.add_argument("memory_id")
    memory_suppress.add_argument("--reason", default="user_boundary")
    memory_suppress.add_argument("--json", action="store_true")
    memory_suppress.set_defaults(func=cmd_memory_suppress)

    memory_unsuppress = sub.add_parser("memory-unsuppress", help="restore a suppressed memory to retrieval and active recall")
    memory_unsuppress.add_argument("memory_id")
    memory_unsuppress.add_argument("--reason", default="user_boundary_removed")
    memory_unsuppress.add_argument("--json", action="store_true")
    memory_unsuppress.set_defaults(func=cmd_memory_unsuppress)

    memory_restore_archive = sub.add_parser("memory-restore-archive", help="restore a cold-archived memory to default retrieval")
    memory_restore_archive.add_argument("memory_id")
    memory_restore_archive.add_argument("--reason", default="user_restore_archive")
    memory_restore_archive.add_argument("--json", action="store_true")
    memory_restore_archive.set_defaults(func=cmd_memory_restore_archive)

    memory_verify = sub.add_parser("memory-verify", help="mark a memory as human-verified and raise confidence")
    memory_verify.add_argument("memory_id")
    memory_verify.add_argument("--reason", default="user_verified")
    memory_verify.add_argument("--json", action="store_true")
    memory_verify.set_defaults(func=cmd_memory_verify)

    memory_calibrate = sub.add_parser("memory-calibrate", help="record whether a memory confidence judgement was correct")
    memory_calibrate.add_argument("memory_id")
    memory_calibrate.add_argument("outcome", choices=["correct", "incorrect", "uncertain"])
    memory_calibrate.add_argument("--reason", default="user_calibration")
    memory_calibrate.add_argument("--json", action="store_true")
    memory_calibrate.set_defaults(func=cmd_memory_calibrate)

    retention_feedback = sub.add_parser("retention-feedback", help="calibrate reverse-decay retention from user feedback")
    retention_feedback.add_argument("memory_id")
    retention_feedback.add_argument("outcome", choices=["valuable", "over_retained", "under_retained", "stale"])
    retention_feedback.add_argument("--reason", default="user_retention_feedback")
    retention_feedback.add_argument("--json", action="store_true")
    retention_feedback.set_defaults(func=cmd_retention_feedback)

    time_conflict_resolve = sub.add_parser("time-conflict-resolve", help="resolve a detected source-time conflict after user clarification")
    time_conflict_resolve.add_argument("conflict_id")
    time_conflict_resolve.add_argument("resolution", choices=["prefer_memory", "both_valid", "uncertain"])
    time_conflict_resolve.add_argument("--preferred-memory-id")
    time_conflict_resolve.add_argument("--note")
    time_conflict_resolve.add_argument("--json", action="store_true")
    time_conflict_resolve.set_defaults(func=cmd_time_conflict_resolve)

    inside_joke_status = sub.add_parser("inside-joke-status", help="deactivate or reactivate an inside joke")
    inside_joke_status.add_argument("memory_id")
    inside_joke_status.add_argument("action", choices=["deactivate", "reactivate"])
    inside_joke_status.add_argument("--reason", default="user_inside_joke_control")
    inside_joke_status.add_argument("--json", action="store_true")
    inside_joke_status.set_defaults(func=cmd_inside_joke_status)

    thread_resolve = sub.add_parser("thread-resolve", help="mark an unresolved thread completed or no longer tracked")
    thread_resolve.add_argument("memory_id")
    thread_resolve.add_argument("resolution", choices=["completed", "no_longer_track", "muted"])
    thread_resolve.add_argument("--reason", default="user_thread_control")
    thread_resolve.add_argument("--json", action="store_true")
    thread_resolve.set_defaults(func=cmd_thread_resolve)

    milestone = sub.add_parser("mark-milestone", help="promote a memory to a protected milestone")
    milestone.add_argument("memory_id")
    milestone.set_defaults(func=cmd_mark_milestone)

    milestone_confirm = sub.add_parser("milestone-confirm", help="confirm or edit an auto-detected milestone")
    milestone_confirm.add_argument("memory_id")
    milestone_confirm.add_argument("--title")
    milestone_confirm.add_argument("--description")
    milestone_confirm.add_argument("--json", action="store_true")
    milestone_confirm.set_defaults(func=cmd_confirm_milestone)

    milestone_edit = sub.add_parser("milestone-edit", help="edit a relationship milestone title or description")
    milestone_edit.add_argument("memory_id")
    milestone_edit.add_argument("--title")
    milestone_edit.add_argument("--description")
    milestone_edit.add_argument("--json", action="store_true")
    milestone_edit.set_defaults(func=cmd_edit_milestone)

    milestone_reject = sub.add_parser("milestone-reject", help="reject an auto-detected milestone and downgrade it")
    milestone_reject.add_argument("memory_id")
    milestone_reject.add_argument("--reason", default="user_rejected")
    milestone_reject.add_argument("--json", action="store_true")
    milestone_reject.set_defaults(func=cmd_reject_milestone)

    downgrade = sub.add_parser("downgrade", help="downgrade a relationship memory back to the information layer")
    downgrade.add_argument("memory_id")
    downgrade.add_argument("--reason", default="user_downgrade")
    downgrade.set_defaults(func=cmd_downgrade)

    batch_downgrade = sub.add_parser("batch-downgrade", help="downgrade multiple relationship memories back to the information layer")
    batch_downgrade.add_argument("memory_ids", nargs="*")
    batch_downgrade.add_argument("--relationship-id")
    batch_downgrade.add_argument("--user", default="user")
    batch_downgrade.add_argument("--ai", default="companion")
    batch_downgrade.add_argument("--memory-type", choices=[item.value for item in MemoryType])
    batch_downgrade.add_argument("--context-tag", choices=[item.value for item in ContextTag])
    batch_downgrade.add_argument("--storage-layer", choices=[item.value for item in MemoryLayer])
    batch_downgrade.add_argument("--reason", default="user_batch_downgrade")
    batch_downgrade.add_argument("--json", action="store_true")
    batch_downgrade.set_defaults(func=cmd_batch_downgrade)

    reminders = sub.add_parser("reminders", help="check due or upcoming commitment reminders")
    reminders.add_argument("--relationship-id")
    reminders.add_argument("--user", default="user")
    reminders.add_argument("--ai", default="companion")
    reminders.add_argument("--window-days", type=int, default=1)
    reminders.add_argument("--json", action="store_true")
    reminders.set_defaults(func=cmd_reminders)

    reminder_complete = sub.add_parser("reminder-complete", help="mark a commitment reminder as completed")
    reminder_complete.add_argument("reminder_id")
    reminder_complete.set_defaults(func=cmd_reminder_complete)

    memory_delete_request = sub.add_parser("memory-delete-request", help="request delayed deletion of one memory and its derived traces")
    memory_delete_request.add_argument("memory_id")
    memory_delete_request.add_argument("--reason", default="user_delete")
    memory_delete_request.add_argument("--json", action="store_true")
    memory_delete_request.set_defaults(func=cmd_memory_delete_request)

    memory_delete_confirm = sub.add_parser("memory-delete-confirm", help="confirm a pending memory delete request after 24 hours")
    memory_delete_confirm.add_argument("request_id")
    memory_delete_confirm.add_argument("--force", action="store_true")
    memory_delete_confirm.add_argument("--json", action="store_true")
    memory_delete_confirm.set_defaults(func=cmd_memory_delete_confirm)

    memory_delete_cancel = sub.add_parser("memory-delete-cancel", help="cancel a pending memory delete request")
    memory_delete_cancel.add_argument("request_id")
    memory_delete_cancel.add_argument("--json", action="store_true")
    memory_delete_cancel.set_defaults(func=cmd_memory_delete_cancel)

    deletion_compliance = sub.add_parser("deletion-compliance", help="auditor-only access to sealed deletion compliance records")
    deletion_compliance.add_argument("--relationship-id")
    deletion_compliance.add_argument("--user", default="user")
    deletion_compliance.add_argument("--ai", default="companion")
    deletion_compliance.add_argument("--auditor-token", required=True)
    deletion_compliance.add_argument("--json", action="store_true")
    deletion_compliance.set_defaults(func=cmd_deletion_compliance)

    l4_delete_request = sub.add_parser("l4-delete-request", help="request delayed deletion of a protected L4 core identity")
    l4_delete_request.add_argument("identity_id")
    l4_delete_request.add_argument("--reason", default="user_delete")
    l4_delete_request.add_argument("--json", action="store_true")
    l4_delete_request.set_defaults(func=cmd_l4_delete_request)

    l4_delete_confirm = sub.add_parser("l4-delete-confirm", help="confirm an L4 core identity delete request after 24 hours")
    l4_delete_confirm.add_argument("request_id")
    l4_delete_confirm.add_argument("--force", action="store_true")
    l4_delete_confirm.add_argument("--json", action="store_true")
    l4_delete_confirm.set_defaults(func=cmd_l4_delete_confirm)

    l4_delete_cancel = sub.add_parser("l4-delete-cancel", help="cancel a pending L4 core identity delete request")
    l4_delete_cancel.add_argument("request_id")
    l4_delete_cancel.add_argument("--json", action="store_true")
    l4_delete_cancel.set_defaults(func=cmd_l4_delete_cancel)

    l4_review = sub.add_parser("l4-review", help="confirm or reject an L4 core identity review")
    l4_review.add_argument("identity_id")
    l4_review.add_argument("--decision", choices=["confirm", "reject", "needs_review"], default="confirm")
    l4_review.add_argument("--reason", default="user_confirmed")
    l4_review.add_argument("--json", action="store_true")
    l4_review.set_defaults(func=cmd_l4_review)
    return parser


def main() -> None:
    global STATE_PATH
    parser = build_parser()
    args = parser.parse_args()
    if args.state:
        STATE_PATH = Path(args.state)
    args.func(args)


if __name__ == "__main__":
    main()
