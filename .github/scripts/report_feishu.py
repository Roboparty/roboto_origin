#!/usr/bin/env python3
"""Append one deterministic snapshot result to a Feishu docx document."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


FEISHU_API = "https://open.feishu.cn/open-apis"


def request_json(url: str, method: str = "GET", payload: object | None = None, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Feishu HTTP {exc.code}: {detail}") from exc
    if result.get("code", 0) != 0:
        raise RuntimeError(f"Feishu API error {result.get('code')}: {result.get('msg')}")
    return result


def rich_text(content: str) -> dict:
    return {
        "elements": [
            {
                "text_run": {
                    "content": content[:5000],
                    "text_element_style": {},
                }
            }
        ],
        "style": {},
    }


def text_block(content: str) -> dict:
    return {"block_type": 2, "text": rich_text(content)}


def heading_block(content: str) -> dict:
    return {"block_type": 4, "heading2": rich_text(content)}


def load_report(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "failed",
        "changed": False,
        "updated": [],
        "changed_files": 0,
        "deleted_files": 0,
        "changed_lines": 0,
        "base_sha": "unknown",
        "final_sha": "unknown",
        "error": "GitHub Actions 未生成同步报告。",
    }


def build_blocks(report: dict) -> list[dict]:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    workflow_result = os.environ.get("WORKFLOW_RESULT", "unknown")
    if workflow_result != "success":
        report["status"] = "failed"
        if not report.get("error"):
            report["error"] = f"GitHub 同步任务结果：{workflow_result}"

    status = report.get("status", "failed")
    changed = bool(report.get("changed"))
    updated = list(report.get("updated") or [])
    status_text = "✅ 同步成功" if status == "success" else "❌ 同步失败"
    if status == "success" and not changed:
        status_text = "ℹ️ 今日无改动"

    run_url = (
        f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
        f"{os.environ.get('GITHUB_REPOSITORY', 'Roboparty/roboto_origin')}/actions/runs/"
        f"{os.environ.get('GITHUB_RUN_ID', '')}"
    )
    pr_url = os.environ.get("PR_URL", "").strip()
    base_sha = str(report.get("base_sha", "unknown"))[:12]
    final_sha = str(report.get("final_sha", "unknown"))[:12]

    blocks = [
        heading_block(f"{now:%Y-%m-%d} 自动快照报告"),
        text_block(f"状态：{status_text}"),
        text_block(f"时间：{now:%Y-%m-%d %H:%M:%S}（Asia/Shanghai）"),
        text_block(
            "统计："
            f"{report.get('changed_files', 0)} 个文件，"
            f"{report.get('deleted_files', 0)} 个删除，"
            f"{report.get('changed_lines', 0)} 行增删"
        ),
        text_block(f"快照提交：{base_sha} → {final_sha}"),
        text_block(f"更新仓库：{', '.join(updated) if updated else '无'}"),
        text_block(f"GitHub Actions：{run_url}"),
    ]
    if pr_url:
        blocks.append(text_block(f"dev → main PR：{pr_url}"))
    if report.get("error"):
        blocks.append(text_block(f"错误：{str(report['error'])[:1500]}"))
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    app_id = os.environ.get("FEISHU_APP_ID", "")
    app_secret = os.environ.get("FEISHU_APP_SECRET", "")
    document_id = os.environ.get("FEISHU_DOCUMENT_ID", "")
    if not all((app_id, app_secret, document_id)):
        print("Feishu document secrets are incomplete", file=sys.stderr)
        return 1

    auth = request_json(
        f"{FEISHU_API}/auth/v3/tenant_access_token/internal",
        method="POST",
        payload={"app_id": app_id, "app_secret": app_secret},
    )
    token = auth.get("tenant_access_token")
    if not token:
        raise RuntimeError("Feishu authentication returned no tenant_access_token")

    report = load_report(args.report)
    blocks = build_blocks(report)
    result = request_json(
        f"{FEISHU_API}/docx/v1/documents/{document_id}/blocks/{document_id}/children",
        method="POST",
        payload={"children": blocks, "index": -1},
        token=token,
    )
    created = len(result.get("data", {}).get("children", []))
    print(f"Appended {created} report blocks to the Feishu document")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
