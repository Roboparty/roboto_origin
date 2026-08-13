#!/usr/bin/env python3
"""Send a signed Feishu group card for real subtree file changes only."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


MAX_FILES_PER_REPOSITORY = 8
MAX_REPOSITORIES = 12
MAX_PAYLOAD_BYTES = 19_000


def load_report(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError("GitHub Actions did not produce a synchronization report")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise RuntimeError("The synchronization report must be a JSON object")
    return report


def sign(timestamp: int, secret: str) -> str:
    string_to_sign = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(string_to_sign, digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def validate_webhook(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    path_parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "https"
        or parsed.hostname != "open.feishu.cn"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or path_parts[:4] != ["open-apis", "bot", "v2", "hook"]
        or len(path_parts) != 5
        or not path_parts[4]
    ):
        raise RuntimeError("FEISHU_GROUP_WEBHOOK is not an allowed Feishu v2 bot webhook")
    return value


def github_url(value: str) -> str | None:
    if not value:
        return None
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme == "https"
        and parsed.hostname == "github.com"
        and parsed.port is None
        and parsed.username is None
        and parsed.password is None
    ):
        return value
    return None


def feishu_document_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "roboparty.feishu.cn"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.startswith("/wiki/")
        or parsed.fragment
    ):
        raise RuntimeError("FEISHU_DOCUMENT_URL is not an allowed Roboparty Feishu wiki URL")
    return value


def inline(value: object, limit: int = 180) -> str:
    text = " ".join(str(value).replace("`", "ʼ").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def file_line(repository: dict, file_change: dict) -> str:
    path = str(file_change.get("path", ""))
    prefix = str(repository.get("path", "")).rstrip("/") + "/"
    if path.startswith(prefix):
        path = path[len(prefix) :]
    additions = file_change.get("additions")
    deletions = file_change.get("deletions")
    line_stat = "二进制" if additions is None or deletions is None else f"+{additions} / -{deletions}"
    return f"• `{inline(file_change.get('status', '?'), 12)}` `{inline(path, 240)}`（{line_stat}）"


def build_card(
    report: dict,
    repository_limit: int = MAX_REPOSITORIES,
    file_limit: int = MAX_FILES_PER_REPOSITORY,
) -> dict:
    repositories = list(report.get("changed_repositories") or [])
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    document_url = feishu_document_url(os.environ.get("FEISHU_DOCUMENT_URL", ""))

    summary = (
        f"**更新时间：** {now:%Y-%m-%d %H:%M:%S}（Asia/Shanghai）\n"
        f"**实际改动仓库：** {len(repositories)} 个\n"
        f"**文件统计：** {report.get('changed_files', 0)} 个文件，"
        f"{report.get('deleted_files', 0)} 个删除，"
        f"+{report.get('additions', 0)} / -{report.get('deletions', 0)}\n"
        "**同步方向：** 子仓库 → `dev`；`main` 仍需人工审核"
    )
    elements: list[dict] = [
        {"tag": "div", "text": {"tag": "lark_md", "content": summary}},
        {"tag": "hr"},
    ]

    visible_repositories = repositories[:repository_limit]
    for repository in visible_repositories:
        before = inline(str(repository.get("before", "unknown"))[:12], 12)
        upstream = inline(str(repository.get("upstream", "unknown"))[:12], 12)
        lines = [
            f"**🔔 发生改动：{inline(repository.get('name', 'unknown'), 120)}**",
            f"分支：`{inline(repository.get('branch', 'unknown'), 80)}`　提交：`{before}` → `{upstream}`",
            (
                f"统计：{repository.get('changed_files', 0)} 个文件，"
                f"{repository.get('deleted_files', 0)} 个删除，"
                f"+{repository.get('additions', 0)} / -{repository.get('deletions', 0)}"
            ),
        ]
        files = list(repository.get("files") or [])
        lines.extend(file_line(repository, file_change) for file_change in files[:file_limit])
        if len(files) > file_limit:
            lines.append(f"• 另有 {len(files) - file_limit} 个文件未在卡片中展开")
        compare_url = github_url(str(repository.get("compare_url") or ""))
        if compare_url:
            lines.append(f"[查看该仓库上游差异]({compare_url})")
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(lines)}})

    if len(repositories) > repository_limit:
        elements.append(
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": f"另有 {len(repositories) - repository_limit} 个变化仓库未在卡片中展开，请查看飞书文档。",
                    }
                ],
            }
        )
    elements.append(
        {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": "未变化仓库不展示；完整详情已写入飞书文档。",
                }
            ],
        }
    )

    elements.append(
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "前往飞书文档"},
                    "url": document_url,
                    "type": "primary",
                }
            ],
        }
    )

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": "🔔 roboto_origin 子仓库快照有更新"},
        },
        "elements": elements,
    }


def build_payload(report: dict, timestamp: int, signing_secret: str) -> dict:
    limits = (
        (MAX_REPOSITORIES, MAX_FILES_PER_REPOSITORY),
        (10, 6),
        (8, 5),
        (6, 4),
        (4, 3),
        (2, 2),
        (1, 1),
        (0, 0),
    )
    for repository_limit, file_limit in limits:
        payload = {
            "timestamp": str(timestamp),
            "sign": sign(timestamp, signing_secret),
            "msg_type": "interactive",
            "card": build_card(report, repository_limit, file_limit),
        }
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        if size <= MAX_PAYLOAD_BYTES:
            return payload
    raise RuntimeError("Unable to fit the Feishu group card within the payload limit")


def send(webhook: str, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        webhook,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Feishu group bot HTTP {exc.code}: {detail}") from exc

    code = result.get("code", result.get("StatusCode", -1))
    if code != 0:
        message = result.get("msg", result.get("StatusMessage", "unknown error"))
        raise RuntimeError(f"Feishu group bot error {code}: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    report = load_report(args.report)
    repositories = list(report.get("changed_repositories") or [])
    if report.get("status") != "success" or not report.get("changed") or not repositories:
        print("No successful repository file changes; group notification skipped")
        return 0

    webhook = validate_webhook(os.environ.get("FEISHU_GROUP_WEBHOOK", ""))
    signing_secret = os.environ.get("FEISHU_GROUP_SIGNING_SECRET", "")
    if not signing_secret:
        raise RuntimeError("FEISHU_GROUP_SIGNING_SECRET is missing")

    payload = build_payload(report, int(time.time()), signing_secret)
    send(webhook, payload)
    print(f"Sent one Feishu group card for {len(repositories)} changed repositories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
