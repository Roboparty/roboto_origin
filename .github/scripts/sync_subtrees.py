#!/usr/bin/env python3
"""Synchronize the fixed roboto_origin subtree snapshot allowlist.

The script deliberately fails closed: new .gitmodules entries, unknown gitlinks,
ordinary merge conflicts, out-of-scope paths, and unusually large deletions stop
the run before the workflow is allowed to push dev.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable


@dataclass(frozen=True)
class Entry:
    level: int
    name: str
    prefix: str
    repository: str
    branch: str
    mode: str
    legacy_prefix: str | None


@dataclass
class EntryReport:
    name: str
    path: str
    repository: str
    branch: str
    before: str
    upstream: str
    result: str = "pending"


class SyncError(RuntimeError):
    pass


class SnapshotSync:
    def __init__(self, repository: Path, manifest: Path, report_dir: Path) -> None:
        self.repository = repository.resolve()
        self.manifest = manifest.resolve()
        self.report_dir = report_dir.resolve()
        self.entries = self._load_manifest()
        self.base_sha = self.git("rev-parse", "HEAD", quiet=True).strip()
        self.remote_heads: dict[str, str] = {}
        self.records: dict[str, EntryReport] = {}

    def git(
        self,
        *args: str,
        check: bool = True,
        quiet: bool = False,
    ) -> str:
        command = ["git", "-C", str(self.repository), *args]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        output = completed.stdout
        if output and not quiet:
            print(output, end="" if output.endswith("\n") else "\n")
        if check and completed.returncode != 0:
            raise SyncError(
                f"Command failed ({completed.returncode}): {' '.join(command)}\n"
                f"{output[-4000:]}"
            )
        return output

    def _load_manifest(self) -> list[Entry]:
        entries: list[Entry] = []
        with self.manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if not row or row[0].startswith("#"):
                    continue
                if len(row) != 7:
                    raise SyncError(f"Invalid manifest row: {row!r}")
                level, name, prefix, repository, branch, mode, legacy = row
                entry = Entry(
                    level=int(level),
                    name=name,
                    prefix=prefix.rstrip("/"),
                    repository=repository,
                    branch=branch,
                    mode=mode,
                    legacy_prefix=None if legacy == "-" else legacy.rstrip("/"),
                )
                entries.append(entry)

        if not entries:
            raise SyncError("The subtree manifest is empty")

        names: set[str] = set()
        prefixes: set[str] = set()
        for entry in entries:
            path = PurePosixPath(entry.prefix)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "modules":
                raise SyncError(f"Unsafe snapshot prefix: {entry.prefix}")
            if entry.name in names or entry.prefix in prefixes:
                raise SyncError(f"Duplicate manifest entry: {entry.name} / {entry.prefix}")
            if entry.mode not in {"full", "squash"}:
                raise SyncError(f"Invalid subtree mode for {entry.name}: {entry.mode}")
            if not entry.repository.startswith("https://github.com/"):
                raise SyncError(f"Only public GitHub HTTPS repositories are allowed: {entry.name}")
            names.add(entry.name)
            prefixes.add(entry.prefix)

        return sorted(entries, key=lambda item: (item.level, item.name))

    @staticmethod
    def normalize_repository(value: str) -> str:
        normalized = value.strip().rstrip("/")
        if normalized.endswith(".git"):
            normalized = normalized[:-4]
        return normalized.lower()

    def parent_for(self, entry: Entry) -> Entry | None:
        candidates = [
            candidate
            for candidate in self.entries
            if candidate.level < entry.level
            and entry.prefix.startswith(candidate.prefix + "/")
        ]
        return max(candidates, key=lambda item: len(item.prefix), default=None)

    def children_for(self, entry: Entry) -> list[Entry]:
        return [
            candidate
            for candidate in self.entries
            if self.parent_for(candidate) == entry
        ]

    def validate_gitmodules(self, require_all: bool) -> None:
        expected: dict[str, Entry] = {}
        for entry in self.entries:
            if entry.level == 0:
                continue
            parent = self.parent_for(entry)
            if parent is None:
                raise SyncError(f"No manifest parent for {entry.name}")
            expected[entry.prefix] = entry

        tracked = self.git("ls-files", "-z", "modules", quiet=True).split("\0")
        gitmodule_files = sorted(path for path in tracked if path.endswith("/.gitmodules"))
        actual: dict[str, tuple[str, str | None]] = {}

        for relative_file in gitmodule_files:
            parser = configparser.RawConfigParser()
            parser.read(self.repository / relative_file, encoding="utf-8")
            parent_dir = PurePosixPath(relative_file).parent
            for section in parser.sections():
                if not section.startswith('submodule "'):
                    continue
                if not parser.has_option(section, "path") or not parser.has_option(section, "url"):
                    raise SyncError(f"Incomplete submodule declaration in {relative_file}: {section}")
                child_path = (parent_dir / parser.get(section, "path")).as_posix()
                branch = parser.get(section, "branch", fallback=None)
                actual[child_path] = (parser.get(section, "url"), branch)

        unknown = sorted(set(actual) - set(expected))
        if unknown:
            raise SyncError("Unknown submodule paths are not allowed: " + ", ".join(unknown))

        if require_all:
            missing = sorted(set(expected) - set(actual))
            if missing:
                raise SyncError("Expected submodule paths are missing: " + ", ".join(missing))

        for path, (repository, branch) in actual.items():
            entry = expected[path]
            if self.normalize_repository(repository) != self.normalize_repository(entry.repository):
                raise SyncError(
                    f"Repository mismatch for {path}: {repository} != {entry.repository}"
                )
            if branch and branch != entry.branch:
                raise SyncError(f"Branch mismatch for {path}: {branch} != {entry.branch}")

    def fetch_remote_heads(self) -> None:
        def fetch(entry: Entry) -> tuple[str, str]:
            completed = subprocess.run(
                ["git", "ls-remote", entry.repository, f"refs/heads/{entry.branch}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if completed.returncode != 0 or not completed.stdout.strip():
                raise SyncError(
                    f"Cannot resolve {entry.name} {entry.branch}: "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )
            return entry.name, completed.stdout.split()[0]

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(fetch, entry) for entry in self.entries]
            for future in as_completed(futures):
                name, sha = future.result()
                self.remote_heads[name] = sha

    def latest_splits(self) -> dict[str, str]:
        history = self.git("log", "HEAD", "--format=%B%x00", quiet=True)
        splits: dict[str, str] = {}
        for message in history.split("\0"):
            directories = [
                line.split(": ", 1)[1].rstrip("/")
                for line in message.splitlines()
                if line.startswith("git-subtree-dir: ")
            ]
            split_values = [
                line.split(": ", 1)[1]
                for line in message.splitlines()
                if line.startswith("git-subtree-split: ")
            ]
            if not split_values:
                continue
            for directory in directories:
                splits.setdefault(directory, split_values[0])
        return splits

    def commit_exists(self, sha: str) -> bool:
        completed = subprocess.run(
            ["git", "-C", str(self.repository), "cat-file", "-e", f"{sha}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.returncode == 0

    def is_ancestor(self, ancestor: str, descendant: str = "HEAD") -> bool:
        completed = subprocess.run(
            ["git", "-C", str(self.repository), "merge-base", "--is-ancestor", ancestor, descendant],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.returncode == 0

    def current_reference(self, entry: Entry, splits: dict[str, str]) -> tuple[str, bool]:
        remote_head = self.remote_heads[entry.name]
        if entry.mode == "full" and self.commit_exists(remote_head) and self.is_ancestor(remote_head):
            return remote_head, True

        current = splits.get(entry.prefix)
        if current:
            return current, current == remote_head
        if entry.legacy_prefix:
            legacy = splits.get(entry.legacy_prefix)
            if legacy:
                return legacy, legacy == remote_head
        return "unknown", False

    def safe_remove_worktree_path(self, relative: str) -> None:
        target = (self.repository / relative).resolve()
        modules_root = (self.repository / "modules").resolve()
        if target == modules_root or modules_root not in target.parents:
            raise SyncError(f"Refusing to remove unsafe path: {target}")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()

    def reseed_squashed_entry(self, entry: Entry) -> None:
        print(f"Re-seeding known squash subtree metadata: {entry.name}")
        self.git("rm", "-rf", "--ignore-unmatch", "--", entry.prefix)
        if (self.repository / entry.prefix).exists():
            self.safe_remove_worktree_path(entry.prefix)
        self.git("add", "-A", "--", entry.prefix)
        staged = self.git("diff", "--cached", "--name-only", quiet=True).strip()
        if staged:
            self.git("commit", "-m", f"chore: prepare {entry.name} snapshot refresh")
        self.subtree_add(entry)

    def subtree_add(self, entry: Entry) -> None:
        args = [
            "subtree",
            "add",
            f"--prefix={entry.prefix}",
            entry.repository,
            entry.branch,
        ]
        if entry.mode == "squash":
            args.append("--squash")
        self.git(*args)

    def conflict_paths(self) -> list[str]:
        output = self.git("diff", "--name-only", "--diff-filter=U", "-z", quiet=True)
        return [path for path in output.split("\0") if path]

    def resolve_known_gitlink_conflict(self, entry: Entry) -> None:
        merge_head = self.repository / ".git" / "MERGE_HEAD"
        if not merge_head.exists():
            # Worktrees store gitdir in a file; rev-parse works in both layouts.
            git_dir = Path(self.git("rev-parse", "--git-dir", quiet=True).strip())
            if not git_dir.is_absolute():
                git_dir = self.repository / git_dir
            merge_head = git_dir / "MERGE_HEAD"
        if not merge_head.exists():
            raise SyncError(f"Subtree pull failed for {entry.name} without a merge to resolve")

        children = self.children_for(entry)
        module_file = f"{entry.prefix}/.gitmodules"
        conflicts = self.conflict_paths()

        def allowed(path: str) -> bool:
            if path == module_file:
                return True
            return any(path == child.prefix or path.startswith(child.prefix + "~") for child in children)

        unknown = [path for path in conflicts if not allowed(path)]
        if not conflicts or unknown:
            raise SyncError(
                f"Non-gitlink conflicts in {entry.name}: " + ", ".join(unknown or conflicts)
            )

        print(f"Resolving allowlisted gitlink conflicts for {entry.name}")
        if module_file in conflicts:
            self.git("checkout", "--theirs", "--", module_file)

        for child in children:
            self.git("rm", "-rf", "--ignore-unmatch", "--", child.prefix)
            if (self.repository / child.prefix).exists():
                self.safe_remove_worktree_path(child.prefix)

        indexed = self.git("ls-files", "-s", "-z", "--", entry.prefix, quiet=True)
        marker_paths: set[str] = set()
        for item in indexed.split("\0"):
            if "\t" not in item:
                continue
            path = item.split("\t", 1)[1]
            if any(path.startswith(child.prefix + "~") for child in children):
                marker_paths.add(path)
        for marker in sorted(marker_paths):
            self.git("rm", "-rf", "--ignore-unmatch", "--", marker)
            if (self.repository / marker).exists():
                self.safe_remove_worktree_path(marker)

        self.git("add", "-A", "--", entry.prefix)
        remaining = self.conflict_paths()
        if remaining:
            raise SyncError(f"Unresolved paths remain in {entry.name}: {', '.join(remaining)}")
        self.git("commit", "-m", f"chore: sync {entry.name} snapshot")
        self.validate_gitmodules(require_all=False)

    def subtree_pull(self, entry: Entry) -> None:
        args = [
            "subtree",
            "pull",
            f"--prefix={entry.prefix}",
            entry.repository,
            entry.branch,
        ]
        if entry.mode == "squash":
            args.append("--squash")
        command = ["git", "-C", str(self.repository), *args]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.returncode != 0:
            self.resolve_known_gitlink_conflict(entry)

    def sync_entry(self, entry: Entry, before: str) -> None:
        path = self.repository / entry.prefix
        if not path.exists():
            self.subtree_add(entry)
            return

        if entry.mode == "squash" and before == "unknown":
            self.reseed_squashed_entry(entry)
            return

        self.subtree_pull(entry)

    def validate_final_state(self) -> tuple[int, int, int, str]:
        self.validate_gitmodules(require_all=True)

        unmerged = self.conflict_paths()
        if unmerged:
            raise SyncError("Unresolved merge paths: " + ", ".join(unmerged))

        gitlinks_output = self.git("ls-files", "-s", "-z", "modules", quiet=True)
        gitlinks: list[str] = []
        for item in gitlinks_output.split("\0"):
            if not item or "\t" not in item:
                continue
            metadata, path = item.split("\t", 1)
            if metadata.split()[0] == "160000":
                gitlinks.append(path)
        if gitlinks:
            raise SyncError("Gitlinks remain after synchronization: " + ", ".join(gitlinks))

        porcelain = self.git("status", "--porcelain", quiet=True).strip()
        if porcelain:
            raise SyncError("The synchronization left an uncommitted working tree:\n" + porcelain)

        changed_paths = [
            path
            for path in self.git("diff", "--name-only", f"{self.base_sha}..HEAD", quiet=True).splitlines()
            if path
        ]
        outside = [path for path in changed_paths if not path.startswith("modules/")]
        if outside:
            raise SyncError("Snapshot changed paths outside modules/: " + ", ".join(outside))

        statuses = self.git("diff", "--name-status", f"{self.base_sha}..HEAD", quiet=True).splitlines()
        deleted = sum(1 for line in statuses if line.startswith("D\t"))
        max_changed = int(os.environ.get("MAX_CHANGED_FILES", "5000"))
        max_deleted = int(os.environ.get("MAX_DELETED_FILES", "500"))
        if len(changed_paths) > max_changed or deleted > max_deleted:
            raise SyncError(
                f"Change-size guard failed: {len(changed_paths)} changed files, "
                f"{deleted} deleted files (limits {max_changed}/{max_deleted})"
            )

        additions = 0
        deletions = 0
        for line in self.git("diff", "--numstat", f"{self.base_sha}..HEAD", quiet=True).splitlines():
            added, removed, _ = line.split("\t", 2)
            if added.isdigit():
                additions += int(added)
            if removed.isdigit():
                deletions += int(removed)

        diff_stat = self.git("diff", "--stat", f"{self.base_sha}..HEAD", quiet=True).strip()
        return len(changed_paths), deleted, additions + deletions, diff_stat

    def write_report(
        self,
        status: str,
        error: str | None,
        metrics: tuple[int, int, int, str] | None = None,
    ) -> dict[str, object]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        final_sha = self.git("rev-parse", "HEAD", quiet=True).strip()
        changed_files, deleted_files, changed_lines, diff_stat = metrics or (0, 0, 0, "")
        changed = status == "success" and final_sha != self.base_sha
        updated = [record.name for record in self.records.values() if record.result == "updated"]
        report: dict[str, object] = {
            "status": status,
            "changed": changed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "base_sha": self.base_sha,
            "final_sha": final_sha,
            "changed_files": changed_files,
            "deleted_files": deleted_files,
            "changed_lines": changed_lines,
            "updated": updated,
            "error": error,
            "diff_stat": diff_stat,
            "entries": [asdict(self.records[entry.name]) for entry in self.entries],
        }
        (self.report_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        status_text = {"success": "成功", "failed": "失败"}.get(status, status)
        lines = [
            "# roboto_origin 每日快照报告",
            "",
            f"- 状态：{status_text}",
            f"- 是否有改动：{'是' if changed else '否'}",
            f"- 基准提交：`{self.base_sha[:12]}`",
            f"- 最终提交：`{final_sha[:12]}`",
            f"- 更新仓库：{', '.join(updated) if updated else '无'}",
            f"- 文件统计：{changed_files} 个文件，{deleted_files} 个删除，{changed_lines} 行增删",
        ]
        if error:
            lines.extend(["", "## 错误", "", error[:2000]])
        lines.extend(["", "## 白名单检查", "", "| 仓库 | 路径 | 上游 | 结果 |", "| --- | --- | --- | --- |"])
        for entry in self.entries:
            record = self.records[entry.name]
            lines.append(
                f"| {record.name} | `{record.path}` | `{record.upstream[:12]}` | {record.result} |"
            )
        if diff_stat:
            lines.extend(["", "## Git 统计", "", "```text", diff_stat, "```"])
        (self.report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report

    def execute(self) -> int:
        error: str | None = None
        metrics: tuple[int, int, int, str] | None = None
        status = "failed"
        try:
            if self.git("status", "--porcelain", quiet=True).strip():
                raise SyncError("The snapshot worktree must be clean before synchronization")
            self.validate_gitmodules(require_all=True)
            self.fetch_remote_heads()
            splits = self.latest_splits()

            for entry in self.entries:
                remote_head = self.remote_heads[entry.name]
                before, current = self.current_reference(entry, splits)
                self.records[entry.name] = EntryReport(
                    name=entry.name,
                    path=entry.prefix,
                    repository=entry.repository,
                    branch=entry.branch,
                    before=before,
                    upstream=remote_head,
                )
                if current:
                    self.records[entry.name].result = "unchanged"
                    print(f"[unchanged] {entry.name} {remote_head[:12]}")
                    continue

                print(f"[sync] {entry.name}: {before[:12]} -> {remote_head[:12]}")
                self.sync_entry(entry, before)
                self.records[entry.name].result = "updated"
                self.validate_gitmodules(require_all=False)
                splits = self.latest_splits()

            metrics = self.validate_final_state()
            status = "success"
        except Exception as exc:  # The report must survive every fail-closed path.
            error = str(exc)
            print(f"ERROR: {error}", file=sys.stderr)
            for entry in self.entries:
                if entry.name not in self.records:
                    self.records[entry.name] = EntryReport(
                        name=entry.name,
                        path=entry.prefix,
                        repository=entry.repository,
                        branch=entry.branch,
                        before="unknown",
                        upstream=self.remote_heads.get(entry.name, "unknown"),
                    )
                elif self.records[entry.name].result == "pending":
                    self.records[entry.name].result = "failed"

        self.write_report(status, error, metrics)
        return 0 if status == "success" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return SnapshotSync(args.repository, args.manifest, args.report_dir).execute()


if __name__ == "__main__":
    raise SystemExit(main())
