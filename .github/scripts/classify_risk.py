#!/usr/bin/env python3
"""
Classify PR / push risk for fastlane governance.
Uses git diff vs base branch when available (CI), else HEAD~1..HEAD locally.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG_PATH = Path(".github/risk-config.json")
DOCS_SUFFIXES = {".md", ".mdx", ".rst", ".txt"}
DOCS_NAMES = {
    "readme",
    "changelog",
    "license",
    "licence",
    "contributing",
    "authors",
    "code_of_conduct",
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print("Missing .github/risk-config.json — defaulting to low risk", file=sys.stderr)
        return {
            "highRiskThreshold": 5,
            "fileCountHighRisk": 10,
            "criticalPathPatterns": [],
            "dependencyFiles": [],
        }
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def matches_critical_path(path: str, pattern: str) -> bool:
    normalized = path.lower().replace("\\", "/")
    needle = pattern.lower().strip("/")
    if not needle:
        return False
    segments = [segment for segment in normalized.split("/") if segment]
    for segment in segments:
        if segment == needle or Path(segment).stem == needle:
            return True
    return False


def is_docs_only(files: list[str]) -> bool:
    if not files:
        return False
    for path in files:
        normalized = path.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        if not parts:
            return False
        if parts[0] == "docs":
            continue
        name = parts[-1].lower()
        stem = Path(name).stem.lower()
        suffix = Path(name).suffix.lower()
        if suffix in DOCS_SUFFIXES or stem in DOCS_NAMES:
            continue
        return False
    return True


def changed_files() -> list[str]:
    event_name = os.getenv("GITHUB_EVENT_NAME", "")
    base_ref = os.getenv("FASTLANE_BASE_REF", "")
    push_before = os.getenv("FASTLANE_PUSH_BEFORE", "")
    push_after = os.getenv("FASTLANE_PUSH_AFTER", "")

    if (
        push_before
        and push_after
        and event_name == "push"
        and push_before != "0000000000000000000000000000000000000000"
    ):
        diff = subprocess.run(
            ["git", "diff", "--name-only", f"{push_before}...{push_after}"],
            capture_output=True,
            text=True,
        )
        if diff.returncode != 0:
            return []
        return [x.strip() for x in diff.stdout.splitlines() if x.strip()]

    if event_name == "pull_request" and base_ref:
        base = f"origin/{base_ref}"
        merge_base = subprocess.run(
            ["git", "merge-base", base, "HEAD"], capture_output=True, text=True
        )
        if merge_base.returncode != 0 or not merge_base.stdout.strip():
            p = subprocess.run(
                ["git", "diff", "--name-only", f"{base}...HEAD"], capture_output=True, text=True
            )
            if p.returncode != 0:
                return []
            return [x.strip() for x in p.stdout.splitlines() if x.strip()]

        mb = merge_base.stdout.strip()
        d = subprocess.run(
            ["git", "diff", "--name-only", f"{mb}...HEAD"],
            capture_output=True,
            text=True,
        )
        if d.returncode != 0:
            return []
        return [x.strip() for x in d.stdout.splitlines() if x.strip()]

    p = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1..HEAD"],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        return []
    return [x.strip() for x in p.stdout.splitlines() if x.strip()]


def classify(files: list[str], cfg: dict) -> tuple[str, int]:
    if is_docs_only(files):
        return ("low", 0)

    score = 0
    if len(files) > cfg.get("fileCountHighRisk", 10):
        score += 5

    dep_files = set(cfg.get("dependencyFiles", []))

    for f in files:
        name = Path(f).name
        if name in dep_files:
            score += 3
        for pattern in cfg.get("criticalPathPatterns", []):
            if matches_critical_path(f, pattern):
                score += 2
                break

    threshold = cfg.get("highRiskThreshold", 5)
    return ("high" if score >= threshold else "low", score)


def main() -> None:
    cfg = load_config()
    files = changed_files()
    risk, score = classify(files, cfg)
    out = os.getenv("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(f"risk={risk}\n")
    print(f"Risk={risk} Score={score} Files={len(files)}")


if __name__ == "__main__":
    main()