#!/usr/bin/env python3
"""Load .fastlane.yaml and run quick/deep checks with auto-detection fallback."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

CONFIG_PATH = Path('.fastlane.yaml')
GOVERNANCE_VERSION = 2


def parse_simple_yaml(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data: dict[str, str] = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.split('#', maxsplit=1)[0].strip()
        if not line or ':' not in line:
            continue
        key, value = line.split(':', maxsplit=1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def load_config() -> dict[str, str]:
    return parse_simple_yaml(CONFIG_PATH)


def governance_version(cfg: dict[str, str]) -> int | None:
    raw = cfg.get('governance_version', '').strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def make_target(name: str) -> bool:
    if not Path('Makefile').exists():
        return False
    result = subprocess.run(['make', '-n', name], capture_output=True)
    return result.returncode == 0


def run_shell(command: str) -> int:
    print(f'fastlane: running `{command}`')
    return subprocess.run(command, shell=True, check=False).returncode


def run_quick(cfg: dict[str, str]) -> int:
    explicit = cfg.get('quick_command', '').strip()
    if explicit:
        return run_shell(explicit)
    if make_target('quick'):
        return subprocess.run(['make', 'quick'], check=False).returncode
    if Path('Cargo.toml').exists():
        return subprocess.run(['cargo', 'check'], check=False).returncode
    if Path('package.json').exists():
        lint = subprocess.run(['npm', 'run', '-s', 'lint', '--if-present'], check=False).returncode
        test = subprocess.run(['npm', 'test', '--if-present', '--silent'], check=False).returncode
        return lint or test
    if Path('pubspec.yaml').exists():
        return subprocess.run(['flutter', 'analyze'], check=False).returncode
    if Path('pyproject.toml').exists() or Path('requirements.txt').exists():
        return subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=False).returncode
    print('fastlane: no quick targets; set quick_command in .fastlane.yaml or add Makefile quick.')
    return 0


def run_deep(cfg: dict[str, str]) -> int:
    explicit = cfg.get('deep_command', '').strip()
    if explicit:
        return run_shell(explicit)
    if make_target('ci'):
        return subprocess.run(['make', 'ci'], check=False).returncode
    if Path('Cargo.toml').exists():
        fmt = subprocess.run(['cargo', 'fmt', '--all', '--', '--check'], check=False).returncode
        clippy = subprocess.run(
            ['cargo', 'clippy', '--workspace', '--all-targets', '--', '-D', 'warnings'],
            check=False,
        ).returncode
        test = subprocess.run(['cargo', 'test', '--workspace'], check=False).returncode
        return fmt or clippy or test
    if Path('package.json').exists():
        build = subprocess.run(['npm', 'run', '-s', 'build', '--if-present'], check=False).returncode
        test = subprocess.run(['npm', 'test', '--if-present'], check=False).returncode
        return build or test
    if Path('pubspec.yaml').exists():
        return subprocess.run(['flutter', 'test'], check=False).returncode
    if Path('pyproject.toml').exists() or Path('requirements.txt').exists():
        return subprocess.run([sys.executable, '-m', 'pytest'], check=False).returncode
    print('fastlane: no deep targets; set deep_command in .fastlane.yaml or add Makefile ci.')
    return 0


def needs_rust(cfg: dict[str, str]) -> bool:
    stack = cfg.get('stack', '').lower()
    return stack == 'rust' or (not stack and Path('Cargo.toml').exists())


def needs_flutter(cfg: dict[str, str]) -> bool:
    stack = cfg.get('stack', '').lower()
    return stack == 'flutter' or (not stack and Path('pubspec.yaml').exists())


def doctor(cfg: dict[str, str]) -> int:
    version = governance_version(cfg)
    print(f'governance_version: {version if version is not None else "missing"} (expected {GOVERNANCE_VERSION})')
    print(f'quick_command: {cfg.get("quick_command") or "(auto)"}')
    print(f'deep_command: {cfg.get("deep_command") or "(auto)"}')
    print(f'stack: {cfg.get("stack") or "(auto)"}')
    for tool in ('git', 'python3'):
        print(f'{tool}: {shutil.which(tool) or "missing"}')
    if version != GOVERNANCE_VERSION:
        return 1
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print('usage: fastlane_config.py <quick|deep|version|doctor>', file=sys.stderr)
        return 2
    cmd = sys.argv[1]
    cfg = load_config()
    if cmd == 'quick':
        return run_quick(cfg)
    if cmd == 'deep':
        return run_deep(cfg)
    if cmd == 'version':
        version = governance_version(cfg)
        print(version if version is not None else 0)
        return 0
    if cmd == 'doctor':
        return doctor(cfg)
    print(f'unknown command: {cmd}', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())