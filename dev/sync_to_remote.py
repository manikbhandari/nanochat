#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import time
from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Change this:
REMOTE_DESTINATION = os.environ["REMOTE"]

# These folders and files remain local only.
EXCLUDES = [
    ".git/",
    ".vscode/",
    "node_modules/",
    ".venv/",
    "__pycache__/",
    "*.pyc",
    ".DS_Store",
]

POLL_INTERVAL_SECONDS = 0.75
DEBOUNCE_SECONDS = 0.25


def should_ignore(relative_path: Path) -> bool:
    ignored_directories = {
        ".git",
        ".vscode",
        "node_modules",
        ".venv",
        "__pycache__",
    }
    return any(part in ignored_directories for part in relative_path.parts)

def snapshot() -> dict[str, tuple[int, int]]:
    """Return file modification times and sizes for change detection."""
    files: dict[str, tuple[int, int]] = {}

    for path in PROJECT_DIR.rglob("*"):
        if not path.is_file():
            continue

        relative_path = path.relative_to(PROJECT_DIR)

        if should_ignore(relative_path):
            continue

        try:
            stat = path.stat()
        except FileNotFoundError:
            # A file may disappear while an editor is performing an atomic save.
            continue

        files[str(relative_path)] = (stat.st_mtime_ns, stat.st_size)

    return files


def sync() -> None:
    command = [
        "rsync",
        "-az",
        "--delete",
    ]

    for pattern in EXCLUDES:
        command.extend(["--exclude", pattern])

    # The trailing slash means: sync the contents of the project directory.
    command.extend([f"{PROJECT_DIR}/", REMOTE_DESTINATION])

    print("Syncing local changes to remote machine...", flush=True)

    result = subprocess.run(command, check=False)

    if result.returncode == 0:
        print("Sync complete.", flush=True)
    else:
        print(f"rsync exited with status {result.returncode}.", flush=True)


def main() -> None:
    print(f"Watching: {PROJECT_DIR}", flush=True)
    print(f"Remote:   {REMOTE_DESTINATION}", flush=True)

    sync()
    previous_snapshot = snapshot()

    while True:
        time.sleep(POLL_INTERVAL_SECONDS)
        current_snapshot = snapshot()

        if current_snapshot != previous_snapshot:
            # Allow editors and build tools to finish batches of rapid changes.
            time.sleep(DEBOUNCE_SECONDS)
            sync()
            previous_snapshot = snapshot()

if __name__ == "__main__":
    main()