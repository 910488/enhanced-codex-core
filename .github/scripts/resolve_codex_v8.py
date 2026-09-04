#!/usr/bin/env python3
"""Print the Codex-built V8 cargo environment for one target as KEY=VALUE lines.

`codex-code-mode-runtime` turns on the `v8` crate's `v8_enable_sandbox` feature,
so the prebuilt archive the crate asks for is the `ptrcomp_sandbox` variant.
denoland does not publish that variant -- no rusty_v8 release carries one, for
any target -- so a plain `cargo build -p codex-code-mode-host` 404s and the
crate tells you to set `V8_FROM_SOURCE=1`, which means depot_tools and hours of
V8 per platform. Codex builds and publishes that variant itself, and
`scripts/codex_package/v8.py` already knows how to fetch and verify it. This is
the CI-side entry point to that same code, for jobs that invoke cargo directly
instead of going through `just assemble-codex-package`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# `codex_package.targets` refuses to import without this, and computing it from
# this file's location is what keeps the workflow step down to one command.
os.environ.setdefault("CODEX_REPO_ROOT", str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from codex_package.targets import TARGET_SPECS  # noqa: E402
from codex_package.v8 import fetch_codex_v8_artifacts  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <target-triple>", file=sys.stderr)
        return 2

    target = argv[1]
    spec = TARGET_SPECS.get(target)
    if spec is None:
        supported = ", ".join(sorted(TARGET_SPECS))
        print(f"unsupported target {target}; expected one of {supported}", file=sys.stderr)
        return 2

    artifacts = fetch_codex_v8_artifacts(spec)
    print(f"RUSTY_V8_ARCHIVE={artifacts.archive}")
    print(f"RUSTY_V8_SRC_BINDING_PATH={artifacts.binding}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
