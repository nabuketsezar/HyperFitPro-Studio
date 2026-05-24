#!/usr/bin/env python
from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAST_MODULES = [
    "hyperfitpro.verification.verify_core",
    "hyperfitpro.verification.verify_model_library",
    "hyperfitpro.verification.verify_modern_gui_project",
]

FULL_MODULES = [
    "hyperfitpro.verification.verify_core",
    "hyperfitpro.verification.verify_model_library",
    "hyperfitpro.verification.verify_data_processing",
    "hyperfitpro.verification.verify_optimization",
    "hyperfitpro.verification.verify_fea_export",
    "hyperfitpro.verification.verify_architecture",
    "hyperfitpro.verification.verify_embedded_plugins",
    "hyperfitpro.verification.verify_tool_plugins",
    "hyperfitpro.verification.verify_50_plugins",
]


def run_module(module: str) -> None:
    print(f"\n[dev-check] python -m {module}")
    subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HyperFitPro developer checks.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fast", action="store_true", help="Run fast CI-friendly checks.")
    group.add_argument("--full", action="store_true", help="Run fuller local checks.")
    args = parser.parse_args()

    print("[dev-check] compileall")
    ok = compileall.compile_dir(str(ROOT / "hyperfitpro"), quiet=1)
    ok = compileall.compile_file(str(ROOT / "run_hyperfit_pro.py"), quiet=1) and ok
    if not ok:
        print("[dev-check] compile failed", file=sys.stderr)
        return 1

    modules = FULL_MODULES if args.full else FAST_MODULES
    for module in modules:
        run_module(module)

    print("\n[dev-check] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
