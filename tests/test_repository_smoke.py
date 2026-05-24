from __future__ import annotations

from pathlib import Path


def test_package_version():
    import hyperfitpro

    assert hyperfitpro.__version__ == "1.0.0"


def test_model_registry_loads():
    from hyperfitpro.core.registry import load_models

    models = load_models(include_inactive=True)
    active = [m for m in models if getattr(m, "active", True)]
    assert len(active) >= 42


def test_tool_plugin_count():
    from hyperfitpro.core.tool_plugins import discover_tool_plugins

    plugins = discover_tool_plugins()
    assert len(plugins) >= 50


def test_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "run_hyperfit_pro.py").exists()
    assert (root / "docs" / "GITHUB_RELEASE_CHECKLIST.md").exists()
    assert (root / ".github" / "workflows" / "ci.yml").exists()
