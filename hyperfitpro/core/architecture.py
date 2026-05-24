from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ConfigManager, AppSettings
from .plugin_manager import discover_plugin_model_classes, write_model_plugin_template, PluginRecord
from .tool_plugins import discover_tool_plugins, ToolPluginRecord
from .project import HyperFitProject, save_project, load_project, project_from_state
from .run_database import RunDatabase
from .undo_redo import UndoRedoStack


def architecture_status() -> dict[str, Any]:
    cfg = ConfigManager().load()
    diagnostics: list[PluginRecord] = []
    discover_plugin_model_classes(diagnostics=diagnostics)
    tool_diagnostics: list[ToolPluginRecord] = []
    discover_tool_plugins(diagnostics=tool_diagnostics)
    db = RunDatabase(cfg.run_database_path)
    return {
        'settings_file': str(ConfigManager().path),
        'run_database': str(db.path),
        'plugin_directories': cfg.plugin_directories,
        'loaded_plugin_models': [d.to_dict() for d in diagnostics if d.status == 'loaded'],
        'failed_plugins': [d.to_dict() for d in diagnostics if d.status != 'loaded'],
        'loaded_tool_plugins': [d.to_dict() for d in tool_diagnostics if d.status == 'loaded'],
        'failed_tool_plugins': [d.to_dict() for d in tool_diagnostics if d.status != 'loaded'],
        'recent_runs': db.list_runs(limit=10),
        'recent_projects': cfg.recent_projects,
    }
