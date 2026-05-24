from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any
import json
import os


APP_DIR = Path.home() / '.hyperfitpro_studio'
DEFAULT_CONFIG_FILE = APP_DIR / 'settings.json'
DEFAULT_PLUGIN_DIR = APP_DIR / 'plugins'
DEFAULT_PROJECT_DIR = APP_DIR / 'projects'
DEFAULT_RUN_DB = APP_DIR / 'run_history.sqlite3'


@dataclass
class AppSettings:
    """Persistent application configuration.

    Stored as JSON to avoid adding a YAML dependency. The file is intentionally
    human-readable and versioned so future migrations can be implemented cleanly.
    """
    schema_version: int = 1
    default_working_directory: str = str(Path('C:/HyperFitPro_Runs') if os.name == 'nt' else Path.home() / 'HyperFitPro_Runs')
    plugin_directories: list[str] = field(default_factory=lambda: [str(DEFAULT_PLUGIN_DIR)])
    run_database_path: str = str(DEFAULT_RUN_DB)
    default_optimizer: str = 'recommended_adaptive_bounds'
    default_max_evaluations: int = 8000
    default_regularization_type: str = 'l2'
    default_validation_fraction: float = 0.10
    default_parallel_workers: int = 1
    report_template: str = 'engineering_default'
    export_unit_system: str = 'MPa-mm-N'
    autosave_project: bool = True
    autosave_interval_s: int = 120
    recent_projects: list[str] = field(default_factory=list)
    recent_working_directories: list[str] = field(default_factory=list)

    def normalize(self) -> 'AppSettings':
        self.plugin_directories = [str(Path(p).expanduser()) for p in self.plugin_directories if str(p).strip()]
        if str(DEFAULT_PLUGIN_DIR) not in self.plugin_directories:
            self.plugin_directories.append(str(DEFAULT_PLUGIN_DIR))
        self.default_max_evaluations = max(1, int(self.default_max_evaluations))
        self.default_parallel_workers = max(1, int(self.default_parallel_workers))
        self.default_validation_fraction = min(max(float(self.default_validation_fraction), 0.0), 0.9)
        self.autosave_interval_s = max(10, int(self.autosave_interval_s))
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'AppSettings':
        base = cls()
        for key, value in (data or {}).items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base.normalize()


class ConfigManager:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path else DEFAULT_CONFIG_FILE

    def load(self) -> AppSettings:
        if not self.path.exists():
            settings = AppSettings().normalize()
            self.save(settings)
            return settings
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            return AppSettings.from_dict(data)
        except Exception:
            # Do not crash the engineering workflow because of a corrupted settings file.
            backup = self.path.with_suffix(self.path.suffix + '.corrupt')
            try:
                self.path.replace(backup)
            except Exception:
                pass
            settings = AppSettings().normalize()
            self.save(settings)
            return settings

    def save(self, settings: AppSettings) -> Path:
        settings.normalize()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(settings.to_dict(), indent=2), encoding='utf-8')
        DEFAULT_PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_PROJECT_DIR.mkdir(parents=True, exist_ok=True)
        return self.path

    def update(self, **changes: Any) -> AppSettings:
        settings = self.load()
        for key, value in changes.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        self.save(settings)
        return settings

    def add_recent_project(self, project_path: str | Path, limit: int = 12) -> AppSettings:
        settings = self.load()
        p = str(Path(project_path).expanduser())
        settings.recent_projects = [x for x in settings.recent_projects if x != p]
        settings.recent_projects.insert(0, p)
        settings.recent_projects = settings.recent_projects[:limit]
        self.save(settings)
        return settings

    def add_recent_workdir(self, workdir: str | Path, limit: int = 12) -> AppSettings:
        settings = self.load()
        p = str(Path(workdir).expanduser())
        settings.recent_working_directories = [x for x in settings.recent_working_directories if x != p]
        settings.recent_working_directories.insert(0, p)
        settings.recent_working_directories = settings.recent_working_directories[:limit]
        self.save(settings)
        return settings


def load_app_settings(path: str | Path | None = None) -> AppSettings:
    return ConfigManager(path).load()
