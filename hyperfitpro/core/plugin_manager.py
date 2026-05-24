from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Any
import importlib.util
import sys
import traceback
import os

from .config import load_app_settings, DEFAULT_PLUGIN_DIR

EMBEDDED_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'plugins' / 'embedded_models'


@dataclass
class PluginRecord:
    path: str
    module_name: str
    status: str
    message: str = ''
    model_number: int | None = None
    model_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iter_plugin_dirs(extra_dirs: Iterable[str | Path] | None = None, include_embedded: bool = True) -> list[Path]:
    dirs: list[Path] = []
    if include_embedded:
        dirs.append(EMBEDDED_PLUGIN_DIR)
    try:
        cfg = load_app_settings()
        dirs.extend(Path(p).expanduser() for p in cfg.plugin_directories)
    except Exception:
        dirs.append(DEFAULT_PLUGIN_DIR)
    env = os.environ.get('HYPERFITPRO_PLUGIN_PATH', '')
    if env:
        dirs.extend(Path(p).expanduser() for p in env.split(os.pathsep) if p.strip())
    if extra_dirs:
        dirs.extend(Path(p).expanduser() for p in extra_dirs)
    out = []
    seen = set()
    for d in dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key not in seen:
            seen.add(key); out.append(d)
    return out


def discover_plugin_model_classes(extra_dirs: Iterable[str | Path] | None = None, diagnostics: list[PluginRecord] | None = None, include_embedded: bool = True):
    """Load embedded and user model plugin files.

    A plugin is a .py file that exposes MODEL_CLASS. Embedded plugins live inside
    hyperfitpro/plugins/embedded_models and are shipped with the application.
    User plugins are loaded from configured plugin directories and optional extra_dirs.
    Plugin import failures are recorded and ignored so one bad plugin cannot break the app.
    """
    model_classes = []
    for directory in _iter_plugin_dirs(extra_dirs, include_embedded=include_embedded):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if not directory.exists():
            continue
        for path in sorted(directory.glob('*.py')):
            if path.name.startswith('_'):
                continue
            module_name = f"hyperfitpro_user_plugin_{abs(hash(str(path.resolve())))}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    raise RuntimeError('Could not create import spec')
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                cls = getattr(mod, 'MODEL_CLASS', None)
                if cls is None:
                    raise RuntimeError('Plugin file does not define MODEL_CLASS')
                inst = cls()
                model_classes.append(cls)
                if diagnostics is not None:
                    diagnostics.append(PluginRecord(str(path), module_name, 'loaded', '', getattr(inst, 'number', None), getattr(inst, 'name', None)))
            except Exception as exc:
                if diagnostics is not None:
                    diagnostics.append(PluginRecord(str(path), module_name, 'failed', f'{exc}\n{traceback.format_exc(limit=6)}'))
    return model_classes


def write_model_plugin_template(destination: str | Path) -> Path:
    destination = Path(destination).expanduser()
    if destination.is_dir() or not destination.suffix:
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination / 'my_hyperelastic_model_plugin.py'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(MODEL_PLUGIN_TEMPLATE, encoding='utf-8')
    return destination


MODEL_PLUGIN_TEMPLATE = r'''from __future__ import annotations

# Copy this file to one of the plugin directories listed in ~/.hyperfitpro_studio/settings.json
# or set HYPERFITPRO_PLUGIN_PATH to the directory containing this file.

from hyperfitpro.core.base_model import HyperelasticModel, ParameterSpec, invariants


class MyCustomHyperelasticModel(HyperelasticModel):
    number = 9001
    name = "My custom hyperelastic model"
    category = "User plugin"
    family = "Custom"
    active = True
    recommended_tests = ["uniaxial", "biaxial", "planar"]
    material_classes = ["rubber-like", "user-defined"]
    equation_latex = r"W=C_1(I_1-3)+C_2(I_2-3)"
    parameter_specs = [
        ParameterSpec("C1", 1e-6, 1e2, 0.1, scale="positive_stress", unit="stress", description="I1 coefficient"),
        ParameterSpec("C2", -1e2, 1e2, 0.0, scale="stress", unit="stress", description="I2 coefficient"),
    ]

    def W(self, stretches, p):
        I1, I2 = invariants(stretches)
        return p["C1"] * (I1 - 3.0) + p["C2"] * (I2 - 3.0)


MODEL_CLASS = MyCustomHyperelasticModel
'''

# Non-model tool plugin helpers are re-exported here for convenience so GUI/CLI
# can keep using plugin_manager as the single plugin entry point.
try:  # pragma: no cover - import guard for old environments
    from .tool_plugins import discover_tool_plugins, write_tool_plugin_template, ToolContext, ToolResult, ToolPlugin  # noqa: F401
except Exception:
    pass
