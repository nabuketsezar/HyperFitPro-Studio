from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable
import importlib.util
import sys
import os
import traceback
import json
from datetime import datetime

from .config import load_app_settings, DEFAULT_PLUGIN_DIR

EMBEDDED_TOOL_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'plugins' / 'embedded_tools'
USER_TOOL_PLUGIN_DIR = DEFAULT_PLUGIN_DIR / 'tools'


@dataclass
class ToolContext:
    """Runtime context passed to non-model plugins.

    A tool plugin may use only the fields it needs. This keeps plotters,
    auditors, exporters and advisors decoupled from the GUI.
    """
    model: Any | None = None
    datasets: list[Any] = field(default_factory=list)
    fit_result: Any | None = None
    run_folder: Path | str | None = None
    workdir: Path | str | None = None
    parameters: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def output_root(self) -> Path:
        if self.run_folder:
            root = Path(self.run_folder) / 'tools'
        elif self.workdir:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            root = Path(self.workdir) / 'plugin_runs' / f'tool_run_{stamp}'
        else:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            root = Path.home() / 'HyperFitPro_Tool_Runs' / f'tool_run_{stamp}'
        root.mkdir(parents=True, exist_ok=True)
        return root


@dataclass
class ToolResult:
    plugin_id: str
    name: str
    ok: bool
    message: str = ''
    files: list[str] = field(default_factory=list)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolPlugin:
    """Base class for creative HyperFitPro tool plugins.

    Model plugins expose MODEL_CLASS; tool plugins expose TOOL_CLASS.
    A tool plugin can be a plotter, data inspector, report asset builder,
    optimizer advisor, FEA helper, dashboard generator, etc.
    """
    plugin_id = 'tool.base'
    name = 'Base tool plugin'
    category = 'General'
    description = 'Base class; do not run directly.'
    requires_model = False
    requires_data = False
    requires_fit = False
    output_types: list[str] = []
    autorun_after_fit = False

    def can_run(self, ctx: ToolContext) -> tuple[bool, str]:
        if self.requires_model and ctx.model is None:
            return False, 'This tool requires a selected model.'
        if self.requires_data and not ctx.datasets:
            return False, 'This tool requires at least one imported dataset.'
        if self.requires_fit and ctx.fit_result is None:
            return False, 'This tool requires a completed optimization result.'
        return True, ''

    def run(self, ctx: ToolContext) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {
            'plugin_id': self.plugin_id,
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'requires_model': self.requires_model,
            'requires_data': self.requires_data,
            'requires_fit': self.requires_fit,
            'output_types': self.output_types,
            'autorun_after_fit': self.autorun_after_fit,
        }


def _iter_tool_plugin_dirs(extra_dirs: Iterable[str | Path] | None = None, include_embedded: bool = True) -> list[Path]:
    dirs: list[Path] = []
    if include_embedded:
        dirs.append(EMBEDDED_TOOL_PLUGIN_DIR)
    try:
        cfg = load_app_settings()
        for d in cfg.plugin_directories:
            dirs.append(Path(d).expanduser() / 'tools')
    except Exception:
        dirs.append(USER_TOOL_PLUGIN_DIR)
    env = os.environ.get('HYPERFITPRO_TOOL_PLUGIN_PATH', '')
    if env:
        dirs.extend(Path(p).expanduser() for p in env.split(os.pathsep) if p.strip())
    if extra_dirs:
        dirs.extend(Path(p).expanduser() for p in extra_dirs)
    out=[]; seen=set()
    for d in dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key not in seen:
            seen.add(key); out.append(d)
    return out


@dataclass
class ToolPluginRecord:
    path: str
    module_name: str
    status: str
    message: str = ''
    plugin_id: str | None = None
    name: str | None = None
    category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_tool_plugins(extra_dirs: Iterable[str | Path] | None = None, diagnostics: list[ToolPluginRecord] | None = None, include_embedded: bool = True) -> list[ToolPlugin]:
    plugins: list[ToolPlugin] = []
    for directory in _iter_tool_plugin_dirs(extra_dirs, include_embedded=include_embedded):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if not directory.exists():
            continue
        for path in sorted(directory.glob('*.py')):
            if path.name.startswith('_'):
                continue
            module_name = f"hyperfitpro_tool_plugin_{abs(hash(str(path.resolve())))}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    raise RuntimeError('Could not create import spec')
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                cls = getattr(mod, 'TOOL_CLASS', None)
                if cls is None:
                    raise RuntimeError('Tool plugin file does not define TOOL_CLASS')
                inst = cls()
                if not isinstance(inst, ToolPlugin):
                    # Structural fallback: accept duck-typed plugins but wrap is not required.
                    if not all(hasattr(inst, a) for a in ('plugin_id', 'name', 'run')):
                        raise RuntimeError('TOOL_CLASS must inherit ToolPlugin or expose plugin_id/name/run')
                plugins.append(inst)
                if diagnostics is not None:
                    diagnostics.append(ToolPluginRecord(str(path), module_name, 'loaded', '', getattr(inst, 'plugin_id', None), getattr(inst, 'name', None), getattr(inst, 'category', None)))
            except Exception as exc:
                if diagnostics is not None:
                    diagnostics.append(ToolPluginRecord(str(path), module_name, 'failed', f'{exc}\n{traceback.format_exc(limit=8)}'))
    # Deduplicate by plugin_id while keeping first occurrence; embedded ships first.
    dedup={}
    for p in plugins:
        pid = getattr(p, 'plugin_id', str(id(p)))
        if pid not in dedup:
            dedup[pid] = p
    return sorted(dedup.values(), key=lambda p: (getattr(p, 'category', ''), getattr(p, 'name', '')))


def run_tool_plugins(ctx: ToolContext, selected_ids: set[str] | None = None, autorun_only: bool = False) -> list[ToolResult]:
    results: list[ToolResult] = []
    for tool in discover_tool_plugins():
        pid = getattr(tool, 'plugin_id', '')
        if selected_ids is not None and pid not in selected_ids:
            continue
        if autorun_only and not getattr(tool, 'autorun_after_fit', False):
            continue
        ok, msg = tool.can_run(ctx) if hasattr(tool, 'can_run') else (True, '')
        if not ok:
            results.append(ToolResult(pid, getattr(tool, 'name', pid), False, msg))
            continue
        try:
            res = tool.run(ctx)
            if not isinstance(res, ToolResult):
                res = ToolResult(pid, getattr(tool, 'name', pid), True, str(res))
            results.append(res)
        except Exception:
            results.append(ToolResult(pid, getattr(tool, 'name', pid), False, traceback.format_exc()))
    root = ctx.output_root()
    try:
        (root / 'tool_plugin_results.json').write_text(json.dumps([r.to_dict() for r in results], indent=2, default=str), encoding='utf-8')
    except Exception:
        pass
    return results


def write_tool_plugin_template(destination: str | Path) -> Path:
    destination = Path(destination).expanduser()
    if destination.is_dir() or not destination.suffix:
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination / 'my_tool_plugin.py'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(TOOL_PLUGIN_TEMPLATE, encoding='utf-8')
    return destination


TOOL_PLUGIN_TEMPLATE = r'''from __future__ import annotations

from hyperfitpro.core.tool_plugins import ToolPlugin, ToolResult


class MyToolPlugin(ToolPlugin):
    plugin_id = "user.my_tool"
    name = "My custom tool"
    category = "User"
    description = "Example non-model plugin. It writes a small text file into the run tools folder."
    requires_model = True
    requires_data = False
    requires_fit = False
    output_types = ["txt"]

    def run(self, ctx):
        out_dir = ctx.output_root() / self.plugin_id.replace('.', '_')
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "hello_from_tool_plugin.txt"
        p.write_text(f"Model: {ctx.model.number} - {ctx.model.name}\n", encoding="utf-8")
        return ToolResult(self.plugin_id, self.name, True, "Custom tool finished.", [str(p)])


TOOL_CLASS = MyToolPlugin
'''
