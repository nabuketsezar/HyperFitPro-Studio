from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any
import json
from datetime import datetime

from .config import ConfigManager, DEFAULT_PROJECT_DIR


PROJECT_SCHEMA_VERSION = 1


@dataclass
class ProjectDatasetRef:
    mode: str
    source_file: str
    weight: float = 1.0
    preprocessing: dict[str, Any] = field(default_factory=dict)
    n_points: int | None = None


@dataclass
class HyperFitProject:
    schema_version: int = PROJECT_SCHEMA_VERSION
    project_name: str = 'Untitled HyperFit Project'
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec='seconds'))
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec='seconds'))
    working_directory: str = ''
    model_number: int | None = None
    model_name: str = ''
    datasets: list[ProjectDatasetRef] = field(default_factory=list)
    optimizer_settings: dict[str, Any] = field(default_factory=dict)
    parameter_bounds: dict[str, list[float]] = field(default_factory=dict)
    report_sections: list[dict[str, Any]] = field(default_factory=list)
    last_run_folder: str | None = None
    last_fit_summary: dict[str, Any] = field(default_factory=dict)
    notes: str = ''

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'HyperFitProject':
        data = dict(data or {})
        drefs = []
        for item in data.get('datasets', []) or []:
            if isinstance(item, ProjectDatasetRef):
                drefs.append(item)
            else:
                drefs.append(ProjectDatasetRef(**{k: v for k, v in dict(item).items() if k in ProjectDatasetRef.__dataclass_fields__}))
        data['datasets'] = drefs
        allowed = {k for k in cls.__dataclass_fields__}
        obj = cls(**{k: v for k, v in data.items() if k in allowed})
        if obj.schema_version != PROJECT_SCHEMA_VERSION:
            obj.schema_version = PROJECT_SCHEMA_VERSION
        return obj


def dataset_ref_from_dataset(dataset) -> ProjectDatasetRef:
    return ProjectDatasetRef(
        mode=getattr(dataset, 'mode', ''),
        source_file=getattr(dataset, 'source_file', ''),
        weight=float(getattr(dataset, 'weight', 1.0)),
        preprocessing=getattr(dataset, 'preprocessing', None) or {},
        n_points=int(len(getattr(dataset, 'x', []))) if hasattr(dataset, 'x') else None,
    )


def project_from_state(model=None, datasets=None, working_directory: str | Path = '', optimizer_settings: dict | None = None,
                       parameter_bounds: dict | None = None, report_sections: list | None = None,
                       last_run_folder: str | Path | None = None, last_fit_result=None, name: str | None = None) -> HyperFitProject:
    fit_summary = {}
    if last_fit_result is not None:
        for key in ['success','method','objective','rmse','mae','r2','aic','aicc','bic','elapsed_s','nfev']:
            if hasattr(last_fit_result, key):
                fit_summary[key] = getattr(last_fit_result, key)
        if hasattr(last_fit_result, 'parameters'):
            fit_summary['parameters'] = getattr(last_fit_result, 'parameters')
    return HyperFitProject(
        project_name=name or (f'{getattr(model, "number", "")}_{getattr(model, "name", "HyperFitProject")}'.strip('_') if model else 'Untitled HyperFit Project'),
        working_directory=str(working_directory or ''),
        model_number=getattr(model, 'number', None),
        model_name=getattr(model, 'name', '') if model else '',
        datasets=[dataset_ref_from_dataset(d) for d in (datasets or [])],
        optimizer_settings=optimizer_settings or {},
        parameter_bounds=parameter_bounds or {},
        report_sections=report_sections or [],
        last_run_folder=str(last_run_folder) if last_run_folder else None,
        last_fit_summary=fit_summary,
    )


def save_project(project: HyperFitProject, path: str | Path | None = None) -> Path:
    if path is None:
        safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in project.project_name)[:80] or 'HyperFitProject'
        path = DEFAULT_PROJECT_DIR / f'{safe}.hyp2fit'
    path = Path(path).expanduser()
    if path.suffix.lower() not in {'.hyp2fit', '.hfp'}:
        path = path.with_suffix('.hyp2fit')
    project.modified_at = datetime.now().isoformat(timespec='seconds')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(project.to_dict(), indent=2), encoding='utf-8')
    ConfigManager().add_recent_project(path)
    return path


def load_project(path: str | Path) -> HyperFitProject:
    path = Path(path).expanduser()
    data = json.loads(path.read_text(encoding='utf-8'))
    project = HyperFitProject.from_dict(data)
    ConfigManager().add_recent_project(path)
    return project
