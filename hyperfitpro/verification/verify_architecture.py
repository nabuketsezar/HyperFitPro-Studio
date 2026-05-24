from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json

from hyperfitpro.core.config import ConfigManager, AppSettings
from hyperfitpro.core.plugin_manager import write_model_plugin_template
from hyperfitpro.core.registry import load_models
from hyperfitpro.core.project import project_from_state, save_project, load_project
from hyperfitpro.core.run_database import RunDatabase
from hyperfitpro.core.undo_redo import UndoRedoStack
from hyperfitpro.core.tests import load_test_data
from hyperfitpro.core.optimizer import fit_model
from hyperfitpro.core.run_manager import create_run_folder, save_run


def main():
    with TemporaryDirectory() as td:
        root = Path(td)
        cfg_path = root / 'settings.json'
        cfg = ConfigManager(cfg_path)
        settings = AppSettings(
            default_working_directory=str(root / 'runs'),
            plugin_directories=[str(root / 'plugins')],
            run_database_path=str(root / 'run_history.sqlite3'),
        )
        cfg.save(settings)
        loaded = cfg.load()
        assert Path(loaded.run_database_path).name == 'run_history.sqlite3'

        # Plugin loading
        plugin_path = write_model_plugin_template(root / 'plugins' / 'custom_plugin.py')
        models = load_models(include_inactive=True, plugin_dirs=[root / 'plugins'])
        assert any(m.number == 9001 for m in models), 'Plugin model 9001 was not discovered'

        # Project save/load
        model = next(m for m in models if m.number == 6)
        sample = Path('sample_data/neo_hookean_uniaxial.csv')
        data = load_test_data(sample, 'uniaxial', 1.0)
        project = project_from_state(model, [data], working_directory=root / 'runs', optimizer_settings={'method': 'least_squares_trf'})
        project_path = save_project(project, root / 'demo_project.hfp')
        project2 = load_project(project_path)
        assert project2.model_number == 6
        assert len(project2.datasets) == 1

        # Undo/redo stack
        stack = UndoRedoStack(max_depth=3)
        stack.push('initial', {'x': 1})
        snap = stack.undo({'x': 2})
        assert snap and snap.state['x'] == 1
        snap2 = stack.redo({'x': 1})
        assert snap2 and snap2.state['x'] == 2

        # Run database indexing directly and through save_run.
        db = RunDatabase(root / 'run_history.sqlite3')
        summary = {'model': {'number': 6, 'name': 'Neo-Hookean'}, 'fit_result': {'method': 'demo', 'rmse': 0.1}, 'datasets': [{'n': 3}]}
        run_id = db.index_run_summary(root / 'manual_run', summary)
        assert run_id > 0
        assert db.list_runs(limit=1)

        # Very small real run validates integration with save_run/indexing.
        res = fit_model(model, [data], method='least_squares_trf', max_evals=80)
        folder = create_run_folder(root / 'runs', model)
        save_run(model, [data], res, folder)
        db2 = RunDatabase(root / 'run_history.sqlite3')
        assert db2.list_runs(limit=20), 'Run history is empty after save_run'

        print('Architecture verification passed.')
        print(f'Plugin template: {plugin_path}')
        print(f'Project file: {project_path}')
        print(f'Run database: {db2.path}')


if __name__ == '__main__':
    main()
