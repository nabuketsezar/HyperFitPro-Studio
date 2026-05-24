from pathlib import Path
import tempfile
from hyperfitpro.core.registry import load_models
from hyperfitpro.core.tests import load_test_data
from hyperfitpro.core.project import project_from_state, save_project, load_project
from hyperfitpro.core.tool_plugins import discover_tool_plugins

models = load_models(include_inactive=True)
assert len(models) >= 55, len(models)
plugins = discover_tool_plugins()
assert len(plugins) == 50, len(plugins)
model = [m for m in models if m.number == 6][0]
# Create and load new default project extension
with tempfile.TemporaryDirectory() as td:
    proj = project_from_state(model=model, datasets=[], working_directory=td, name='verify_project')
    path = save_project(proj, Path(td) / 'verify_project.hyp2fit')
    assert path.suffix == '.hyp2fit'
    loaded = load_project(path)
    assert loaded.model_number == 6
    # Legacy extension remains supported
    legacy = save_project(proj, Path(td) / 'legacy.hfp')
    assert legacy.suffix == '.hfp'
    loaded2 = load_project(legacy)
    assert loaded2.model_number == 6
print('modern GUI/project verification passed')
