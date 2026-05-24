from __future__ import annotations
import importlib, pkgutil
from .plugin_manager import discover_plugin_model_classes


def _instantiate_model_class(cls):
    try:
        return cls()
    except Exception:
        return None


def load_models(include_inactive=False, include_plugins=True, plugin_dirs=None, diagnostics=None):
    import hyperfitpro.models as models_pkg
    models=[]
    # Built-in model modules
    for info in pkgutil.iter_modules(models_pkg.__path__):
        if not info.name.startswith('model_'):
            continue
        mod=importlib.import_module(f'hyperfitpro.models.{info.name}')
        cls=getattr(mod,'MODEL_CLASS',None)
        if cls is None:
            continue
        m=_instantiate_model_class(cls)
        if m is None:
            continue
        if include_inactive or getattr(m,'active',True):
            models.append(m)
    # User plugin model modules
    if include_plugins:
        for cls in discover_plugin_model_classes(extra_dirs=plugin_dirs, diagnostics=diagnostics):
            m=_instantiate_model_class(cls)
            if m is None:
                continue
            if include_inactive or getattr(m,'active',True):
                models.append(m)
    # Deduplicate by model number while keeping built-ins first. Plugin collisions are ignored.
    dedup={}
    for m in models:
        key=getattr(m,'number',None)
        if key not in dedup:
            dedup[key]=m
    out=list(dedup.values())
    out.sort(key=lambda m:m.number)
    return out


def get_model(number:int, include_plugins=True):
    for m in load_models(include_inactive=True, include_plugins=include_plugins):
        if m.number==number:
            return m
    raise KeyError(f'Model number not found: {number}')
