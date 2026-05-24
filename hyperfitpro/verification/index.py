from __future__ import annotations

DIAGNOSTICS = [
    ("Smoke test", "hyperfitpro.verification.smoke_test", "Runs a compact end-to-end fit and saves a test run."),
    ("Core mechanics", "hyperfitpro.verification.verify_core", "Checks Neo-Hookean analytical response and stability scan."),
    ("Model library", "hyperfitpro.verification.verify_model_library", "Checks active model registry and model metadata."),
    ("Data processing", "hyperfitpro.verification.verify_data_processing", "Checks stress-strain, force-displacement, cyclic branch and weighting preprocessing."),
    ("Optimization", "hyperfitpro.verification.verify_optimization", "Checks optimization diagnostics, confidence and comparison utilities."),
    ("FEA export", "hyperfitpro.verification.verify_fea_export", "Checks Abaqus/ANSYS/CalculiX/export bundle generation."),
    ("Architecture", "hyperfitpro.verification.verify_architecture", "Checks config, project, run database and plugin template architecture."),
    ("Embedded model plugins", "hyperfitpro.verification.verify_embedded_plugins", "Checks embedded model plugins load correctly."),
    ("Tool plugins", "hyperfitpro.verification.verify_tool_plugins", "Checks tool plugin discovery and execution."),
    ("Surprise plugins", "hyperfitpro.verification.verify_surprise_plugins", "Checks advanced studio/lab plugins."),
    ("50 plugin verification", "hyperfitpro.verification.verify_50_plugins", "Checks exactly 50 tool plugins and runs all compatible plugins."),
    ("Modern GUI/project", "hyperfitpro.verification.verify_modern_gui_project", "Checks .hyp2fit/.hfp project load/save and modern project workbench assumptions."),
]
