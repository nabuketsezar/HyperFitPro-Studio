from __future__ import annotations

from pathlib import Path

from hyperfitpro.core.tool_plugins import discover_tool_plugins

REQUIRED = {
    'studio.material_passport',
    'lab.virtual_test_lab',
    'diagnostics.identifiability_svd',
    'advisor.next_test_designer',
    'plot.optimization_cinema',
    'report.interactive_html_dossier',
    'advisor.model_risk_scorecard',
    'plot.response_surface_3d',
    'story.calibration_storyboard',
    'fea.export_auditor',
}

def main():
    tools = discover_tool_plugins()
    ids = {getattr(t, 'plugin_id', '') for t in tools}
    missing = sorted(REQUIRED - ids)
    print(f'tool_plugins_total={len(tools)}')
    for pid in sorted(ids):
        if pid in REQUIRED:
            print('advanced_plugin_ok', pid)
    if missing:
        raise SystemExit('Missing advanced plugins: ' + ', '.join(missing))
    print('Surprise plugin verification passed.')

if __name__ == '__main__':
    main()
