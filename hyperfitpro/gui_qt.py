from __future__ import annotations

import json
import sys
import time
import traceback
import subprocess
import html
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

import numpy as np

from .core.registry import load_models
from .core.tests import load_test_data, VALID_MODES, auto_balance_dataset_weights, stress_scale
from .core.optimizer import fit_model, OPTIMIZATION_METHODS
from .core.optimizer_control import OptimizationController
from .core.run_manager import create_run_folder, save_run, DEFAULT_ROOT
from .core.license_manager import current_license_status
from .core.reporting import DEFAULT_REPORT_SECTIONS, generate_pdf_report
from .core.excel_calculator import export_excel_calculator
from .core.exporter import export_fea_bundle
from .core.project import project_from_state, save_project, load_project
from .core.run_database import RunDatabase
from .core.plugin_manager import write_model_plugin_template, write_tool_plugin_template
from .core.tool_plugins import discover_tool_plugins, ToolContext, run_tool_plugins
from .verification.index import DIAGNOSTICS
from .core.i18n import LANGUAGES, tr
from .core.base_model import max_invariant_from_test_inputs


def main():
    try:
        from PySide6.QtCore import Qt, QThread, Signal, QUrl, QSize
        from PySide6.QtGui import QDesktopServices, QPixmap, QAction, QIcon
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox, QVBoxLayout,
            QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QComboBox,
            QSplitter, QGroupBox, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
            QSpinBox, QDoubleSpinBox, QStatusBar, QScrollArea, QCheckBox, QFrame,
            QSizePolicy, QToolBar, QProgressBar, QStackedWidget, QTreeWidget,
            QTreeWidgetItem, QListWidget, QListWidgetItem, QDialog, QDialogButtonBox,
            QFormLayout, QAbstractItemView, QButtonGroup, QRadioButton
        )
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
        from matplotlib.figure import Figure
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            'PySide6 and matplotlib Qt backend are required for the Studio GUI. '
            'Install requirements.txt or run the CLI.'
        ) from e

    class MplCanvas(FigureCanvas):
        def __init__(self, width=5.0, height=3.0, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='white')
            super().__init__(self.fig)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.updateGeometry()

    class EquationCanvas(MplCanvas):
        def __init__(self):
            super().__init__(width=8.0, height=1.6, dpi=110)
            self.ax = self.fig.add_subplot(111)
            self.render_equation(r'W = W(I_1, I_2, \lambda_i)')

        def render_equation(self, equation_latex: str):
            self.ax.clear(); self.ax.axis('off')
            eq = (equation_latex or '').replace('\\dfrac', '\\frac').replace('\\displaystyle', '').strip()
            width_inches = max(7.2, min(36.0, 3.8 + 0.055 * len(eq)))
            self.fig.set_size_inches(width_inches, 1.75, forward=True)
            self.setMinimumWidth(int(width_inches * self.fig.dpi))
            self.setMinimumHeight(int(1.75 * self.fig.dpi))
            try:
                self.ax.text(0.01, 0.52, f'${eq}$', ha='left', va='center', fontsize=11, color='#111827', wrap=False)
                self.fig.tight_layout(pad=0.35)
            except Exception:
                self.ax.text(0.01, 0.52, 'LaTeX preview could not render. Use copyable equation text below.', ha='left', va='center', fontsize=10, color='#b91c1c')
            self.draw_idle()

    class CollapsibleBox(QGroupBox):
        def __init__(self, title: str):
            super().__init__(title)
            self.setCheckable(True)
            self.setChecked(False)
            self._content = QWidget()
            self._layout = QVBoxLayout(self)
            self._layout.setContentsMargins(10, 16, 10, 10)
            self._layout.addWidget(self._content)
            self._content.setVisible(False)
            self.toggled.connect(self._content.setVisible)

        def content_layout(self):
            if self._content.layout() is None:
                self._content.setLayout(QVBoxLayout())
            return self._content.layout()

    class FitWorker(QThread):
        log_signal = Signal(str)
        iteration_signal = Signal(object)
        done_signal = Signal(object, object)
        fail_signal = Signal(str)

        def __init__(self, model, datasets, workdir, method, max_evals, overrides=None,
                     regularization=0.0, regularization_type='l2', validation_fraction=0.0,
                     workers=1, near_bound_penalty=0.0):
            super().__init__()
            self.model = model
            self.datasets = datasets
            self.workdir = workdir
            self.method = method
            self.max_evals = int(max_evals)
            self.overrides = overrides or {}
            self.regularization = float(regularization or 0.0)
            self.regularization_type = regularization_type
            self.validation_fraction = float(validation_fraction or 0.0)
            self.workers = int(workers or 1)
            self.near_bound_penalty = float(near_bound_penalty or 0.0)
            self.controller = OptimizationController()

        def _progress(self, payload):
            if isinstance(payload, dict) and payload.get('type') == 'iteration':
                self.iteration_signal.emit(payload)
            else:
                self.log_signal.emit(str(payload))

        def run(self):
            try:
                self.log_signal.emit('=' * 96)
                self.log_signal.emit(f'RUN STARTED | model={self.model.number} {self.model.name} | method={self.method}')
                run_folder = create_run_folder(self.workdir, self.model)
                self.log_signal.emit(f'Run folder: {run_folder}')
                checkpoint_path = run_folder / 'logs' / 'optimizer_checkpoint.json'
                result = fit_model(
                    self.model, self.datasets, method=self.method, max_evals=self.max_evals,
                    override_bounds=self.overrides, regularization=self.regularization,
                    regularization_type=self.regularization_type, validation_fraction=self.validation_fraction,
                    workers=self.workers, near_bound_penalty=self.near_bound_penalty,
                    checkpoint_path=checkpoint_path, control=self.controller, trace_stride=5,
                    progress=self._progress,
                )
                save_run(self.model, self.datasets, result, run_folder)
                self.done_signal.emit(result, str(run_folder))
            except Exception:
                self.fail_signal.emit(traceback.format_exc())

    class DiagnosticWorker(QThread):
        log_signal = Signal(str)
        done_signal = Signal(int)

        def __init__(self, diagnostics, cwd):
            super().__init__()
            self.diagnostics = diagnostics
            self.cwd = str(cwd)

        def run(self):
            failures = 0
            for label, module, description in self.diagnostics:
                self.log_signal.emit(f"\n{'='*96}\nRUNNING: {label}\nMODULE : {module}\nINFO   : {description}\n")
                try:
                    proc = subprocess.run(
                        [sys.executable, '-m', module],
                        cwd=self.cwd, capture_output=True, text=True, timeout=1800
                    )
                    out = (proc.stdout or '') + (proc.stderr or '')
                    self.log_signal.emit(out.strip() if out.strip() else '(no output)')
                    if proc.returncode == 0:
                        self.log_signal.emit(f"PASS: {label}")
                    else:
                        failures += 1
                        self.log_signal.emit(f"FAIL: {label} | returncode={proc.returncode}")
                except Exception as exc:
                    failures += 1
                    self.log_signal.emit(f"FAIL: {label} | {exc}")
            self.done_signal.emit(failures)

    class ModernMainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('HyperFitPro Studio 1.0')
            self.resize(1500, 900)
            self.models = load_models(include_inactive=True)
            self.model_map = {}
            self.filtered_model_labels = []
            self.tool_plugins = discover_tool_plugins()
            self.datasets = []
            self.last_result = None
            self.last_run_folder = None
            self.last_model = None
            self.last_datasets = []
            self.worker = None
            self.live_trace = []
            self.plugin_results = []
            self.imported_projects = []
            self._last_live_draw = 0.0
            self._last_data_draw = 0.0
            self.language = 'tr'
            self._build_ui()
            self._apply_theme()
            self._load_models_into_combo()
            self._load_plugins_tree()
            self._refresh_project_targets()
            self._set_status(tr(self.language, 'ready'), 'warn')
            lic = current_license_status()
            self.log(lic.get('message', 'License loaded') + ' | fingerprint=' + lic.get('fingerprint', ''))

        # -------------------- UI construction --------------------
        def _build_ui(self):
            root = QWidget(); self.setCentralWidget(root)
            outer = QVBoxLayout(root); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
            outer.addWidget(self._build_header())

            self.toolbar = QToolBar('Main actions'); self.toolbar.setMovable(False); self.toolbar.setIconSize(QSize(16, 16)); self.toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self.addToolBar(Qt.TopToolBarArea, self.toolbar)
            self.action_sidebar = self.toolbar.addAction('☰ Menu', self.toggle_sidebar)
            self.toolbar.addSeparator()
            self.action_import_project = self.toolbar.addAction('Import', self.import_project_dialog)
            self.action_save_project = self.toolbar.addAction('Save', self.save_project_dialog)
            self.action_run_history = self.toolbar.addAction('Runs', self.show_run_history)
            self.action_diagnostics = self.toolbar.addAction('Checks', lambda: self.stack.setCurrentIndex(8))
            self.action_help = self.toolbar.addAction('Help', lambda: self.stack.setCurrentIndex(9))
            self.toolbar.addSeparator()
            self.action_open_workdir = self.toolbar.addAction('Open Dir', self.open_workdir)

            body = QSplitter(Qt.Horizontal)
            outer.addWidget(body, 1)
            self.sidebar = self._build_sidebar()
            body.addWidget(self.sidebar)
            body.setCollapsible(0, True)

            self.stack = QStackedWidget()
            body.addWidget(self.stack)
            body.setStretchFactor(0, 0); body.setStretchFactor(1, 1)
            body.setSizes([245, 1255])

            self.stack.addWidget(self._page_dashboard())
            self.stack.addWidget(self._page_model())
            self.stack.addWidget(self._page_data())
            self.stack.addWidget(self._page_optimization())
            self.stack.addWidget(self._page_report_export())
            self.stack.addWidget(self._page_plugins())
            self.stack.addWidget(self._page_plugin_results())
            self.stack.addWidget(self._page_project_library())
            self.stack.addWidget(self._page_diagnostics())
            self.stack.addWidget(self._page_help())
            self.stack.addWidget(self._page_group_overview('workflow'))
            self.stack.addWidget(self._page_group_overview('plugins'))
            self.stack.addWidget(self._page_group_overview('projects'))
            self.stack.addWidget(self._page_group_overview('tools'))

            self.status = QStatusBar(); self.setStatusBar(self.status)
            self.nav_tree.expandAll()
            self._apply_language()

        def _build_header(self):
            header = QFrame(); header.setObjectName('Header')
            lay = QHBoxLayout(header); lay.setContentsMargins(16, 8, 16, 8)
            title_col = QVBoxLayout()
            self.header_title = QLabel('HyperFitPro Studio'); self.header_title.setObjectName('Title')
            self.header_subtitle = QLabel('Modern minimal hyperelastic calibration, project library and plugin workbench')
            self.header_subtitle.setObjectName('Subtitle')
            title_col.addWidget(self.header_title); title_col.addWidget(self.header_subtitle)
            lay.addLayout(title_col)
            lay.addStretch(1)
            self.language_label = QLabel('Dil')
            self.language_combo = QComboBox()
            for code, name in LANGUAGES.items():
                self.language_combo.addItem(name, code)
            self.language_combo.currentIndexChanged.connect(self.on_language_changed)
            lay.addWidget(self.language_label)
            lay.addWidget(self.language_combo)
            self.status_badge = QLabel('Ready'); self.status_badge.setObjectName('BadgeWarn')
            lay.addWidget(self.status_badge)
            return header

        def _build_sidebar(self):
            frame = QFrame(); frame.setObjectName('Sidebar')
            lay = QVBoxLayout(frame); lay.setContentsMargins(8, 8, 8, 8)
            top = QHBoxLayout()
            self.nav_title_label = QLabel('Navigation'); self.nav_title_label.setObjectName('SideTitle')
            self.hide_sidebar_button = QPushButton('‹'); self.hide_sidebar_button.setObjectName('Flat'); self.hide_sidebar_button.clicked.connect(self.toggle_sidebar)
            top.addWidget(self.nav_title_label); top.addStretch(1); top.addWidget(self.hide_sidebar_button)
            lay.addLayout(top)
            self.nav_tree = QTreeWidget(); self.nav_tree.setHeaderHidden(True); self.nav_tree.setObjectName('NavTree')
            self.nav_tree.itemClicked.connect(self.on_nav_clicked)
            lay.addWidget(self.nav_tree, 1)
            self._add_nav_group('workflow', 10, [
                ('dashboard', 0), ('model', 1), ('data_import', 2), ('optimization', 3), ('report_export', 4),
            ])
            self._add_nav_group('plugins', 11, [('plugin_workbench', 5), ('plugin_results', 6)])
            self._add_nav_group('projects', 12, [('project_library', 7)])
            self._add_nav_group('tools', 13, [('diagnostics', 8), ('help', 9)])
            return frame

        def _add_nav_group(self, key, overview_index, children):
            root = QTreeWidgetItem([tr(self.language, key)]); root.setData(0, Qt.UserRole, overview_index); root.setData(0, Qt.UserRole + 1, key)
            root.setExpanded(True)
            self.nav_tree.addTopLevelItem(root)
            for n, (child_key, idx) in enumerate(children, start=1):
                child = QTreeWidgetItem([f'{n:02d}  ' + tr(self.language, child_key)])
                child.setData(0, Qt.UserRole, idx); child.setData(0, Qt.UserRole + 1, child_key); root.addChild(child)

        def _section_title(self, title, subtitle=''):
            box = QWidget(); lay = QVBoxLayout(box); lay.setContentsMargins(0, 0, 0, 8)
            t = QLabel(title); t.setObjectName('PageTitle'); lay.addWidget(t)
            if subtitle:
                s = QLabel(subtitle); s.setObjectName('PageSubtitle'); s.setWordWrap(True); lay.addWidget(s)
            return box

        def _page_group_overview(self, group_key):
            page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(18, 16, 18, 16); lay.setSpacing(12)
            title = tr(self.language, group_key)
            subtitle = tr(self.language, group_key + '_overview_subtitle')
            page.title_widget = self._section_title(title, subtitle)
            lay.addWidget(page.title_widget)
            intro = QLabel(tr(self.language, group_key + '_overview_text'))
            intro.setObjectName('Tip'); intro.setWordWrap(True); lay.addWidget(intro)
            page.intro_label = intro
            flow_box = QGroupBox(tr(self.language, 'guided_flow'))
            flow = QHBoxLayout(flow_box); flow.setSpacing(10)
            mapping = {
                'workflow': [('dashboard',0), ('model',1), ('data_import',2), ('optimization',3), ('report_export',4)],
                'plugins': [('plugin_workbench',5), ('plugin_results',6)],
                'projects': [('project_library',7)],
                'tools': [('diagnostics',8), ('help',9)],
            }
            page.flow_buttons = []
            for i, (key, idx) in enumerate(mapping.get(group_key, []), start=1):
                btn = QPushButton(f'{i}. ' + tr(self.language, key))
                btn.setMinimumHeight(46)
                btn.setObjectName('Secondary')
                btn.clicked.connect(lambda checked=False, ix=idx: self.stack.setCurrentIndex(ix))
                flow.addWidget(btn)
                page.flow_buttons.append((btn, key, i))
                if i < len(mapping.get(group_key, [])):
                    arrow = QLabel('→'); arrow.setAlignment(Qt.AlignCenter); arrow.setObjectName('FlowArrow'); flow.addWidget(arrow)
            lay.addWidget(flow_box)
            page.flow_box = flow_box
            details = QTextEdit(); details.setReadOnly(True); details.setMinimumHeight(220)
            details.setPlainText(tr(self.language, group_key + '_overview_steps'))
            lay.addWidget(details, 1)
            page.details = details
            return page

        def _page_dashboard(self):
            page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(14, 12, 14, 12)
            self.dashboard_title_widget = self._section_title('Dashboard', 'Tek sayfadan çalışma klasörü, aktif proje durumu, hızlı import/export ve genel yönlendirme.')
            lay.addWidget(self.dashboard_title_widget)
            self.dashboard_flow_label = QLabel('Directory → Model → Data → Optimize → Report / Export → Plugins → Save Project')
            self.dashboard_flow_label.setObjectName('FlowPill')
            self.dashboard_flow_label.setAlignment(Qt.AlignCenter)
            lay.addWidget(self.dashboard_flow_label)
            grid = QGridLayout()
            grid.addWidget(self._card_workdir(), 0, 0)
            grid.addWidget(self._card_current_state(), 0, 1)
            grid.addWidget(self._card_quick_actions(), 1, 0)
            grid.addWidget(self._card_guidance(), 1, 1)
            lay.addLayout(grid)
            self.dashboard_log = QTextEdit(); self.dashboard_log.setReadOnly(True); self.dashboard_log.setMinimumHeight(135)
            self.session_log_label = QLabel('Session log'); lay.addWidget(self.session_log_label)
            lay.addWidget(self.dashboard_log, 1)
            return page

        def _card_workdir(self):
            self.workdir_group = QGroupBox('Working directory'); g = self.workdir_group
            l = QGridLayout(g)
            self.workdir = QLineEdit(str(DEFAULT_ROOT))
            l.addWidget(self.workdir, 0, 0, 1, 3)
            self.workdir_select_btn = QPushButton('Select'); self.workdir_select_btn.clicked.connect(self.choose_workdir); l.addWidget(self.workdir_select_btn, 0, 3)
            b2 = QPushButton('Open'); self.workdir_open_btn = b2; b2.setObjectName('Secondary'); b2.clicked.connect(self.open_workdir); l.addWidget(b2, 0, 4)
            note = QLabel('Her run için input_data, plots, logs, reports, exports ve tools klasörleri otomatik oluşur.')
            note.setWordWrap(True); l.addWidget(note, 1, 0, 1, 5)
            return g

        def _card_current_state(self):
            self.current_workspace_group = QGroupBox('Current workspace'); g = self.current_workspace_group
            l = QVBoxLayout(g)
            self.state_label = QLabel('No model/data/run yet.'); self.state_label.setWordWrap(True)
            l.addWidget(self.state_label)
            return g

        def _card_quick_actions(self):
            self.quick_actions_group = QGroupBox('Quick actions'); g = self.quick_actions_group
            l = QGridLayout(g)
            actions = [
                ('import_project', self.import_project_dialog), ('load_selected_project', self.load_selected_project_into_workspace),
                ('save_hyp2fit', self.save_project_dialog), ('run_optimization', self.start_fit),
                ('run_selected_plugins', lambda: self.run_plugins(selected_only=True)), ('view_plugin_results', lambda: self.stack.setCurrentIndex(6)),
                ('diagnostics_center', lambda: self.stack.setCurrentIndex(8)), ('help_user_guide', lambda: self.stack.setCurrentIndex(9)),
            ]
            self.quick_action_buttons = []
            for i, (key, cb) in enumerate(actions):
                b = QPushButton(tr(self.language, key)); b.setObjectName('Secondary' if i % 2 else '')
                b.clicked.connect(cb); l.addWidget(b, i // 2, i % 2)
                self.quick_action_buttons.append((b, key))
            return g

        def _card_guidance(self):
            self.recommended_workflow_group = QGroupBox('Recommended workflow'); g = self.recommended_workflow_group
            l = QVBoxLayout(g)
            self.workflow_text = QLabel(tr(self.language, 'workflow_guidance'))
            self.workflow_text.setWordWrap(True); l.addWidget(self.workflow_text)
            return g

        def _page_model(self):
            page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(14, 12, 14, 12)
            self.model_title_widget = self._section_title('Model selection', 'Model bilgisi, denklem, ham LaTeX ve parametre min/max aralıkları tek kaydırılabilir kutuda gösterilir.')
            lay.addWidget(self.model_title_widget)
            top = QHBoxLayout()
            self.model_search = QLineEdit(); self.model_search.setPlaceholderText('Search model name, number, category, family...')
            self.model_search.textChanged.connect(self._load_models_into_combo)
            self.category_filter = QComboBox(); self.category_filter.addItem('All categories'); self.category_filter.currentIndexChanged.connect(self._load_models_into_combo)
            self.model_search_label = QLabel('Search'); self.model_category_label = QLabel('Category')
            top.addWidget(self.model_search_label); top.addWidget(self.model_search, 2)
            top.addWidget(self.model_category_label); top.addWidget(self.category_filter, 1)
            lay.addLayout(top)
            self.model_combo = QComboBox(); self.model_combo.currentIndexChanged.connect(self.on_model_changed)
            lay.addWidget(self.model_combo)

            self.model_info = QTextEdit()
            self.model_info.setReadOnly(True)
            self.model_info.setAcceptRichText(True)
            self.model_info.setLineWrapMode(QTextEdit.NoWrap)
            self.model_info.setMinimumHeight(460)
            self.model_info.setPlaceholderText('Select a model to view equation, calibration notes and parameter bounds...')
            lay.addWidget(self.model_info, 1)

            btn_row = QHBoxLayout()
            self.copy_model_details_btn = QPushButton('Copy model details'); self.copy_model_details_btn.setObjectName('Secondary'); self.copy_model_details_btn.clicked.connect(self.copy_equation)
            btn_row.addStretch(1); btn_row.addWidget(self.copy_model_details_btn)
            lay.addLayout(btn_row)
            return page

        def _page_data(self):
            page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(14, 12, 14, 12)
            self.data_title_widget = self._section_title('Data import', 'Çoklu CSV/XLSX içeri aktar. Ham force-displacement veya işlenmiş stress-strain datası aynı panelden alınır.')
            lay.addWidget(self.data_title_widget)
            controls = QGroupBox('Import settings'); self.data_import_settings_group = controls
            g = QGridLayout(controls)
            self.mode_combo = QComboBox(); self.mode_combo.addItems(VALID_MODES)
            self.data_kind_combo = QComboBox(); self.data_kind_combo.addItems(['auto','stress_strain','true_stress_strain','force_displacement','stretch_stress','volumetric'])
            self.weight = QDoubleSpinBox(); self.weight.setRange(0, 100); self.weight.setValue(1.0); self.weight.setDecimals(3)
            self.force_unit_combo = QComboBox(); self.force_unit_combo.addItems(['N','kN','lbf'])
            self.length_unit_combo = QComboBox(); self.length_unit_combo.addItems(['mm','m','cm','in'])
            self.stress_unit_combo = QComboBox(); self.stress_unit_combo.addItems(['MPa','N/mm^2','Pa','kPa','GPa','psi','ksi'])
            self.area_spin = QDoubleSpinBox(); self.area_spin.setRange(0,1e12); self.area_spin.setDecimals(6)
            self.width_spin = QDoubleSpinBox(); self.width_spin.setRange(0,1e12); self.width_spin.setDecimals(6)
            self.thickness_spin = QDoubleSpinBox(); self.thickness_spin.setRange(0,1e12); self.thickness_spin.setDecimals(6)
            self.diameter_spin = QDoubleSpinBox(); self.diameter_spin.setRange(0,1e12); self.diameter_spin.setDecimals(6)
            self.gauge_spin = QDoubleSpinBox(); self.gauge_spin.setRange(0,1e12); self.gauge_spin.setDecimals(6)
            self.strain_percent_check = QCheckBox('strain column is %')
            labels = [('mode_label', 'Mode', self.mode_combo), ('data_kind_label', 'Data kind', self.data_kind_combo), ('weight_label', 'Weight', self.weight), ('force_unit_label', 'Force unit', self.force_unit_combo), ('length_unit_label', 'Length unit', self.length_unit_combo), ('stress_unit_label', 'Stress unit', self.stress_unit_combo), ('area_label', 'Area', self.area_spin), ('width_label', 'Width', self.width_spin), ('thickness_label', 'Thickness', self.thickness_spin), ('diameter_label', 'Diameter', self.diameter_spin), ('gauge_length_label', 'Gauge length', self.gauge_spin)]
            self.data_form_labels = []
            for i, (key, lab, w) in enumerate(labels):
                r = i // 3; c = (i % 3) * 2
                ql = QLabel(lab); g.addWidget(ql, r, c); g.addWidget(w, r, c+1); self.data_form_labels.append((ql, key))
            g.addWidget(self.strain_percent_check, 4, 0, 1, 2)
            adv = CollapsibleBox('Advanced preprocessing'); self.advanced_preprocessing_group = adv
            av = QGridLayout(); adv._content.setLayout(av)
            self.branch_combo = QComboBox(); self.branch_combo.addItems(['all','loading','unloading','first_monotonic'])
            self.zero_offset_check = QCheckBox('zero-offset correction'); self.zero_offset_check.setChecked(True)
            self.smooth_method_combo = QComboBox(); self.smooth_method_combo.addItems(['none','moving_average','savitzky_golay'])
            self.smooth_window_spin = QSpinBox(); self.smooth_window_spin.setRange(0,1001)
            self.outlier_combo = QComboBox(); self.outlier_combo.addItems(['none','zscore','mad'])
            self.toe_fraction_spin = QDoubleSpinBox(); self.toe_fraction_spin.setRange(0,0.9); self.toe_fraction_spin.setDecimals(4); self.toe_fraction_spin.setSingleStep(0.01)
            self.downsample_spin = QSpinBox(); self.downsample_spin.setRange(0,1000000)
            self.region_weight_combo = QComboBox(); self.region_weight_combo.addItems(['uniform','low_strain','high_strain','balanced_bins'])
            self.preprocess_form_labels = []
            for i, (key, lab, w) in enumerate([('branch_label', 'Branch', self.branch_combo), ('smoothing_label', 'Smoothing', self.smooth_method_combo), ('window_label', 'Window', self.smooth_window_spin), ('outlier_label', 'Outlier', self.outlier_combo), ('toe_remove_label', 'Toe remove %', self.toe_fraction_spin), ('downsample_label', 'Downsample max', self.downsample_spin), ('region_weighting_label', 'Region weighting', self.region_weight_combo)]):
                ql = QLabel(lab); av.addWidget(ql, i//2, (i%2)*2); av.addWidget(w, i//2, (i%2)*2+1); self.preprocess_form_labels.append((ql, key))
            av.addWidget(self.zero_offset_check, 4, 0, 1, 2)
            lay.addWidget(controls); lay.addWidget(adv)
            btns = QHBoxLayout()
            self.add_data_btn = QPushButton('Add one or multiple files'); self.add_data_btn.clicked.connect(self.add_data); btns.addWidget(self.add_data_btn)
            self.remove_data_btn = QPushButton('Remove selected'); self.remove_data_btn.setObjectName('Secondary'); self.remove_data_btn.clicked.connect(self.remove_data); btns.addWidget(self.remove_data_btn)
            self.autobalance_btn = QPushButton('Auto-balance weights'); self.autobalance_btn.setObjectName('Secondary'); self.autobalance_btn.clicked.connect(self.auto_balance_weights); btns.addWidget(self.autobalance_btn)
            btns.addStretch(1); lay.addLayout(btns)
            split = QSplitter(Qt.Horizontal); lay.addWidget(split, 1)
            self.data_table = QTableWidget(0, 5); self.data_table.setHorizontalHeaderLabels(['Mode','File','Points','Weight','Preprocessing'])
            self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); split.addWidget(self.data_table)
            plotbox = QWidget(); pl = QVBoxLayout(plotbox)
            self.data_canvas = MplCanvas(6.5, 4.0); pl.addWidget(NavigationToolbar(self.data_canvas, self)); pl.addWidget(self.data_canvas)
            split.addWidget(plotbox); split.setStretchFactor(0, 1); split.setStretchFactor(1, 1)
            return page

        def _page_optimization(self):
            page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(14, 12, 14, 12)
            self.optimization_title_widget = self._section_title('Optimization', 'Sağdaki canlı monitor üzerinde objective, best objective ve fit preview iterasyon sırasında güncellenir.')
            lay.addWidget(self.optimization_title_widget)
            controls = QGroupBox('Run controls'); self.run_controls_group = controls
            g = QGridLayout(controls)
            self.method = QComboBox(); self.method.addItems(OPTIMIZATION_METHODS); self.method.setCurrentText('recommended_adaptive_bounds')
            self.maxeval = QSpinBox(); self.maxeval.setRange(10, 1_000_000); self.maxeval.setValue(6000)
            self.reg_strength = QDoubleSpinBox(); self.reg_strength.setRange(0, 1e9); self.reg_strength.setDecimals(8); self.reg_strength.setSingleStep(1e-4)
            self.reg_type = QComboBox(); self.reg_type.addItems(['none','l2','l1','elastic_net','magnitude'])
            self.validation_fraction = QDoubleSpinBox(); self.validation_fraction.setRange(0,0.8); self.validation_fraction.setDecimals(3); self.validation_fraction.setSingleStep(0.05)
            self.worker_count = QSpinBox(); self.worker_count.setRange(1,64); self.worker_count.setValue(1)
            self.near_bound_penalty = QDoubleSpinBox(); self.near_bound_penalty.setRange(0,1e9); self.near_bound_penalty.setDecimals(8)
            self.optimization_form_labels = []
            for i, (key, lab, w) in enumerate([('method_label', 'Method', self.method), ('max_evaluations_label', 'Max evaluations', self.maxeval), ('regularization_label', 'Regularization', self.reg_strength), ('reg_type_label', 'Reg. type', self.reg_type), ('validation_split_label', 'Validation split', self.validation_fraction), ('workers_label', 'Workers', self.worker_count), ('near_bound_penalty_label', 'Near-bound penalty', self.near_bound_penalty)]):
                ql = QLabel(lab); g.addWidget(ql, i//4, (i%4)*2); g.addWidget(w, i//4, (i%4)*2+1); self.optimization_form_labels.append((ql, key))
            self.optimization_tip = QLabel('Recommended: start with recommended_adaptive_bounds. Global methods can be slow; the GUI will keep updating live plots while running.')
            self.optimization_tip.setObjectName('Tip'); self.optimization_tip.setWordWrap(True); g.addWidget(self.optimization_tip, 2, 0, 1, 8)
            btns = QHBoxLayout()
            self.run_btn = QPushButton('Run optimization'); self.run_btn.clicked.connect(self.start_fit)
            self.pause_btn = QPushButton('Pause'); self.pause_btn.setObjectName('Secondary'); self.pause_btn.setEnabled(False); self.pause_btn.clicked.connect(self.pause_resume_fit)
            self.stop_btn = QPushButton('Stop'); self.stop_btn.setObjectName('Danger'); self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.stop_fit)
            btns.addWidget(self.run_btn); btns.addWidget(self.pause_btn); btns.addWidget(self.stop_btn); btns.addStretch(1)
            g.addLayout(btns, 3, 0, 1, 8)
            lay.addWidget(controls)
            split = QSplitter(Qt.Horizontal); lay.addWidget(split, 1)
            left = QWidget(); ll = QVBoxLayout(left)
            self.iter_table = QTableWidget(0, 4); self.iter_table.setHorizontalHeaderLabels(['Eval','Objective','Best objective','Parameters'])
            self.iter_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); ll.addWidget(self.iter_table)
            self.run_log = QTextEdit(); self.run_log.setReadOnly(True); self.run_log.setMaximumHeight(150); ll.addWidget(QLabel('Run log')); ll.addWidget(self.run_log)
            split.addWidget(left)
            right = QWidget(); rl = QVBoxLayout(right)
            self.live_canvas = MplCanvas(7.0, 4.0); rl.addWidget(NavigationToolbar(self.live_canvas, self)); rl.addWidget(self.live_canvas, 1)
            self.fit_preview_canvas = MplCanvas(7.0, 4.0); rl.addWidget(NavigationToolbar(self.fit_preview_canvas, self)); rl.addWidget(self.fit_preview_canvas, 1)
            split.addWidget(right); split.setStretchFactor(0, 1); split.setStretchFactor(1, 1)
            return page

        def _page_report_export(self):
            page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(14, 12, 14, 12)
            self.report_title_widget = self._section_title('Report / Export', 'PDF report, FEA export bundle, Excel calculator ve run klasörü yönetimi.')
            lay.addWidget(self.report_title_widget)
            actions = QHBoxLayout()
            self.report_action_buttons = []
            for key, cb, sec in [
                ('open_last_run_folder', self.open_last_run, False), ('generate_fea_bundle', self.generate_fea_exports, True),
                ('generate_pdf_report', self.generate_pdf_report_from_gui, True), ('generate_excel_calculator', self.generate_excel_calculator_from_gui, True),
            ]:
                b = QPushButton(tr(self.language, key)); b.setObjectName('Secondary' if sec else ''); b.clicked.connect(cb); actions.addWidget(b); self.report_action_buttons.append((b, key))
            actions.addStretch(1); lay.addLayout(actions)
            self.results_table = QTableWidget(0, 2); self.results_table.setHorizontalHeaderLabels(['Parameter / Metric','Value'])
            self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); lay.addWidget(self.results_table, 1)
            self.report_note = QTextEdit(); self.report_note.setReadOnly(True); self.report_note.setMaximumHeight(180)
            self.report_note.setPlainText(tr(self.language, 'report_note'))
            lay.addWidget(self.report_note)
            return page

        def _page_plugins(self):
            page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(14, 12, 14, 12)
            self.plugin_title_widget = self._section_title('Plugin Workbench', '50 embedded plugins are grouped by category. Run them on the current workspace or on imported .hyp2fit/.hfp projects.')
            lay.addWidget(self.plugin_title_widget)
            top = QHBoxLayout()
            self.plugin_target_combo = QComboBox(); self.plugin_target_label = QLabel('Target project'); top.addWidget(self.plugin_target_label); top.addWidget(self.plugin_target_combo, 2)
            self.plugin_import_btn = QPushButton('Import project'); self.plugin_import_btn.setObjectName('Secondary'); self.plugin_import_btn.clicked.connect(self.import_project_dialog); top.addWidget(self.plugin_import_btn)
            self.plugin_reload_btn = QPushButton('Reload plugins'); self.plugin_reload_btn.setObjectName('Secondary'); self.plugin_reload_btn.clicked.connect(self.reload_plugins); top.addWidget(self.plugin_reload_btn)
            self.plugin_run_selected_btn = QPushButton('Run selected'); self.plugin_run_selected_btn.clicked.connect(lambda: self.run_plugins(selected_only=True)); top.addWidget(self.plugin_run_selected_btn)
            self.plugin_run_category_btn = QPushButton('Run compatible in category'); self.plugin_run_category_btn.setObjectName('Secondary'); self.plugin_run_category_btn.clicked.connect(lambda: self.run_plugins(selected_only=False)); top.addWidget(self.plugin_run_category_btn)
            lay.addLayout(top)
            split = QSplitter(Qt.Horizontal); lay.addWidget(split, 1)
            self.plugin_tree = QTreeWidget(); self.plugin_tree.setHeaderLabels([tr(self.language, 'plugin_categories')]); self.plugin_tree.setSelectionMode(QAbstractItemView.ExtendedSelection); self.plugin_tree.itemSelectionChanged.connect(self.on_plugin_selection_changed)
            split.addWidget(self.plugin_tree)
            right = QWidget(); rl = QVBoxLayout(right)
            self.plugin_detail = QTextEdit(); self.plugin_detail.setReadOnly(True); self.plugin_detail_label = QLabel('Plugin detail'); rl.addWidget(self.plugin_detail_label); rl.addWidget(self.plugin_detail, 1)
            self.plugin_matrix = QTableWidget(0, 5); self.plugin_matrix.setHorizontalHeaderLabels(['Category','ID','Name','Requires','Outputs']); self.plugin_matrix.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.flat_plugin_list_label = QLabel('Flat plugin list'); rl.addWidget(self.flat_plugin_list_label); rl.addWidget(self.plugin_matrix, 2)
            split.addWidget(right); split.setStretchFactor(0, 1); split.setStretchFactor(1, 2)
            return page

        def _page_plugin_results(self):
            page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(14, 12, 14, 12)
            self.plugin_results_title_widget = self._section_title('Plugin Results', 'Çalıştırılan plugin çıktıları bu sekmede incelenir. Metin, CSV, JSON, HTML ve görseller önizlenebilir; diğer dosyalar dışarıdan açılır.')
            lay.addWidget(self.plugin_results_title_widget)
            split = QSplitter(Qt.Horizontal); lay.addWidget(split, 1)
            left = QWidget(); ll = QVBoxLayout(left)
            self.plugin_results_table = QTableWidget(0, 6); self.plugin_results_table.setHorizontalHeaderLabels(['Time','Target','Plugin','OK','Message','Files'])
            self.plugin_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.plugin_results_table.itemSelectionChanged.connect(self.on_plugin_result_selected)
            ll.addWidget(self.plugin_results_table)
            self.plugin_file_list = QListWidget(); self.plugin_file_list.itemSelectionChanged.connect(self.on_plugin_file_selected)
            ll.addWidget(QLabel('Files from selected result')); ll.addWidget(self.plugin_file_list, 1)
            split.addWidget(left)
            right = QWidget(); rl = QVBoxLayout(right)
            preview_actions = QHBoxLayout()
            b = QPushButton('Open selected file'); b.setObjectName('Secondary'); b.clicked.connect(self.open_selected_plugin_file); preview_actions.addWidget(b)
            b = QPushButton('Open output folder'); b.setObjectName('Secondary'); b.clicked.connect(self.open_selected_plugin_folder); preview_actions.addWidget(b)
            preview_actions.addStretch(1); rl.addLayout(preview_actions)
            self.preview_stack = QStackedWidget(); rl.addWidget(self.preview_stack, 1)
            self.text_preview = QTextEdit(); self.text_preview.setReadOnly(True); self.preview_stack.addWidget(self.text_preview)
            self.image_preview_scroll = QScrollArea(); self.image_label = QLabel(); self.image_label.setAlignment(Qt.AlignCenter); self.image_preview_scroll.setWidgetResizable(True); self.image_preview_scroll.setWidget(self.image_label); self.preview_stack.addWidget(self.image_preview_scroll)
            split.addWidget(right); split.setStretchFactor(0, 1); split.setStretchFactor(1, 2)
            return page

        def _page_project_library(self):
            page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(14, 12, 14, 12)
            self.project_title_widget = self._section_title('Project Library', 'Daha önce kaydedilmiş .hyp2fit veya eski .hfp projeleri içeri alınabilir; pluginler bu projelere de uygulanabilir.')
            lay.addWidget(self.project_title_widget)
            actions = QHBoxLayout()
            for txt, cb, sec in [
                ('Import .hyp2fit / .hfp project(s)', self.import_project_dialog, False), ('Load selected into workspace', self.load_selected_project_into_workspace, True),
                ('Save current as .hyp2fit', self.save_project_dialog, True), ('Remove selected from library', self.remove_selected_project, True),
            ]:
                b = QPushButton(txt); b.setObjectName('Secondary' if sec else ''); b.clicked.connect(cb); actions.addWidget(b)
            actions.addStretch(1); lay.addLayout(actions)
            self.project_table = QTableWidget(0, 7); self.project_table.setHorizontalHeaderLabels(['File','Project','Model','Datasets','Last run','Modified','Notes'])
            self.project_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); lay.addWidget(self.project_table, 1)
            self.project_detail = QTextEdit(); self.project_detail.setReadOnly(True); self.project_detail.setMaximumHeight(210); lay.addWidget(QLabel('Selected project detail')); lay.addWidget(self.project_detail)
            self.project_table.itemSelectionChanged.connect(self.on_project_selected)
            return page

        def _page_diagnostics(self):
            page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(14, 12, 14, 12)
            lay.addWidget(self._section_title('Diagnostics / Verification Center', 'Root klasörde dağınık duran verify_*.py dosyaları artık GUI içinde yönetilir. İstersen tek test, istersen tüm doğrulama setini buradan çalıştır.'))
            top = QHBoxLayout()
            self.diag_run_selected_btn = QPushButton('Run selected checks')
            self.diag_run_selected_btn.clicked.connect(self.run_selected_diagnostics)
            self.diag_run_all_btn = QPushButton('Run all checks')
            self.diag_run_all_btn.setObjectName('Secondary')
            self.diag_run_all_btn.clicked.connect(self.run_all_diagnostics)
            self.diag_clear_btn = QPushButton('Clear log')
            self.diag_clear_btn.setObjectName('Secondary')
            self.diag_clear_btn.clicked.connect(lambda: self.diagnostic_output.clear())
            top.addWidget(self.diag_run_selected_btn); top.addWidget(self.diag_run_all_btn); top.addWidget(self.diag_clear_btn); top.addStretch(1)
            lay.addLayout(top)
            split = QSplitter(Qt.Horizontal); lay.addWidget(split, 1)
            left = QWidget(); ll = QVBoxLayout(left)
            self.diagnostic_list = QListWidget(); self.diagnostic_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
            for label, module, description in DIAGNOSTICS:
                item = QListWidgetItem(f'{label}\n{module}')
                item.setData(Qt.UserRole, (label, module, description))
                item.setToolTip(description)
                self.diagnostic_list.addItem(item)
            self.available_checks_label = QLabel('Available diagnostic checks'); ll.addWidget(self.available_checks_label); ll.addWidget(self.diagnostic_list, 1)
            self.diagnostic_note_label = QLabel(tr(self.language, 'diagnostic_note'))
            self.diagnostic_note_label.setWordWrap(True); self.diagnostic_note_label.setObjectName('Tip'); ll.addWidget(self.diagnostic_note_label)
            split.addWidget(left)
            right = QWidget(); rl = QVBoxLayout(right)
            self.diagnostic_output = QTextEdit(); self.diagnostic_output.setReadOnly(True)
            self.diagnostic_output_label = QLabel('Diagnostic output'); rl.addWidget(self.diagnostic_output_label); rl.addWidget(self.diagnostic_output, 1)
            split.addWidget(right); split.setStretchFactor(0, 1); split.setStretchFactor(1, 2)
            return page


        def _page_help(self):
            page = QWidget(); lay = QVBoxLayout(page); lay.setContentsMargins(14, 12, 14, 12)
            self.help_title_widget = self._section_title('Help / User Guide', 'Programın içinde çalışan offline yardım merkezi. Anotasyonlu ekran görüntüleri, PDF/HTML manual, quick-start ve sorun giderme burada.')
            lay.addWidget(self.help_title_widget)
            actions = QHBoxLayout()
            self.help_pdf_btn = QPushButton('Open PDF manual'); self.help_pdf_btn.clicked.connect(self.open_help_pdf); actions.addWidget(self.help_pdf_btn)
            self.help_html_btn = QPushButton('Open HTML guide'); self.help_html_btn.setObjectName('Secondary'); self.help_html_btn.clicked.connect(self.open_help_html); actions.addWidget(self.help_html_btn)
            self.help_folder_btn = QPushButton('Open help folder'); self.help_folder_btn.setObjectName('Secondary'); self.help_folder_btn.clicked.connect(self.open_help_folder); actions.addWidget(self.help_folder_btn)
            self.help_capture_btn = QPushButton('Capture current GUI screenshot'); self.help_capture_btn.setObjectName('Secondary'); self.help_capture_btn.clicked.connect(self.capture_current_gui_screenshot); actions.addWidget(self.help_capture_btn)
            actions.addStretch(1); lay.addLayout(actions)

            split = QSplitter(Qt.Horizontal); lay.addWidget(split, 1)
            left = QWidget(); ll = QVBoxLayout(left)
            self.help_topic_list = QListWidget(); self.help_topic_list.itemSelectionChanged.connect(self.on_help_topic_selected)
            self.help_topics_label = QLabel('Help topics / annotated screenshots'); ll.addWidget(self.help_topics_label)
            ll.addWidget(self.help_topic_list, 1)
            self.help_note_label = QLabel('Bu yardım sistemi programın yanında offline gelir. Görseller önceden anotasyonlu hazırlanmıştır; ayrıca kendi aktif ekran görüntünü de alabilirsin.')
            note = self.help_note_label
            note.setObjectName('Tip'); note.setWordWrap(True); ll.addWidget(note)
            split.addWidget(left)

            right = QWidget(); rl = QVBoxLayout(right)
            self.help_title = QLabel('Select a help topic'); self.help_title.setObjectName('PageTitle'); rl.addWidget(self.help_title)
            self.help_text = QTextEdit(); self.help_text.setReadOnly(True); self.help_text.setMaximumHeight(210); rl.addWidget(self.help_text)
            self.help_image_scroll = QScrollArea(); self.help_image_scroll.setWidgetResizable(True)
            self.help_image_label = QLabel(); self.help_image_label.setAlignment(Qt.AlignCenter); self.help_image_scroll.setWidget(self.help_image_label)
            rl.addWidget(self.help_image_scroll, 1)
            row = QHBoxLayout()
            b = QPushButton('Open this screenshot'); b.setObjectName('Secondary'); b.clicked.connect(self.open_selected_help_asset); row.addWidget(b)
            b = QPushButton('Copy screenshot path'); b.setObjectName('Secondary'); b.clicked.connect(self.copy_selected_help_asset_path); row.addWidget(b)
            row.addStretch(1); rl.addLayout(row)
            split.addWidget(right); split.setStretchFactor(0, 0); split.setStretchFactor(1, 1); split.setSizes([360, 1180])
            self._load_help_topics()
            return page

        def _help_docs_dir(self):
            return Path(__file__).resolve().parents[1] / 'docs' / 'help'

        def _help_assets_dir(self):
            return Path(__file__).resolve().parent / 'help' / 'assets'

        def _help_topics(self):
            a = self._help_assets_dir()
            return [
                ('Quick start / Dashboard', a / '01_dashboard_annotated.png', 'Başlangıç ekranı: çalışma klasörü, hızlı aksiyonlar, aktif durum ve session log. İlk yapılacak işlem working directory seçmektir.'),
                ('Model selection and equation preview', a / '02_model_annotated.png', 'Model arama, kategori filtresi, model bilgileri ve profesyonel LaTeX denklem önizlemesi. Uzun denklemler scroll edilebilir.'),
                ('Multi data import and preprocessing', a / '03_data_import_annotated.png', 'Çoklu veri içeri aktarma, force-displacement dönüşümü, birim seçimi, smoothing/outlier/toe-region işlemleri ve anlık plot.'),
                ('Optimization and live iteration monitoring', a / '04_optimization_annotated.png', 'Optimizer seçimi, recommended adaptive bounds, uzun optimizasyon uyarısı, iterasyon tablosu ve canlı convergence plotları.'),
                ('Plugin Workbench', a / '05_plugin_workbench_annotated.png', '50 plugin kategori bazlı çalışır. Hedef olarak Current workspace veya Project Library içindeki eski projeler seçilebilir.'),
                ('Plugin Results viewer', a / '06_plugin_results_annotated.png', 'Plugin çıktılarını GUI içinde görüntüleme. PNG/CSV/TXT/HTML gibi dosyalar önizlenebilir, klasörler açılabilir.'),
                ('Project Library and .hyp2fit import', a / '07_project_library_annotated.png', '.hyp2fit ve eski .hfp projeleri içeri alınır, aktif workspace içine yüklenir veya plugin hedefi yapılır.'),
                ('Integrated Help System', a / '08_help_annotated.png', 'PDF/HTML manual açma, anotasyonlu ekranları izleme, aktif GUI ekran görüntüsü alma, 5 analiz örneğini takip etme ve sorun giderme.'),
                ('Language selector / 5 UI languages', a / '09_language_selector_annotated.png', 'Üst başlıktaki dil seçici ile Türkçe, İngilizce, Almanca, Fransızca ve İspanyolca ana arayüz metinleri arasında geçiş yapılır.'),
                ('Example 1 - Neo-Hookean quick calibration', a / '10_example_neo_hookean.png', 'Uniaxial + biaxial örnek datası ile Neo-Hookean modeli fit edilir; live iteration plot izlenir ve Excel/PDF çıktısı alınır.'),
                ('Example 2 - Mooney-Rivlin multi-mode fit', a / '11_example_mooney_rivlin.png', 'Uniaxial, biaxial ve planar dataları beraber yüklenir; 2-term Mooney-Rivlin modelinde ağırlıklı çoklu test kalibrasyonu yapılır.'),
                ('Example 3 - Yeoh force-displacement workflow', a / '12_example_yeoh_force_displacement.png', 'Ham force-displacement CSV içeri alınır; alan/gauge length ve birimlerle nominal stress-strain datasına dönüştürülür, smoothing/outlier kontrolleri uygulanır.'),
                ('Example 4 - Ogden global optimization', a / '13_example_ogden_global.png', 'Ogden gibi hassas modellerde differential_evolution veya recommended_adaptive_bounds kullanılır; uzun sürebilecek optimizasyon için uyarılar ve alternatifler takip edilir.'),
                ('Example 5 - Gent + FEA export verification', a / '14_example_gent_fea_export.png', 'Gent veya Van der Waals gibi finite extensibility modelleri için limit-domain kontrolü yapılır; Abaqus/ANSYS/CalculiX export ve single element target dosyaları üretilir.'),
            ]

        def _load_help_topics(self):
            if not hasattr(self, 'help_topic_list'):
                return
            self.help_topic_list.clear()
            for title, path, desc in self._help_topics():
                item = QListWidgetItem(title)
                item.setData(Qt.UserRole, {'title': title, 'path': str(path), 'desc': desc})
                item.setToolTip(desc)
                self.help_topic_list.addItem(item)
            if self.help_topic_list.count():
                self.help_topic_list.setCurrentRow(0)

        def on_help_topic_selected(self):
            items = self.help_topic_list.selectedItems() if hasattr(self, 'help_topic_list') else []
            if not items:
                return
            data = items[0].data(Qt.UserRole) or {}
            title = data.get('title','Help')
            path = Path(data.get('path',''))
            desc = data.get('desc','')
            self.help_title.setText(title)
            self.help_text.setPlainText(desc + '\n\nÖneri: PDF manual içindeki 5 örnek analizi adım adım takip et. HTML guide ekran görüntülerini ve örnek dosya adlarını tek sayfada gösterir.')
            if path.exists():
                pix = QPixmap(str(path))
                if not pix.isNull():
                    self.help_image_label.setPixmap(pix.scaled(self.help_image_scroll.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    return
            self.help_image_label.setText('Annotated screenshot not found:\n' + str(path))

        def _selected_help_asset_path(self):
            items = self.help_topic_list.selectedItems() if hasattr(self, 'help_topic_list') else []
            if not items:
                return None
            data = items[0].data(Qt.UserRole) or {}
            return Path(data.get('path',''))

        def open_help_pdf(self):
            p = self._help_docs_dir() / 'HyperFitPro_User_Guide.pdf'
            if p.exists(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
            else: QMessageBox.warning(self, 'Help PDF missing', str(p))

        def open_help_html(self):
            p = self._help_docs_dir() / 'HyperFitPro_User_Guide.html'
            if p.exists(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
            else: QMessageBox.warning(self, 'Help HTML missing', str(p))

        def open_help_folder(self):
            p = self._help_docs_dir()
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

        def open_selected_help_asset(self):
            p = self._selected_help_asset_path()
            if p and p.exists(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
            elif p: QMessageBox.warning(self, 'Screenshot missing', str(p))

        def copy_selected_help_asset_path(self):
            p = self._selected_help_asset_path()
            if p:
                QApplication.clipboard().setText(str(p))
                self._set_status('Help screenshot path copied.', 'ok')

        def capture_current_gui_screenshot(self):
            try:
                out = Path(self.workdir.text() or str(DEFAULT_ROOT)) / 'help_captures'
                out.mkdir(parents=True, exist_ok=True)
                name = 'hyperfitpro_gui_capture_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'
                path = out / name
                pix = self.grab()
                pix.save(str(path), 'PNG')
                self._set_status(f'GUI screenshot captured: {path}', 'ok')
                QMessageBox.information(self, 'Screenshot captured', f'Saved current GUI screenshot:\n{path}')
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))
            except Exception as exc:
                QMessageBox.critical(self, 'Screenshot failed', str(exc))

        def _set_section_title_text(self, widget, title, subtitle=''):
            try:
                labels = widget.findChildren(QLabel)
                if labels:
                    labels[0].setText(title)
                if len(labels) > 1:
                    labels[1].setText(subtitle)
            except Exception:
                pass

        def on_language_changed(self):
            try:
                self.language = self.language_combo.currentData() or 'tr'
            except Exception:
                self.language = 'tr'
            self._apply_language()
            if hasattr(self, 'language_combo'):
                self._set_status(tr(self.language, 'language_changed', language=LANGUAGES.get(self.language, self.language)), 'ok')

        def _apply_language(self):
            lang = getattr(self, 'language', 'tr')
            self.setWindowTitle('HyperFitPro Studio 1.0 - ' + LANGUAGES.get(lang, 'Türkçe'))
            if hasattr(self, 'header_title'): self.header_title.setText(tr(lang, 'app_title'))
            if hasattr(self, 'header_subtitle'): self.header_subtitle.setText(tr(lang, 'app_subtitle'))
            if hasattr(self, 'language_label'): self.language_label.setText(tr(lang, 'language'))
            # Toolbar
            for attr, key in [
                ('action_sidebar','menu'), ('action_import_project','import_project_short'), ('action_save_project','save_short'),
                ('action_run_history','runs_short'), ('action_diagnostics','checks_short'), ('action_help','help_short'), ('action_open_workdir','open_dir_short')]:
                act = getattr(self, attr, None)
                if act is not None:
                    act.setText(('☰ ' if attr == 'action_sidebar' else '') + tr(lang, key))
            # Sidebar tree
            if hasattr(self, 'nav_title_label'): self.nav_title_label.setText(tr(lang, 'navigation'))
            if hasattr(self, 'hide_sidebar_button'): self.hide_sidebar_button.setText(tr(lang, 'hide'))
            if hasattr(self, 'nav_tree'):
                for gi in range(self.nav_tree.topLevelItemCount()):
                    root = self.nav_tree.topLevelItem(gi)
                    if root:
                        gkey = root.data(0, Qt.UserRole + 1) or ''
                        root.setText(0, tr(lang, gkey))
                        for ci in range(root.childCount()):
                            ch = root.child(ci); ckey = ch.data(0, Qt.UserRole + 1) or ''
                            ch.setText(0, f'{ci+1:02d}  ' + tr(lang, ckey))
            # Overview pages
            if hasattr(self, 'stack'):
                for idx, gkey in [(10,'workflow'),(11,'plugins'),(12,'projects'),(13,'tools')]:
                    if idx < self.stack.count():
                        page = self.stack.widget(idx)
                        if hasattr(page, 'title_widget'):
                            self._set_section_title_text(page.title_widget, tr(lang, gkey), tr(lang, gkey + '_overview_subtitle'))
                        if hasattr(page, 'intro_label'): page.intro_label.setText(tr(lang, gkey + '_overview_text'))
                        if hasattr(page, 'flow_box'): page.flow_box.setTitle(tr(lang, 'guided_flow'))
                        if hasattr(page, 'details'): page.details.setPlainText(tr(lang, gkey + '_overview_steps'))
                        if hasattr(page, 'flow_buttons'):
                            for btn, key, n in page.flow_buttons:
                                btn.setText(f'{n}. ' + tr(lang, key))
            # Dashboard
            if hasattr(self, 'dashboard_title_widget'): self._set_section_title_text(self.dashboard_title_widget, tr(lang, 'dashboard'), tr(lang, 'dashboard_subtitle'))
            if hasattr(self, 'dashboard_flow_label'): self.dashboard_flow_label.setText(tr(lang, 'main_flow'))
            for attr, key in [('workdir_group','working_directory'),('current_workspace_group','current_workspace'),('quick_actions_group','quick_actions'),('recommended_workflow_group','recommended_workflow')]:
                w = getattr(self, attr, None)
                if w is not None: w.setTitle(tr(lang, key))
            if hasattr(self, 'workdir_select_btn'): self.workdir_select_btn.setText(tr(lang, 'select'))
            if hasattr(self, 'workdir_open_btn'): self.workdir_open_btn.setText(tr(lang, 'open'))
            if hasattr(self, 'session_log_label'): self.session_log_label.setText(tr(lang, 'session_log'))
            if hasattr(self, 'workflow_text'): self.workflow_text.setText(tr(lang, 'workflow_guidance'))
            for btn, key in getattr(self, 'quick_action_buttons', []): btn.setText(tr(lang, key))
            # Model page
            if hasattr(self, 'model_title_widget'): self._set_section_title_text(self.model_title_widget, tr(lang, 'model_selection'), tr(lang, 'model_subtitle'))
            if hasattr(self, 'model_search_label'): self.model_search_label.setText(tr(lang, 'search'))
            if hasattr(self, 'model_category_label'): self.model_category_label.setText(tr(lang, 'category'))
            if hasattr(self, 'model_search'): self.model_search.setPlaceholderText(tr(lang, 'model_search_placeholder'))
            if hasattr(self, 'copy_model_details_btn'): self.copy_model_details_btn.setText(tr(lang, 'copy_model_details'))
            # Data page
            if hasattr(self, 'data_title_widget'): self._set_section_title_text(self.data_title_widget, tr(lang, 'data_import'), tr(lang, 'data_subtitle'))
            if hasattr(self, 'data_import_settings_group'): self.data_import_settings_group.setTitle(tr(lang, 'import_settings'))
            if hasattr(self, 'advanced_preprocessing_group'): self.advanced_preprocessing_group.setTitle(tr(lang, 'advanced_preprocessing'))
            if hasattr(self, 'strain_percent_check'): self.strain_percent_check.setText(tr(lang, 'strain_percent'))
            if hasattr(self, 'zero_offset_check'): self.zero_offset_check.setText(tr(lang, 'zero_offset'))
            for label, key in getattr(self, 'data_form_labels', []): label.setText(tr(lang, key))
            for label, key in getattr(self, 'preprocess_form_labels', []): label.setText(tr(lang, key))
            for attr, key in [('add_data_btn','add_files'),('remove_data_btn','remove_selected'),('autobalance_btn','auto_balance_weights')]:
                w = getattr(self, attr, None)
                if w is not None: w.setText(tr(lang, key))
            if hasattr(self, 'data_table'): self.data_table.setHorizontalHeaderLabels([tr(lang,'mode_label'), tr(lang,'file_label'), tr(lang,'points_label'), tr(lang,'weight_label'), tr(lang,'preprocessing_label')])
            # Optimization page
            if hasattr(self, 'optimization_title_widget'): self._set_section_title_text(self.optimization_title_widget, tr(lang, 'optimization'), tr(lang, 'optimization_subtitle'))
            if hasattr(self, 'run_controls_group'): self.run_controls_group.setTitle(tr(lang, 'run_controls'))
            if hasattr(self, 'optimization_tip'): self.optimization_tip.setText(tr(lang, 'optimization_tip'))
            for label, key in getattr(self, 'optimization_form_labels', []): label.setText(tr(lang, key))
            for attr, key in [('run_btn','run_optimization'),('pause_btn','pause'),('stop_btn','stop')]:
                w = getattr(self, attr, None)
                if w is not None: w.setText(tr(lang, key))
            if hasattr(self, 'iter_table'): self.iter_table.setHorizontalHeaderLabels([tr(lang,'eval_label'), tr(lang,'objective_label'), tr(lang,'best_objective_label'), tr(lang,'parameters_label')])
            # Report/export
            if hasattr(self, 'report_title_widget'): self._set_section_title_text(self.report_title_widget, tr(lang, 'report_export'), tr(lang, 'report_subtitle'))
            for btn, key in getattr(self, 'report_action_buttons', []): btn.setText(tr(lang, key))
            if hasattr(self, 'results_table'): self.results_table.setHorizontalHeaderLabels([tr(lang,'parameter_metric'), tr(lang,'value_label')])
            if hasattr(self, 'report_note'): self.report_note.setPlainText(tr(lang, 'report_note'))
            # Plugins
            if hasattr(self, 'plugin_title_widget'): self._set_section_title_text(self.plugin_title_widget, tr(lang, 'plugin_workbench'), tr(lang, 'plugin_subtitle'))
            if hasattr(self, 'plugin_target_label'): self.plugin_target_label.setText(tr(lang, 'target_project'))
            for attr, key in [('plugin_import_btn','import_project'),('plugin_reload_btn','reload_plugins'),('plugin_run_selected_btn','run_selected'),('plugin_run_category_btn','run_category')]:
                w = getattr(self, attr, None)
                if w is not None: w.setText(tr(lang, key))
            if hasattr(self, 'plugin_categories_label'): self.plugin_categories_label.setText(tr(lang, 'plugin_categories'))
            if hasattr(self, 'plugin_detail_label'): self.plugin_detail_label.setText(tr(lang, 'plugin_detail'))
            if hasattr(self, 'plugin_tree'): self.plugin_tree.setHeaderLabels([tr(lang, 'plugin_categories')])
            if hasattr(self, 'flat_plugin_list_label'): self.flat_plugin_list_label.setText(tr(lang, 'flat_plugin_list'))
            if hasattr(self, 'plugin_matrix'): self.plugin_matrix.setHorizontalHeaderLabels([tr(lang,'category'), 'ID', tr(lang,'name'), tr(lang,'requires'), tr(lang,'outputs')])
            if hasattr(self, 'plugin_results_title_widget'): self._set_section_title_text(self.plugin_results_title_widget, tr(lang, 'plugin_results'), tr(lang, 'plugin_results_subtitle'))
            if hasattr(self, 'plugin_results_table'): self.plugin_results_table.setHorizontalHeaderLabels([tr(lang,'time_label'), tr(lang,'target'), tr(lang,'plugin'), 'OK', tr(lang,'message'), tr(lang,'files')])
            # Projects
            if hasattr(self, 'project_title_widget'): self._set_section_title_text(self.project_title_widget, tr(lang, 'project_library'), tr(lang, 'project_subtitle'))
            for btn, key in getattr(self, 'project_action_buttons', []): btn.setText(tr(lang, key))
            if hasattr(self, 'project_table'): self.project_table.setHorizontalHeaderLabels([tr(lang,'file_label'), tr(lang,'project'), tr(lang,'model'), tr(lang,'datasets'), tr(lang,'last_run'), tr(lang,'modified'), tr(lang,'notes')])
            # Diagnostics
            if hasattr(self, 'diagnostics_title_widget'): self._set_section_title_text(self.diagnostics_title_widget, tr(lang, 'diagnostics'), tr(lang, 'diagnostics_subtitle'))
            for attr, key in [('diag_run_selected_btn','run_selected_checks'),('diag_run_all_btn','run_all_checks'),('diag_clear_btn','clear_log')]:
                w = getattr(self, attr, None)
                if w is not None: w.setText(tr(lang, key))
            if hasattr(self, 'available_checks_label'): self.available_checks_label.setText(tr(lang, 'available_checks'))
            if hasattr(self, 'diagnostic_note_label'): self.diagnostic_note_label.setText(tr(lang, 'diagnostic_note'))
            if hasattr(self, 'diagnostic_output_label'): self.diagnostic_output_label.setText(tr(lang, 'diagnostic_output'))
            # Help
            if hasattr(self, 'help_title_widget'): self._set_section_title_text(self.help_title_widget, tr(lang, 'help_page_title'), tr(lang, 'help_page_subtitle'))
            for attr, key in [('help_pdf_btn','open_pdf_manual'),('help_html_btn','open_html_guide'),('help_folder_btn','open_help_folder'),('help_capture_btn','capture_screenshot')]:
                w = getattr(self, attr, None)
                if w is not None: w.setText(tr(lang, key))
            if hasattr(self, 'help_topics_label'): self.help_topics_label.setText(tr(lang, 'help_topics'))
            if hasattr(self, 'help_note_label'): self.help_note_label.setText(tr(lang, 'help_note'))
            if hasattr(self, 'help_title') and self.help_title.text() in ('Select a help topic','Bir help konusu seç','Hilfethema auswählen','Choisir un sujet d’aide','Seleccione un tema de ayuda'):
                self.help_title.setText(tr(lang, 'select_topic'))
            if hasattr(self, 'state_label') and ('No model/data/run' in self.state_label.text() or 'Henüz model' in self.state_label.text()): self.state_label.setText(tr(lang, 'no_model_data_run'))
        def _apply_theme(self):
            self.setStyleSheet('''
                QMainWindow, QWidget { background:#ffffff; color:#111827; font-family:Segoe UI, Arial, sans-serif; font-size:9.3pt; }
                QFrame#Header { background:#ffffff; border-bottom:1px solid #e5e7eb; }
                QFrame#Sidebar { background:#f8fafc; border-right:1px solid #e5e7eb; }
                QLabel#Title { font-size:20px; font-weight:800; color:#0f172a; }
                QLabel#Subtitle, QLabel#PageSubtitle { color:#64748b; font-size:9.0pt; }
                QLabel#PageTitle { font-size:18px; font-weight:760; color:#0f172a; }
                QLabel#SideTitle { font-size:11px; font-weight:760; color:#334155; letter-spacing:0.35px; }
                QLabel#BadgeOK { color:#065f46; background:#ecfdf5; border:1px solid #a7f3d0; border-radius:9px; padding:3px 8px; font-weight:700; }
                QLabel#BadgeWarn { color:#92400e; background:#fffbeb; border:1px solid #fcd34d; border-radius:9px; padding:3px 8px; font-weight:700; }
                QLabel#BadgeBlock { color:#991b1b; background:#fef2f2; border:1px solid #fecaca; border-radius:9px; padding:3px 8px; font-weight:700; }
                QLabel#Tip { color:#334155; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:6px; }
                QLabel#FlowPill { color:#1e3a8a; background:#eff6ff; border:1px solid #bfdbfe; border-radius:13px; padding:8px 12px; font-weight:760; }
                QLabel#FlowArrow { color:#2563eb; font-size:18px; font-weight:800; }
                QGroupBox { font-weight:700; border:1px solid #dbe2ea; border-radius:9px; margin-top:10px; padding:9px; background:#ffffff; }
                QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 5px; background:#ffffff; color:#0f172a; }
                QPushButton { background:#2563eb; color:white; border:none; border-radius:7px; padding:5px 10px; font-weight:700; min-height:22px; }
                QPushButton:hover { background:#1d4ed8; }
                QPushButton#Secondary { background:#f8fafc; color:#0f172a; border:1px solid #cbd5e1; }
                QPushButton#Secondary:hover { background:#eef2f7; }
                QPushButton#Danger { background:#dc2626; color:white; }
                QPushButton#Flat { background:#ffffff; color:#475569; border:1px solid #e2e8f0; padding:3px 7px; min-height:20px; }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { padding:4px 6px; border:1px solid #cbd5e1; border-radius:7px; background:#ffffff; min-height:22px; }
                QTextEdit { background:#ffffff; color:#111827; font-family:Consolas, 'Cascadia Mono', monospace; font-size:9.0pt; border:1px solid #e2e8f0; border-radius:8px; padding:6px; }
                QTableWidget { background:#ffffff; gridline-color:#e5e7eb; border:1px solid #e2e8f0; border-radius:8px; selection-background-color:#dbeafe; selection-color:#111827; }
                QHeaderView::section { background:#f8fafc; color:#334155; border:0px; border-right:1px solid #e2e8f0; border-bottom:1px solid #e2e8f0; padding:5px; font-weight:700; }
                QTreeWidget#NavTree, QTreeWidget { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:3px; outline:0; }
                QTreeWidget::branch { background:#f8fafc; }
                QTreeWidget::item { padding:4px 5px; border-radius:5px; min-height:20px; }
                QTreeWidget::item:hover { background:#eef2ff; color:#1e3a8a; }
                QTreeWidget::item:selected { background:#dbeafe; color:#1e3a8a; font-weight:700; }
                QListWidget { background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; }
                QListWidget::item { padding:4px 6px; min-height:20px; }
                QToolBar { background:#ffffff; border-bottom:1px solid #e5e7eb; spacing:4px; padding:2px 4px; }
                QToolButton { padding:4px 7px; border-radius:6px; color:#334155; background:#ffffff; }
                QToolButton:hover { background:#eff6ff; color:#1d4ed8; }
                QStatusBar { background:#ffffff; border-top:1px solid #e5e7eb; color:#475569; }
            ''')

        # -------------------- Navigation/status --------------------
        def toggle_sidebar(self):
            self.sidebar.setVisible(not self.sidebar.isVisible())

        def on_nav_clicked(self, item, col):
            idx = item.data(0, Qt.UserRole)
            if idx is not None:
                self.stack.setCurrentIndex(int(idx))

        def _set_status(self, text, state='warn'):
            self.status_badge.setText(text)
            self.status_badge.setObjectName({'ok':'BadgeOK', 'block':'BadgeBlock'}.get(state, 'BadgeWarn'))
            self.status_badge.style().unpolish(self.status_badge); self.status_badge.style().polish(self.status_badge)
            try: self.status.showMessage(text, 5000)
            except Exception: pass

        def log(self, msg):
            line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
            if hasattr(self, 'dashboard_log'):
                self.dashboard_log.append(line)
            if hasattr(self, 'run_log'):
                self.run_log.append(line)

        def _update_state_label(self):
            model = self.current_model()
            parts = []
            parts.append(f'Model: {model.number} - {model.name}' if model else 'Model: not selected')
            parts.append(f'Datasets: {len(self.datasets)}')
            if self.last_result:
                parts.append(f'Last run: RMSE={self.last_result.rmse:.6g}, R²={self.last_result.r2:.6g}')
            if self.last_run_folder:
                parts.append(f'Run folder: {self.last_run_folder}')
            self.state_label.setText('\n'.join(parts))

        # -------------------- Model --------------------
        def _load_models_into_combo(self):
            if not hasattr(self, 'model_combo'):
                return
            search = (self.model_search.text() if hasattr(self, 'model_search') else '').lower().strip()
            current_category = self.category_filter.currentText() if hasattr(self, 'category_filter') else 'All categories'
            # categories
            cats = sorted({getattr(m, 'category', 'Uncategorized') or 'Uncategorized' for m in self.models})
            if hasattr(self, 'category_filter') and self.category_filter.count() <= 1:
                for c in cats: self.category_filter.addItem(c)
            self.model_combo.blockSignals(True); self.model_combo.clear(); self.model_map.clear()
            for m in sorted(self.models, key=lambda x: x.number):
                label = f'{m.number:03d} | {m.name} [{getattr(m, "category", "")}]'
                blob = ' '.join([label, getattr(m, 'family', ''), ' '.join(getattr(m, 'application_tags', []) or [])]).lower()
                if current_category != 'All categories' and getattr(m, 'category', '') != current_category:
                    continue
                if search and search not in blob:
                    continue
                self.model_combo.addItem(label); self.model_map[label] = m
            self.model_combo.blockSignals(False)
            if self.model_combo.count():
                self.model_combo.setCurrentIndex(0); self.on_model_changed()

        def current_model(self):
            if not hasattr(self, 'model_combo'):
                return None
            return self.model_map.get(self.model_combo.currentText())

        def _fmt_bound_value(self, value):
            try:
                value = float(value)
                if abs(value) >= 1.0e4 or (abs(value) > 0 and abs(value) < 1.0e-3):
                    return f'{value:.6e}'
                return f'{value:.6g}'
            except Exception:
                return str(value)

        def _parameter_bounds_html(self, m):
            specs = list(getattr(m, 'parameter_specs', []) or [])
            if not specs:
                return '<p><b>Parameter bounds:</b> No parameter definition found for this model.</p>'
            scale_note = ''
            rows = []
            resolved = None
            if hasattr(self, 'datasets') and self.datasets:
                try:
                    S = stress_scale(self.datasets)
                    max_I1 = max_invariant_from_test_inputs([(d.mode, d.x) for d in self.datasets])
                    lb, ub, x0 = m.resolved_bounds(S, max_I1)
                    resolved = {spec.name: (lb[i], ub[i], x0[i]) for i, spec in enumerate(specs)}
                    scale_note = '<p class="note">Resolved bounds are shown using current imported data scale: stress scale = %s, max I<sub>1</sub> = %s.</p>' % (self._fmt_bound_value(S), self._fmt_bound_value(max_I1))
                except Exception as exc:
                    scale_note = '<p class="warn">Resolved bounds could not be calculated yet: %s</p>' % html.escape(str(exc))
            for spec in specs:
                rlo = rhi = rini = '-'
                if resolved and spec.name in resolved:
                    rlo, rhi, rini = [self._fmt_bound_value(v) for v in resolved[spec.name]]
                rows.append(
                    '<tr>'
                    '<td><b>%s</b></td>' % html.escape(spec.name) +
                    '<td>%s</td>' % html.escape(self._fmt_bound_value(spec.lower)) +
                    '<td>%s</td>' % html.escape(self._fmt_bound_value(spec.upper)) +
                    '<td>%s</td>' % (html.escape(self._fmt_bound_value(spec.initial)) if spec.initial is not None else '-') +
                    '<td>%s</td>' % html.escape(getattr(spec, 'scale', '-') or '-') +
                    '<td>%s</td>' % html.escape(getattr(spec, 'unit', '-') or '-') +
                    '<td>%s</td>' % html.escape(rlo) +
                    '<td>%s</td>' % html.escape(rhi) +
                    '<td>%s</td>' % html.escape(rini) +
                    '<td>%s</td>' % html.escape(getattr(spec, 'description', '') or '-') +
                    '</tr>'
                )
            return '''
                <h3>Parameter min / max bounds</h3>
                <p class="note">Raw bounds are the model-library optimizer search limits. Stress-scaled parameters are automatically resolved after data import.</p>
                %s
                <table>
                    <tr>
                        <th>Parameter</th><th>Raw min</th><th>Raw max</th><th>Initial</th><th>Scale</th><th>Unit</th>
                        <th>Resolved min</th><th>Resolved max</th><th>Resolved initial</th><th>Description</th>
                    </tr>
                    %s
                </table>
            ''' % (scale_note, ''.join(rows))

        def _model_details_html(self, m):
            eq = getattr(m, 'equation_latex', '') or '-'
            def esc(x): return html.escape(str(x if x is not None else '-'))
            tags = ', '.join(getattr(m, 'application_tags', []) or []) or '-'
            mats = ', '.join(getattr(m, 'material_classes', []) or []) or '-'
            tests = ', '.join(getattr(m, 'recommended_tests', []) or []) or '-'
            params = ', '.join([p.name for p in getattr(m, 'parameter_specs', [])]) or '-'
            return f'''
            <html>
            <head>
            <style>
                body {{ font-family: Segoe UI, Arial, sans-serif; font-size: 9.5pt; color: #172033; background: #ffffff; }}
                h2 {{ margin: 0 0 6px 0; color: #0f172a; font-size: 15pt; }}
                h3 {{ margin: 12px 0 5px 0; color: #1d4ed8; font-size: 11pt; }}
                .meta {{ color: #475569; margin-bottom: 8px; }}
                .equation {{ font-family: Cambria Math, Times New Roman, serif; font-size: 15pt; color: #111827; background: #f8fafc; border: 1px solid #dbeafe; border-radius: 8px; padding: 10px; margin: 6px 0; white-space: nowrap; }}
                .latex {{ font-family: Consolas, Cascadia Mono, monospace; font-size: 9pt; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px; white-space: nowrap; }}
                .note {{ color: #64748b; }}
                .warn {{ color: #b45309; }}
                table {{ border-collapse: collapse; margin-top: 6px; }}
                th {{ background: #eff6ff; color: #1e3a8a; border: 1px solid #cbd5e1; padding: 5px 8px; font-weight: 700; }}
                td {{ border: 1px solid #e2e8f0; padding: 5px 8px; vertical-align: top; }}
                .section {{ margin-top: 10px; }}
            </style>
            </head>
            <body>
                <h2>{esc(m.number)} - {esc(m.name)}</h2>
                <div class="meta"><b>Category:</b> {esc(getattr(m, 'category', ''))} &nbsp; | &nbsp; <b>Family:</b> {esc(getattr(m, 'family', ''))} &nbsp; | &nbsp; <b>Complexity:</b> {esc(getattr(m, 'complexity_level', ''))}</div>
                <div class="meta"><b>Recommended tests:</b> {esc(tests)} &nbsp; | &nbsp; <b>Parameters:</b> {esc(params)}</div>
                <div class="meta"><b>Material classes:</b> {esc(mats)} &nbsp; | &nbsp; <b>Application tags:</b> {esc(tags)}</div>

                <h3>Professional equation</h3>
                <div class="equation">W = {esc(eq)}</div>

                <h3>Copyable LaTeX equation</h3>
                <div class="latex">{esc(eq)}</div>

                {self._parameter_bounds_html(m)}

                <h3>Calibration notes</h3>
                <div class="section">{esc(getattr(m, 'calibration_notes', '') or '-')}</div>

                <h3>Parameterization notes</h3>
                <div class="section">{esc(getattr(m, 'parameterization_notes', '') or '-')}</div>

                <h3>Limitations</h3>
                <div class="section">{esc(getattr(m, 'limitation_notes', '') or '-')}</div>

                <h3>General notes</h3>
                <div class="section">{esc(getattr(m, 'notes', '') or '-')}</div>
            </body>
            </html>
            '''

        def _model_details_plain_text(self, m):
            lines = [
                f'Model: {m.number} - {m.name}',
                f'Category: {getattr(m, "category", "")}',
                f'Family: {getattr(m, "family", "")}',
                f'Complexity: {getattr(m, "complexity_level", "")}',
                f'Recommended tests: {", ".join(getattr(m, "recommended_tests", []) or [])}',
                f'Equation LaTeX: {getattr(m, "equation_latex", "")}',
                '',
                'Parameter bounds:',
            ]
            for spec in getattr(m, 'parameter_specs', []) or []:
                lines.append(f'  - {spec.name}: min={self._fmt_bound_value(spec.lower)}, max={self._fmt_bound_value(spec.upper)}, initial={self._fmt_bound_value(spec.initial) if spec.initial is not None else "-"}, scale={spec.scale}, unit={spec.unit}, description={spec.description or "-"}')
            lines += ['', 'Calibration notes:', getattr(m, 'calibration_notes', '') or '-', '', 'Limitations:', getattr(m, 'limitation_notes', '') or '-', '', 'Notes:', getattr(m, 'notes', '') or '-']
            return '\n'.join(lines)

        def on_model_changed(self):
            m = self.current_model()
            if not m: return
            self.model_info.setHtml(self._model_details_html(m))
            self._last_model_details_plain = self._model_details_plain_text(m)
            self._update_state_label()

        def copy_equation(self):
            m = self.current_model()
            text = getattr(self, '_last_model_details_plain', '') or (self._model_details_plain_text(m) if m else '')
            QApplication.clipboard().setText(text)
            self._set_status('Model details copied to clipboard.', 'ok')

        # -------------------- Data --------------------
        def _load_kwargs(self):
            return dict(
                data_kind=self.data_kind_combo.currentText(), force_unit=self.force_unit_combo.currentText(),
                length_unit=self.length_unit_combo.currentText(), stress_unit=self.stress_unit_combo.currentText(),
                area=self.area_spin.value() or None, width=self.width_spin.value() or None,
                thickness=self.thickness_spin.value() or None, diameter=self.diameter_spin.value() or None,
                gauge_length=self.gauge_spin.value() or None, strain_percent=self.strain_percent_check.isChecked(),
                branch=self.branch_combo.currentText(), zero_offset=self.zero_offset_check.isChecked(),
                smooth_method=self.smooth_method_combo.currentText(), smooth_window=self.smooth_window_spin.value(),
                outlier_method=self.outlier_combo.currentText(), toe_remove_fraction=self.toe_fraction_spin.value(),
                downsample_max_points=self.downsample_spin.value(), region_weighting=self.region_weight_combo.currentText(),
            )

        def add_data(self):
            paths, _ = QFileDialog.getOpenFileNames(self, 'Import one or multiple test data files', '', 'Data files (*.csv *.txt *.xlsx *.xls);;All files (*.*)')
            if not paths: return
            failures = []
            for path in paths:
                try:
                    d = load_test_data(path, self.mode_combo.currentText(), self.weight.value(), **self._load_kwargs())
                    self.datasets.append(d)
                    self.log(f'Dataset imported: {Path(path).name} | mode={d.mode} | n={len(d.x)}')
                except Exception as exc:
                    failures.append(f'{Path(path).name}: {exc}')
            self._refresh_data_table(); self.plot_imported_data(); self.on_model_changed(); self._update_state_label(); self._refresh_project_targets()
            if failures:
                QMessageBox.warning(self, 'Import warnings', 'Some files failed:\n' + '\n'.join(failures[:12]))

        def remove_data(self):
            rows = sorted({i.row() for i in self.data_table.selectedItems()}, reverse=True)
            for r in rows:
                if 0 <= r < len(self.datasets):
                    self.datasets.pop(r)
            self._refresh_data_table(); self.plot_imported_data(); self.on_model_changed(); self._update_state_label()

        def auto_balance_weights(self):
            try:
                auto_balance_dataset_weights(self.datasets)
                self._refresh_data_table(); self.on_model_changed(); self._set_status('Dataset weights auto-balanced.', 'ok')
            except Exception as exc:
                QMessageBox.warning(self, 'Auto-balance failed', str(exc))

        def _refresh_data_table(self):
            self.data_table.setRowCount(len(self.datasets))
            for r, d in enumerate(self.datasets):
                vals = [d.mode, Path(d.source_file).name, str(len(d.x)), f'{d.weight:.4g}', json.dumps(getattr(d, 'preprocessing', {}) or {}, default=str)[:180]]
                for c, v in enumerate(vals):
                    self.data_table.setItem(r, c, QTableWidgetItem(v))

        def plot_imported_data(self):
            if not hasattr(self, 'data_canvas'): return
            self.data_canvas.fig.clear(); ax = self.data_canvas.fig.add_subplot(111)
            if not self.datasets:
                ax.text(0.5, 0.5, 'No data imported yet', ha='center', va='center'); ax.axis('off')
            else:
                for d in self.datasets:
                    ax.plot(d.x, d.y, marker='o', ms=3, lw=1, label=f'{d.mode} | {Path(d.source_file).stem}')
                ax.set_xlabel('test input / strain / stretch / gamma / volumetric input')
                ax.set_ylabel('nominal stress / shear / pressure [MPa]')
                ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
            self.data_canvas.fig.tight_layout(); self.data_canvas.draw_idle()

        # -------------------- Optimization --------------------
        def start_fit(self):
            model = self.current_model()
            if model is None:
                QMessageBox.warning(self, 'Run optimization', 'Select a model first.'); self.stack.setCurrentIndex(1); return
            if not self.datasets:
                QMessageBox.warning(self, 'Run optimization', 'Import at least one dataset first.'); self.stack.setCurrentIndex(2); return
            method = self.method.currentText()
            slow = method in {'differential_evolution','dual_annealing','basinhopping','shgo','hybrid_global_local','recommended_adaptive_bounds','multi_start_lsq','regularized_multi_start'}
            if slow:
                msg = 'This optimization method can take longer, especially for high-parameter models. Recommended fast alternative: fast_local_refine or least_squares_trf after checking bounds. Continue?'
                if QMessageBox.question(self, 'Long optimization warning', msg) != QMessageBox.Yes:
                    return
            self.live_trace = []; self.iter_table.setRowCount(0); self.live_canvas.fig.clear(); self.live_canvas.draw_idle(); self.fit_preview_canvas.fig.clear(); self.fit_preview_canvas.draw_idle()
            self.run_btn.setEnabled(False); self.pause_btn.setEnabled(True); self.stop_btn.setEnabled(True); self.pause_btn.setText('Pause')
            self.worker = FitWorker(model, list(self.datasets), self.workdir.text(), method, self.maxeval.value(),
                                    regularization=self.reg_strength.value(), regularization_type=self.reg_type.currentText(),
                                    validation_fraction=self.validation_fraction.value(), workers=self.worker_count.value(),
                                    near_bound_penalty=self.near_bound_penalty.value())
            self.worker.log_signal.connect(self.log)
            self.worker.iteration_signal.connect(self.on_iteration)
            self.worker.done_signal.connect(self.on_fit_done)
            self.worker.fail_signal.connect(self.on_fit_failed)
            self.worker.start()
            self._set_status('Optimization running...', 'warn')

        def pause_resume_fit(self):
            if not self.worker: return
            try:
                if self.pause_btn.text().lower().startswith('pause'):
                    self.worker.controller.pause(); self.pause_btn.setText('Resume'); self.log('Optimization pause requested.')
                else:
                    self.worker.controller.resume(); self.pause_btn.setText('Pause'); self.log('Optimization resumed.')
            except Exception as exc:
                QMessageBox.warning(self, 'Pause/resume failed', str(exc))

        def stop_fit(self):
            if self.worker:
                self.worker.controller.stop(); self.log('Optimization stop requested.')

        def on_iteration(self, payload):
            rec = payload.get('record', {}) if isinstance(payload, dict) else {}
            self.live_trace.append(rec)
            r = self.iter_table.rowCount(); self.iter_table.insertRow(r)
            params = payload.get('parameters', {}) if isinstance(payload, dict) else {}
            vals = [str(rec.get('eval','')), f"{rec.get('objective', np.nan):.6e}", f"{rec.get('best_objective', np.nan):.6e}", ', '.join(f'{k}={v:.5g}' for k, v in params.items())]
            for c, v in enumerate(vals): self.iter_table.setItem(r, c, QTableWidgetItem(v))
            if self.iter_table.rowCount() > 800:
                self.iter_table.removeRow(0)
            now = time.time()
            if now - self._last_live_draw > 0.35:
                self._last_live_draw = now; self.update_live_plots(payload)

        def update_live_plots(self, payload=None):
            trace = self.live_trace
            self.live_canvas.fig.clear(); ax = self.live_canvas.fig.add_subplot(111)
            if trace:
                xs = [t.get('eval', i) for i, t in enumerate(trace)]
                obj = [t.get('objective', np.nan) for t in trace]
                best = [t.get('best_objective', np.nan) for t in trace]
                ax.semilogy(xs, obj, label='objective', alpha=0.75)
                ax.semilogy(xs, best, label='best objective', lw=2)
                ax.set_xlabel('evaluation'); ax.set_ylabel('objective'); ax.grid(True, alpha=0.25); ax.legend()
            else:
                ax.text(0.5,0.5,'Waiting for iterations...',ha='center',va='center'); ax.axis('off')
            self.live_canvas.fig.tight_layout(); self.live_canvas.draw_idle()
            # Fit preview from best parameters
            model = self.current_model(); params = (payload or {}).get('best_parameters', {}) if isinstance(payload, dict) else {}
            self.fit_preview_canvas.fig.clear(); ax2 = self.fit_preview_canvas.fig.add_subplot(111)
            if model and self.datasets and params:
                for d in self.datasets:
                    ax2.plot(d.x, d.y, 'o', ms=3, label=f'{d.mode} data')
                    try:
                        xx = np.linspace(float(np.nanmin(d.x)), float(np.nanmax(d.x)), 120)
                        yy = model.predict_nominal(d.mode, xx, params)
                        ax2.plot(xx, yy, '-', lw=1.6, label=f'{d.mode} live fit')
                    except Exception:
                        pass
                ax2.set_xlabel('test input'); ax2.set_ylabel('response [MPa]'); ax2.grid(True, alpha=0.25); ax2.legend(fontsize=8)
            else:
                ax2.text(0.5,0.5,'Live fit preview will appear here',ha='center',va='center'); ax2.axis('off')
            self.fit_preview_canvas.fig.tight_layout(); self.fit_preview_canvas.draw_idle()

        def on_fit_done(self, result, run_folder):
            self.last_result = result; self.last_run_folder = run_folder; self.last_model = self.current_model(); self.last_datasets = list(self.datasets)
            self.run_btn.setEnabled(True); self.pause_btn.setEnabled(False); self.stop_btn.setEnabled(False); self.worker = None
            self._set_status('Optimization completed.', 'ok'); self.log(f'Run completed: {run_folder}')
            self._refresh_results_table(); self._update_state_label(); self._refresh_project_targets()
            self.stack.setCurrentIndex(4)

        def on_fit_failed(self, error_text):
            self.run_btn.setEnabled(True); self.pause_btn.setEnabled(False); self.stop_btn.setEnabled(False); self.worker = None
            self._set_status('Optimization failed.', 'block'); self.log(error_text)
            QMessageBox.critical(self, 'Run failed', error_text)

        def _refresh_results_table(self):
            rows = []
            if self.last_result:
                rows += [('success', str(self.last_result.success)), ('method', self.last_result.method), ('RMSE', f'{self.last_result.rmse:.8g}'), ('MAE', f'{self.last_result.mae:.8g}'), ('R²', f'{self.last_result.r2:.8g}'), ('AICc', str(self.last_result.aicc)), ('BIC', str(self.last_result.bic))]
                rows += [(f'parameter: {k}', f'{v:.12g}') for k, v in self.last_result.parameters.items()]
            self.results_table.setRowCount(len(rows))
            for r, (k, v) in enumerate(rows):
                self.results_table.setItem(r, 0, QTableWidgetItem(k)); self.results_table.setItem(r, 1, QTableWidgetItem(v))

        # -------------------- Reports/exports --------------------
        def open_workdir(self):
            p = Path(self.workdir.text()); p.mkdir(parents=True, exist_ok=True); QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

        def choose_workdir(self):
            p = QFileDialog.getExistingDirectory(self, 'Select working directory', self.workdir.text())
            if p:
                self.workdir.setText(p); self._refresh_project_targets(); self._update_state_label()

        def open_last_run(self):
            if not self.last_run_folder:
                QMessageBox.information(self, 'Open run folder', 'No completed run folder yet.'); return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_run_folder))))

        def generate_fea_exports(self):
            if not (self.last_result and self.last_model and self.last_run_folder):
                QMessageBox.information(self, 'FEA export', 'Complete an optimization first.'); return
            try:
                export_fea_bundle(self.last_model, self.last_result.parameters, Path(self.last_run_folder)/'exports', material_name='HYPERFIT_MATERIAL', stress_unit='MPa', unit_system='MPa-mm-N')
                self._set_status('FEA export bundle generated.', 'ok')
            except Exception as exc:
                QMessageBox.critical(self, 'FEA export failed', str(exc))

        def generate_pdf_report_from_gui(self):
            if not (self.last_result and self.last_model and self.last_run_folder):
                QMessageBox.information(self, 'PDF report', 'Complete an optimization first.'); return
            try:
                plot_files = [str(p) for p in (Path(self.last_run_folder)/'plots').glob('*.png')]
                generate_pdf_report(Path(self.last_run_folder), self.last_model, self.last_datasets, self.last_result, plot_files=plot_files, sections=DEFAULT_REPORT_SECTIONS)
                self._set_status('PDF report generated.', 'ok')
            except Exception as exc:
                QMessageBox.critical(self, 'PDF report failed', str(exc))

        def generate_excel_calculator_from_gui(self):
            if not (self.last_result and self.last_model and self.last_run_folder):
                QMessageBox.information(self, 'Excel calculator', 'Complete an optimization first.'); return
            try:
                out = Path(self.last_run_folder)/'exports'/'model_calculator.xlsx'
                export_excel_calculator(self.last_model, self.last_result.parameters, self.last_datasets, self.last_result, out)
                self._set_status(f'Excel calculator generated: {out.name}', 'ok')
            except Exception as exc:
                QMessageBox.critical(self, 'Excel calculator failed', str(exc))

        # -------------------- Project management --------------------
        def _optimizer_settings(self):
            return dict(method=self.method.currentText(), max_evaluations=self.maxeval.value(), regularization=self.reg_strength.value(), regularization_type=self.reg_type.currentText(), validation_fraction=self.validation_fraction.value(), workers=self.worker_count.value())

        def save_project_dialog(self):
            model = self.current_model()
            if model is None:
                QMessageBox.warning(self, 'Save project', 'Select a model before saving a project.'); return
            default = Path(self.workdir.text()) / f'{model.number:03d}_{model.name.replace(" ", "_")}.hyp2fit'
            path, _ = QFileDialog.getSaveFileName(self, 'Save HyperFitPro project', str(default), 'HyperFitPro project (*.hyp2fit);;Legacy HyperFitPro project (*.hfp);;All files (*.*)')
            if not path: return
            project = project_from_state(model, self.datasets, self.workdir.text(), optimizer_settings=self._optimizer_settings(), parameter_bounds={}, report_sections=DEFAULT_REPORT_SECTIONS, last_run_folder=self.last_run_folder, last_fit_result=self.last_result)
            out = save_project(project, path)
            self._set_status(f'Project saved: {out.name}', 'ok'); self.import_project_path(out, silent=True)

        def import_project_dialog(self):
            paths, _ = QFileDialog.getOpenFileNames(self, 'Import project(s)', '', 'HyperFitPro projects (*.hyp2fit *.hfp);;All files (*.*)')
            for p in paths:
                self.import_project_path(p, silent=False)

        def import_project_path(self, path, silent=False):
            try:
                p = Path(path)
                if any(str(x.get('path')) == str(p) for x in self.imported_projects):
                    return
                project = load_project(p)
                self.imported_projects.append({'path': str(p), 'project': project})
                self._refresh_project_table(); self._refresh_project_targets()
                if not silent: self._set_status(f'Project imported: {p.name}', 'ok')
            except Exception as exc:
                QMessageBox.warning(self, 'Project import failed', f'{path}\n\n{exc}')

        def _refresh_project_table(self):
            if not hasattr(self, 'project_table'): return
            self.project_table.setRowCount(len(self.imported_projects))
            for r, rec in enumerate(self.imported_projects):
                p = Path(rec['path']); pr = rec['project']
                vals = [p.name, pr.project_name, f'{pr.model_number} {pr.model_name}', str(len(pr.datasets)), str(pr.last_run_folder or '-'), pr.modified_at, pr.notes[:80] if pr.notes else '']
                for c, v in enumerate(vals): self.project_table.setItem(r, c, QTableWidgetItem(str(v)))

        def _refresh_project_targets(self):
            if not hasattr(self, 'plugin_target_combo'): return
            self.plugin_target_combo.clear(); self.plugin_target_combo.addItem('Current workspace', {'type':'current'})
            for i, rec in enumerate(self.imported_projects):
                pr = rec['project']; self.plugin_target_combo.addItem(f'{Path(rec["path"]).name} | {pr.model_number} {pr.model_name}', {'type':'project','index':i})

        def on_project_selected(self):
            rows = sorted({i.row() for i in self.project_table.selectedItems()})
            if not rows: return
            rec = self.imported_projects[rows[0]]; pr = rec['project']
            self.project_detail.setPlainText(json.dumps(pr.to_dict(), indent=2, default=str))

        def remove_selected_project(self):
            rows = sorted({i.row() for i in self.project_table.selectedItems()}, reverse=True)
            for r in rows:
                if 0 <= r < len(self.imported_projects): self.imported_projects.pop(r)
            self._refresh_project_table(); self._refresh_project_targets()

        def load_selected_project_into_workspace(self):
            rows = sorted({i.row() for i in self.project_table.selectedItems()}) if hasattr(self, 'project_table') else []
            if not rows:
                if self.imported_projects:
                    rows = [0]
                else:
                    QMessageBox.information(self, 'Load project', 'Import/select a project first.'); return
            rec = self.imported_projects[rows[0]]; pr = rec['project']
            if pr.working_directory: self.workdir.setText(pr.working_directory)
            if pr.model_number is not None:
                prefix = f'{pr.model_number:03d} |'
                for i in range(self.model_combo.count()):
                    if self.model_combo.itemText(i).startswith(prefix): self.model_combo.setCurrentIndex(i); break
            self.datasets = []
            failures = []
            for ref in pr.datasets:
                try:
                    self.datasets.append(load_test_data(ref.source_file, ref.mode, ref.weight, data_kind=(ref.preprocessing or {}).get('data_kind','auto')))
                except Exception as exc:
                    failures.append(f'{Path(ref.source_file).name}: {exc}')
            if pr.last_run_folder: self.last_run_folder = pr.last_run_folder
            if pr.last_fit_summary:
                self.last_result = SimpleNamespace(**pr.last_fit_summary)
            self._refresh_data_table(); self.plot_imported_data(); self.on_model_changed(); self._update_state_label(); self._refresh_project_targets()
            msg = f'Loaded project into workspace: {Path(rec["path"]).name}'
            if failures: msg += '\nDatasets not reloaded:\n' + '\n'.join(failures[:8])
            QMessageBox.information(self, 'Load project', msg)

        def show_run_history(self):
            try:
                rows = RunDatabase().list_runs(limit=50)
                if not rows:
                    QMessageBox.information(self, 'Run history', 'No indexed runs yet.'); return
                text = '\n'.join(f"#{r['id']:04d} | model={r.get('model_number')} {r.get('model_name')} | method={r.get('method')} | RMSE={r.get('rmse')} | {r.get('run_folder')}" for r in rows)
                dlg = QMessageBox(self); dlg.setWindowTitle('Run history'); dlg.setText('Recent indexed runs'); dlg.setDetailedText(text); dlg.exec()
            except Exception as exc:
                QMessageBox.critical(self, 'Run history failed', str(exc))

        # -------------------- Plugin workbench --------------------
        def reload_plugins(self):
            self.tool_plugins = discover_tool_plugins(); self._load_plugins_tree(); self._set_status(f'{len(self.tool_plugins)} tool plugins loaded.', 'ok')

        def _load_plugins_tree(self):
            if not hasattr(self, 'plugin_tree'): return
            self.plugin_tree.clear(); self.plugin_matrix.setRowCount(0)
            cats = {}
            for p in self.tool_plugins:
                cats.setdefault(getattr(p, 'category', 'General') or 'General', []).append(p)
            for cat in sorted(cats):
                root = QTreeWidgetItem([f'{cat} ({len(cats[cat])})']); root.setData(0, Qt.UserRole, {'type':'category','category':cat}); self.plugin_tree.addTopLevelItem(root)
                root.setExpanded(True)
                for tool in sorted(cats[cat], key=lambda x: getattr(x, 'name', '')):
                    item = QTreeWidgetItem([f'{getattr(tool, "name", "")}  —  {getattr(tool, "plugin_id", "")}'])
                    item.setData(0, Qt.UserRole, {'type':'plugin','id':getattr(tool, 'plugin_id', '')}); root.addChild(item)
                    r = self.plugin_matrix.rowCount(); self.plugin_matrix.insertRow(r)
                    req = []
                    if getattr(tool, 'requires_model', False): req.append('model')
                    if getattr(tool, 'requires_data', False): req.append('data')
                    if getattr(tool, 'requires_fit', False): req.append('fit')
                    vals = [cat, getattr(tool,'plugin_id',''), getattr(tool,'name',''), ', '.join(req) or '-', ', '.join(getattr(tool,'output_types',[]) or [])]
                    for c, v in enumerate(vals): self.plugin_matrix.setItem(r, c, QTableWidgetItem(str(v)))
            self.plugin_tree.expandAll()

        def on_plugin_selection_changed(self):
            ids = self.selected_plugin_ids()
            if not ids:
                self.plugin_detail.setPlainText('Select a category or plugin. A category selection can run compatible tools in that category.'); return
            lines = []
            by_id = {getattr(p, 'plugin_id', ''): p for p in self.tool_plugins}
            for pid in ids:
                p = by_id.get(pid)
                if p:
                    lines.append(f'{getattr(p,"name","")}\nID: {pid}\nCategory: {getattr(p,"category","")}\nDescription: {getattr(p,"description","")}\nRequires: model={getattr(p,"requires_model",False)}, data={getattr(p,"requires_data",False)}, fit={getattr(p,"requires_fit",False)}\nOutputs: {", ".join(getattr(p,"output_types",[]) or [])}\n')
            self.plugin_detail.setPlainText('\n'.join(lines))

        def selected_plugin_ids(self):
            ids = set()
            for item in self.plugin_tree.selectedItems():
                data = item.data(0, Qt.UserRole) or {}
                if data.get('type') == 'plugin':
                    ids.add(data.get('id'))
                elif data.get('type') == 'category':
                    cat = data.get('category')
                    for p in self.tool_plugins:
                        if getattr(p, 'category', '') == cat: ids.add(getattr(p, 'plugin_id', ''))
            return {i for i in ids if i}

        def context_for_plugin_target(self):
            data = self.plugin_target_combo.currentData() if hasattr(self, 'plugin_target_combo') else {'type':'current'}
            if not data or data.get('type') == 'current':
                return ToolContext(model=self.current_model(), datasets=list(self.datasets), fit_result=self.last_result, run_folder=self.last_run_folder, workdir=self.workdir.text(), parameters=getattr(self.last_result, 'parameters', {}) if self.last_result else {}, extra={'target_name':'Current workspace'}), 'Current workspace'
            rec = self.imported_projects[int(data.get('index'))]
            pr = rec['project']
            model = None
            for m in self.models:
                if m.number == pr.model_number: model = m; break
            datasets = []
            for ref in pr.datasets:
                try:
                    datasets.append(load_test_data(ref.source_file, ref.mode, ref.weight, data_kind=(ref.preprocessing or {}).get('data_kind','auto')))
                except Exception:
                    pass
            fit_summary = pr.last_fit_summary or {}
            fit_result = SimpleNamespace(**fit_summary) if fit_summary else None
            params = fit_summary.get('parameters', {}) if isinstance(fit_summary, dict) else {}
            target_name = Path(rec['path']).name
            return ToolContext(model=model, datasets=datasets, fit_result=fit_result, run_folder=pr.last_run_folder, workdir=pr.working_directory or self.workdir.text(), parameters=params, extra={'target_name':target_name, 'project_path':rec['path']}), target_name

        def run_plugins(self, selected_only=True):
            ids = self.selected_plugin_ids() if selected_only else None
            if selected_only and not ids:
                QMessageBox.information(self, 'Run plugins', 'Select one or more plugins or a category.'); return
            ctx, target_name = self.context_for_plugin_target()
            results = run_tool_plugins(ctx, selected_ids=ids)
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for r in results:
                self.plugin_results.append({'time': now, 'target': target_name, 'plugin_id': r.plugin_id, 'name': r.name, 'ok': r.ok, 'message': r.message, 'files': list(r.files or [])})
            self.refresh_plugin_results_table()
            self.stack.setCurrentIndex(6)
            ok_count = sum(1 for r in results if r.ok)
            self._set_status(f'Plugin run complete: {ok_count}/{len(results)} OK. Results tab updated.', 'ok' if ok_count else 'warn')

        def refresh_plugin_results_table(self):
            self.plugin_results_table.setRowCount(len(self.plugin_results))
            for r, res in enumerate(self.plugin_results):
                vals = [res['time'], res['target'], f"{res['name']}\n{res['plugin_id']}", 'OK' if res['ok'] else 'FAILED', res['message'][:160], str(len(res['files']))]
                for c, v in enumerate(vals): self.plugin_results_table.setItem(r, c, QTableWidgetItem(str(v)))

        def on_plugin_result_selected(self):
            rows = sorted({i.row() for i in self.plugin_results_table.selectedItems()})
            self.plugin_file_list.clear(); self.text_preview.clear(); self.image_label.clear(); self.preview_stack.setCurrentIndex(0)
            if not rows: return
            res = self.plugin_results[rows[0]]
            files = res.get('files') or []
            if not files:
                self.text_preview.setPlainText(res.get('message','No files produced.'))
                return
            for f in files:
                it = QListWidgetItem(str(f)); it.setData(Qt.UserRole, str(f)); self.plugin_file_list.addItem(it)

        def on_plugin_file_selected(self):
            items = self.plugin_file_list.selectedItems()
            if not items: return
            path = Path(items[0].data(Qt.UserRole))
            self.preview_file(path)

        def preview_file(self, path: Path):
            if not path.exists():
                self.preview_stack.setCurrentIndex(0); self.text_preview.setPlainText(f'File not found:\n{path}'); return
            ext = path.suffix.lower()
            if ext in {'.png','.jpg','.jpeg','.webp','.bmp'}:
                pix = QPixmap(str(path))
                if not pix.isNull():
                    self.image_label.setPixmap(pix.scaled(self.image_preview_scroll.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)); self.preview_stack.setCurrentIndex(1); return
            self.preview_stack.setCurrentIndex(0)
            if ext in {'.txt','.md','.json','.csv','.tex','.html','.htm','.log','.py','.inp','.mac','.k'}:
                txt = path.read_text(encoding='utf-8', errors='replace')
                self.text_preview.setPlainText(txt[:250000])
            else:
                self.text_preview.setPlainText(f'Preview not available for {ext}. Use Open selected file.\n\n{path}')

        def open_selected_plugin_file(self):
            items = self.plugin_file_list.selectedItems()
            if items: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(items[0].data(Qt.UserRole)))))

        def open_selected_plugin_folder(self):
            items = self.plugin_file_list.selectedItems()
            if items: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(items[0].data(Qt.UserRole)).parent)))

        # -------------------- Diagnostics / verification --------------------
        def _selected_diagnostics(self):
            items = self.diagnostic_list.selectedItems() if hasattr(self, 'diagnostic_list') else []
            return [it.data(Qt.UserRole) for it in items]

        def _set_diagnostics_running(self, running: bool):
            if hasattr(self, 'diag_run_selected_btn'):
                self.diag_run_selected_btn.setEnabled(not running)
                self.diag_run_all_btn.setEnabled(not running)
                self.diag_clear_btn.setEnabled(not running)

        def run_selected_diagnostics(self):
            selected = self._selected_diagnostics()
            if not selected:
                QMessageBox.information(self, 'No diagnostic selected', 'Soldaki listeden bir veya birkaç doğrulama seç.')
                return
            self._start_diagnostics(selected)

        def run_all_diagnostics(self):
            self._start_diagnostics(DIAGNOSTICS)

        def _start_diagnostics(self, diagnostics):
            if hasattr(self, 'diagnostic_worker') and self.diagnostic_worker and self.diagnostic_worker.isRunning():
                QMessageBox.information(self, 'Diagnostics running', 'Bir doğrulama seti zaten çalışıyor.')
                return
            self.stack.setCurrentIndex(8)
            self.diagnostic_output.append(f'[{datetime.now().strftime("%H:%M:%S")}] Starting {len(diagnostics)} diagnostic check(s)...')
            project_root = Path(__file__).resolve().parents[1]
            self.diagnostic_worker = DiagnosticWorker(diagnostics, project_root)
            self.diagnostic_worker.log_signal.connect(self.diagnostic_output.append)
            self.diagnostic_worker.done_signal.connect(self._diagnostics_done)
            self._set_diagnostics_running(True)
            self.diagnostic_worker.start()

        def _diagnostics_done(self, failures):
            self._set_diagnostics_running(False)
            if failures:
                self._set_status(f'Diagnostics finished with {failures} failure(s).', 'block')
                self.diagnostic_output.append(f'\nFINISHED WITH {failures} FAILURE(S).')
            else:
                self._set_status('Diagnostics passed.', 'ok')
                self.diagnostic_output.append('\nALL SELECTED DIAGNOSTICS PASSED.')

        # -------------------- Misc --------------------
        def show_about(self):
            QMessageBox.information(self, tr(self.language, 'about_title'), tr(self.language, 'about_text'))

    existing_app = QApplication.instance()
    owns_app = existing_app is None
    app = existing_app or QApplication(sys.argv)
    win = ModernMainWindow()
    win.show()
    # Keep a module-level reference so the window is not garbage-collected when
    # launched from Spyder/IPython where a QApplication already exists.
    global _HYPERFITPRO_MAIN_WINDOW
    _HYPERFITPRO_MAIN_WINDOW = win
    if owns_app:
        return app.exec()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
