
from __future__ import annotations
import os, sys, threading, traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .core.registry import load_models
from .core.tests import load_csv_test_data, VALID_MODES
from .core.optimizer import fit_model
from .core.run_manager import create_run_folder, save_run, DEFAULT_ROOT
from .core.license_manager import current_license_status

class HyperFitProApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('HyperFitPro - Hyperelastic Model Calibration')
        self.geometry('1180x760')
        self.minsize(1000, 650)
        self.models = load_models(include_inactive=True)
        self.datasets = []
        self.working_dir = tk.StringVar(value=str(DEFAULT_ROOT))
        self.method = tk.StringVar(value='hybrid')
        self.max_evals = tk.IntVar(value=3500)
        self.model_var = tk.StringVar()
        self._build_ui()
        self._populate_models()
        lic = current_license_status()
        self.log(f"{lic['message']} Machine fingerprint: {lic['fingerprint']}")

    def _build_ui(self):
        top = ttk.Frame(self, padding=8); top.pack(side='top', fill='x')
        ttk.Label(top, text='Working directory:').pack(side='left')
        ttk.Entry(top, textvariable=self.working_dir, width=80).pack(side='left', padx=5, fill='x', expand=True)
        ttk.Button(top, text='Set...', command=self.choose_workdir).pack(side='left', padx=3)
        ttk.Button(top, text='Open', command=self.open_workdir).pack(side='left', padx=3)

        main = ttk.PanedWindow(self, orient='horizontal'); main.pack(fill='both', expand=True, padx=8, pady=4)
        left = ttk.Frame(main, padding=6); right = ttk.Frame(main, padding=6)
        main.add(left, weight=1); main.add(right, weight=2)

        lf_model = ttk.LabelFrame(left, text='1) Model selection', padding=8); lf_model.pack(fill='x')
        self.model_combo = ttk.Combobox(lf_model, textvariable=self.model_var, state='readonly', width=65)
        self.model_combo.pack(fill='x')
        self.model_combo.bind('<<ComboboxSelected>>', lambda e: self.show_model_info())
        self.model_info = tk.Text(lf_model, height=12, wrap='word')
        self.model_info.pack(fill='x', pady=5)

        lf_data = ttk.LabelFrame(left, text='2) Test data', padding=8); lf_data.pack(fill='both', expand=True, pady=8)
        addrow = ttk.Frame(lf_data); addrow.pack(fill='x')
        self.mode_var = tk.StringVar(value='uniaxial')
        ttk.Combobox(addrow, textvariable=self.mode_var, values=VALID_MODES, width=16, state='readonly').pack(side='left')
        ttk.Label(addrow, text='Weight').pack(side='left', padx=(8,2))
        self.weight_var = tk.DoubleVar(value=1.0)
        ttk.Entry(addrow, textvariable=self.weight_var, width=8).pack(side='left')
        ttk.Button(addrow, text='Add CSV...', command=self.add_csv).pack(side='left', padx=6)
        ttk.Button(addrow, text='Remove selected', command=self.remove_selected_data).pack(side='left')
        cols = ('mode','n','weight','file')
        self.data_tree = ttk.Treeview(lf_data, columns=cols, show='headings', height=9)
        for c, w in [('mode',90),('n',60),('weight',70),('file',360)]:
            self.data_tree.heading(c, text=c); self.data_tree.column(c, width=w, anchor='w')
        self.data_tree.pack(fill='both', expand=True, pady=6)

        lf_fit = ttk.LabelFrame(left, text='3) Fit control', padding=8); lf_fit.pack(fill='x')
        row = ttk.Frame(lf_fit); row.pack(fill='x')
        ttk.Label(row, text='Method').pack(side='left')
        ttk.Combobox(row, textvariable=self.method, values=['hybrid','least_squares','differential_evolution'], state='readonly', width=22).pack(side='left', padx=5)
        ttk.Label(row, text='Max evals').pack(side='left', padx=(12,2))
        ttk.Entry(row, textvariable=self.max_evals, width=10).pack(side='left')
        ttk.Button(lf_fit, text='RUN PARAMETER IDENTIFICATION', command=self.run_fit).pack(fill='x', pady=8)

        lf_log = ttk.LabelFrame(right, text='Run log', padding=8); lf_log.pack(fill='both', expand=True)
        self.log_text = tk.Text(lf_log, wrap='word')
        self.log_text.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(lf_log, command=self.log_text.yview); sb.pack(side='right', fill='y')
        self.log_text.configure(yscrollcommand=sb.set)

    def _populate_models(self):
        vals=[]
        self.model_map = {}
        for m in self.models:
            label = f"{m.number:02d} | {m.category} | {m.name}" + (" [disabled]" if not m.active else "")
            vals.append(label); self.model_map[label]=m
        self.model_combo['values']=vals
        if vals:
            self.model_var.set(vals[0]); self.show_model_info()

    def current_model(self):
        return self.model_map.get(self.model_var.get())

    def show_model_info(self):
        m = self.current_model()
        self.model_info.delete('1.0','end')
        if not m: return
        lines = []
        lines.append(f"No: {m.number}\nName: {m.name}\nCategory: {m.category}\nActive: {m.active}\n")
        lines.append("Recommended tests: " + ', '.join(m.recommended_tests) + "\n")
        lines.append("Parameters:\n")
        for s in m.parameter_specs:
            lines.append(f"  - {s.name}: [{s.lower}, {s.upper}] scale={s.scale} {s.description}\n")
        lines.append("\nEquation:\n" + m.equation_latex + "\n")
        if m.notes: lines.append("\nNotes:\n" + m.notes + "\n")
        self.model_info.insert('1.0',''.join(lines))

    def choose_workdir(self):
        p = filedialog.askdirectory(title='Select HyperFitPro working directory')
        if p: self.working_dir.set(p)

    def open_workdir(self):
        p = Path(self.working_dir.get())
        p.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith('win'):
                os.startfile(str(p))
            elif sys.platform == 'darwin':
                os.system(f'open "{p}"')
            else:
                os.system(f'xdg-open "{p}"')
        except Exception as e:
            messagebox.showerror('Open failed', str(e))

    def add_csv(self):
        p = filedialog.askopenfilename(title='Select CSV test data', filetypes=[('CSV/text','*.csv *.txt *.dat'),('All files','*.*')])
        if not p: return
        try:
            d = load_csv_test_data(p, self.mode_var.get(), self.weight_var.get())
            self.datasets.append(d)
            self.data_tree.insert('', 'end', values=(d.mode, len(d.x), d.weight, d.source_file))
            self.log(f"Loaded {len(d.x)} points: {d.mode} | {d.source_file}")
        except Exception as e:
            messagebox.showerror('Data load failed', str(e))

    def remove_selected_data(self):
        sel = list(self.data_tree.selection())
        if not sel: return
        for item in reversed(sel):
            idx = self.data_tree.index(item)
            self.data_tree.delete(item)
            if 0 <= idx < len(self.datasets):
                self.datasets.pop(idx)

    def log(self, msg):
        self.log_text.insert('end', str(msg) + '\n')
        self.log_text.see('end')
        self.update_idletasks()

    def run_fit(self):
        m = self.current_model()
        if not m:
            messagebox.showwarning('No model', 'Select a model first.'); return
        if not m.active:
            messagebox.showwarning('Disabled model', 'This model is disabled because the source equation was not visible.'); return
        if not self.datasets:
            messagebox.showwarning('No data', 'Load at least one CSV test data file.'); return
        t = threading.Thread(target=self._run_fit_thread, daemon=True)
        t.start()

    def _run_fit_thread(self):
        try:
            m = self.current_model()
            self.log('='*80)
            self.log(f"Fitting model {m.number} - {m.name}")
            self.log(f"Method={self.method.get()} max_evals={self.max_evals.get()}")
            run_folder = create_run_folder(self.working_dir.get(), m)
            self.log(f"Run folder: {run_folder}")
            result = fit_model(m, self.datasets, method=self.method.get(), max_evals=int(self.max_evals.get()), progress=self.log)
            self.log(f"Fit completed: success={result.success} obj={result.objective:.6g} RMSE={result.rmse:.6g} R2={result.r2:.6g}")
            for k, v in result.parameters.items():
                self.log(f"  {k} = {v:.12g}")
            save_run(m, self.datasets, result, run_folder)
            self.log(f"Saved all outputs to: {run_folder}")
            messagebox.showinfo('HyperFitPro', f'Run completed.\nSaved to:\n{run_folder}')
        except Exception:
            err = traceback.format_exc()
            self.log(err)
            messagebox.showerror('Run failed', err)


def main():
    app = HyperFitProApp()
    app.mainloop()

if __name__ == '__main__':
    main()
