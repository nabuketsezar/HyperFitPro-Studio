from pathlib import Path
from tempfile import TemporaryDirectory

from hyperfitpro.core.registry import get_model
from hyperfitpro.core.exporter import export_fea_bundle, abaqus_material_lines, ansys_material_lines


def main():
    checks = [
        (6, {"mu": 2.0, "K": 100.0}),
        (1, {"C10": 0.25, "C01": 0.10, "K": 120.0}),
        (8, {"C10": 0.35, "C20": 0.02, "C30": -0.001, "K": 90.0}),
        (10, {"mu": 1.2, "lambda_L": 4.5, "K": 100.0}),
        (11, {"mu": 1.5, "I_L": 30.0, "K": 100.0}),
        (12, {"mu1": 1.0, "alpha1": 2.0, "mu2": 0.05, "alpha2": -2.0, "K": 100.0}),
        (32, {"mu": 1.0, "lambda_m": 5.0, "a": 0.1, "beta": 0.2, "K": 100.0}),
    ]
    with TemporaryDirectory() as td:
        root = Path(td)
        for no, params in checks:
            model = get_model(no)
            out = root / f"m{no:02d}"
            manifest = export_fea_bundle(model, params, out)
            required = [
                "material_fit.json",
                "abaqus_material.inp",
                "calculix_material.inp",
                "ansys_material.mac",
                "lsdyna_material.k",
                "marc_material.txt",
                "unit_consistency_report.md",
                "single_element_verification_targets.csv",
                "abaqus_single_element_probe_template.inp",
                "ansys_single_element_probe_template.mac",
                "fea_export_manifest.json",
            ]
            missing = [x for x in required if not (out / x).exists()]
            if missing:
                raise RuntimeError(f"model {no} missing files: {missing}")
            if not manifest["files"]:
                raise RuntimeError(f"model {no} empty manifest")
            abaqus = (out / "abaqus_material.inp").read_text(encoding="utf-8")
            ansys = (out / "ansys_material.mac").read_text(encoding="utf-8")
            if "HyperFitPro" not in abaqus or "*MATERIAL" not in abaqus:
                raise RuntimeError(f"model {no} bad Abaqus export")
            if "HyperFitPro" not in ansys or "TB,HYPER" not in ansys and "USERMAT" not in ansys:
                raise RuntimeError(f"model {no} bad ANSYS export")
            targets = (out / "single_element_verification_targets.csv").read_text(encoding="utf-8")
            if "uniaxial" not in targets or "simple_shear" not in targets:
                raise RuntimeError(f"model {no} missing verification target rows")
    print("FEA export verification passed.")


if __name__ == "__main__":
    main()
