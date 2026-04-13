#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PACKAGE_NAME = "julie_filterless_public"
DEFAULT_OUTPUT_BASE = ROOT / "temp_git_upload"
SECRET_TEMPLATE = """import os


SECRETS = {
    "USERNAME": os.environ.get("TOPSTEPX_USERNAME", ""),
    "API_KEY": os.environ.get("TOPSTEPX_API_KEY", ""),
    "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
}
"""

PUBLIC_GITIGNORE = """# Python environments
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/

# Local secrets and env files
config_secrets.py
.env
.env.*
trainer_key.pem

# Local runtime state and logs
logs/
*.log
bot_state.json
live_trade_factors.csv
julie_bot_stdout_stderr.log
topstep_live_bot.log

# Frontend local/build outputs
montecarlo/Backtest-Simulator-main/node_modules/
montecarlo/Backtest-Simulator-main/dist/
montecarlo/Backtest-Simulator-main/public/filterless_live_state.json
montecarlo/Backtest-Simulator-main/.env.local
"""

PUBLIC_SETUP = """# Public Filterless Bot Setup

This package contains the current public-safe filterless live runtime.

## Secrets

Set these environment variables before launching:

- `TOPSTEPX_USERNAME`
- `TOPSTEPX_API_KEY`
- `GEMINI_API_KEY` (optional)

You can also edit `config_secrets.py` locally if you prefer file-based secrets.
Do not commit real credentials.

## Launch

### Windows

1. Run `setup_topstep2.ps1`
2. Run `LaunchFilterlessWorkspace.bat`

### Manual

1. Create a Python 3.13 virtual environment
2. Install `requirements.txt`
3. Run `python launch_filterless_workspace.py`

## Notes

- This package intentionally excludes logs, private keys, local env files, and secret-bearing backups.
- The MLPhysics dist bundle is trimmed to inference-only artifacts so the package stays GitHub-friendly.
- Frontend dependencies are not vendored. `launch_filterless_workspace.py` will use `npm install` if needed.
"""

SETUP_SMOKE_MODULES = [
    "config",
    "client",
    "julie001",
    "launch_filterless_workspace",
    "tools.filterless_dashboard_bridge",
]

ABSOLUTE_PATH_RE = re.compile(
    r"(?i)(?:[a-z]:\\users\\[^\\]+\\(?:onedrive\\desktop\\(?:trading\\)?topstep2|desktop\\topstep2))(\\.*)?"
)

LIVE_PATHS = {
    "de3_v3_member_db": "dynamic_engine3_strategies_v2.json",
    "de3_v3_family_db": "dynamic_engine3_v3_bundle.json",
    "de3_v3_family_inventory": "dynamic_engine3_families_v3.json",
    "de3_v4_member_db": "dynamic_engine3_strategies_v2_research_london_shortfix_20260407.json",
    "de3_v4_bundle": "artifacts/de3_v4_live/dynamic_engine3_v4_bundle_research_london_shortfix_20260407.json",
    "volatility_thresholds": "volatility_thresholds.json",
    "ml_physics_thresholds": "ml_physics_thresholds.json",
    "ml_physics_metrics": "ml_physics_metrics.json",
    "ml_physics_regimes": "regimes.json",
    "ml_physics_run_dir": (
        "runpod_results/restored2_20260225_210635/dist_bracket_ml_runs/"
        "ml_physics_cut20241231/dist_bracket_20260224_041706_unknown"
    ),
    "regimeadaptive_artifact": "artifacts/regimeadaptive_v19_live/latest.json",
    "regimeadaptive_gate_model": "artifacts/regimeadaptive_v19_live/regimeadaptive_gate_model.joblib",
    "aetherflow_model": "model_aetherflow_deploy_2026oos.pkl",
    "aetherflow_thresholds": "aetherflow_thresholds_deploy_2026oos.json",
    "aetherflow_metrics": "aetherflow_metrics_deploy_2026oos.json",
}


def rel(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst_root: Path, relative_path: Path | None = None) -> Path:
    rel_path = relative_path or rel(src)
    dst = dst_root / rel_path
    ensure_dir(dst)
    shutil.copy2(src, dst)
    return dst


def copy_text(src: Path, dst_root: Path, transform, relative_path: Path | None = None) -> Path:
    rel_path = relative_path or rel(src)
    dst = dst_root / rel_path
    ensure_dir(dst)
    text = src.read_text(encoding="utf-8")
    dst.write_text(transform(text), encoding="utf-8")
    return dst


def sanitize_path_string(value: str) -> str:
    text = str(value)
    if not text:
        return text
    normalized = text.replace("/", "\\")
    marker = normalized.lower().find("topstep2\\")
    if marker >= 0:
        suffix = normalized[marker + len("topstep2\\") :].replace("\\", "/")
        return suffix
    match = ABSOLUTE_PATH_RE.fullmatch(normalized)
    if match:
        suffix = (match.group(1) or "").lstrip("\\").replace("\\", "/")
        return suffix or Path(normalized).name
    return text


def sanitize_json_value(value):
    if isinstance(value, dict):
        return {k: sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(v) for v in value]
    if isinstance(value, str):
        return sanitize_path_string(value)
    return value


def sanitize_json_file(src: Path, dst_root: Path, relative_path: Path | None = None) -> Path:
    rel_path = relative_path or rel(src)
    dst = dst_root / rel_path
    ensure_dir(dst)
    payload = json.loads(src.read_text(encoding="utf-8"))
    dst.write_text(json.dumps(sanitize_json_value(payload), indent=2, sort_keys=False), encoding="utf-8")
    return dst


def iter_python_imports(path: Path) -> Iterable[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                continue
            imports.append(node.module)
    return imports


def resolve_local_module(module_name: str) -> Path | None:
    parts = module_name.split(".")
    candidate = ROOT.joinpath(*parts)
    py_path = candidate.with_suffix(".py")
    if py_path.is_file():
        return py_path
    init_path = candidate / "__init__.py"
    if init_path.is_file():
        return init_path
    return None


def build_python_closure(seed_files: Iterable[Path]) -> list[Path]:
    queue = [path.resolve() for path in seed_files]
    seen: set[Path] = set()
    out: list[Path] = []
    while queue:
        current = queue.pop()
        if current in seen or not current.is_file():
            continue
        seen.add(current)
        out.append(current)
        for module_name in iter_python_imports(current):
            local = resolve_local_module(module_name)
            if local is not None and local not in seen:
                queue.append(local)
    return sorted(out)


def copy_tree_filtered(src_dir: Path, dst_root: Path, *, exclude_dirs: set[str], exclude_files: set[str]) -> None:
    for path in src_dir.rglob("*"):
        if any(part in exclude_dirs for part in path.parts):
            continue
        if path.is_file():
            if path.name in exclude_files:
                continue
            copy_file(path, dst_root)


def build_mlphysics_inference_subset(src_run_dir: Path, dst_root: Path, relative_run_dir: Path) -> tuple[int, int]:
    artifact_index = json.loads((src_run_dir / "artifact_index.json").read_text(encoding="utf-8"))
    referenced: set[str] = set()

    model_index = artifact_index.get("model_index", {}) or {}
    for per_session in model_index.values():
        if not isinstance(per_session, dict):
            continue
        for payload in per_session.values():
            if not isinstance(payload, dict):
                continue
            for key in ("ev", "hit"):
                raw = payload.get(key)
                if raw:
                    referenced.add(str(raw))
            for bucket_name in ("mfe", "mae", "evq"):
                bucket = payload.get(bucket_name, {}) or {}
                if isinstance(bucket, dict):
                    for raw in bucket.values():
                        if raw:
                            referenced.add(str(raw))

    gate_cfg = artifact_index.get("gate", {}) or {}
    gate_model_index = gate_cfg.get("model_index", {}) or {}
    for per_session in gate_model_index.values():
        if not isinstance(per_session, dict):
            continue
        for payload in per_session.values():
            if not isinstance(payload, dict):
                continue
            for key in ("classifier", "regressor", "calibrator"):
                raw = payload.get(key)
                if raw:
                    referenced.add(str(raw))

    copied_files = 0
    copied_bytes = 0
    for base_name in ("config.json", "artifact_index.json", "eval_report.json", "gate_metrics.json", "gate_report.md"):
        src = src_run_dir / base_name
        if src.is_file():
            if src.suffix.lower() == ".json":
                sanitize_json_file(src, dst_root, relative_run_dir / base_name)
            else:
                copy_file(src, dst_root, relative_run_dir / base_name)
            copied_files += 1
            copied_bytes += src.stat().st_size

    for rel_text in sorted(referenced):
        rel_path = Path(rel_text.replace("\\", "/"))
        src = src_run_dir / rel_path
        if not src.is_file():
            raise FileNotFoundError(f"MLPhysics inference artifact missing: {src}")
        copy_file(src, dst_root, relative_run_dir / rel_path)
        copied_files += 1
        copied_bytes += src.stat().st_size
    return copied_files, copied_bytes


def patch_setup_script(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_smoke = False
    for line in lines:
        if line.strip() == "$smokeCode = @'":
            in_smoke = True
            out.append(line)
            out.append("import importlib")
            out.append("")
            out.append("modules = [")
            for module in SETUP_SMOKE_MODULES:
                out.append(f'    "{module}",')
            out.append("]")
            out.append("")
            out.append("for name in modules:")
            out.append("    importlib.import_module(name)")
            out.append("")
            out.append('print("Import smoke checks passed.")')
            continue
        if in_smoke:
            if line.strip() == "'@":
                in_smoke = False
                out.append(line)
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def write_manifest(dst_root: Path, payload: dict) -> None:
    manifest_path = dst_root / "PUBLIC_RELEASE_MANIFEST.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def create_zip(src_dir: Path, zip_path: Path) -> None:
    ensure_dir(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))


def build_package(output_base: Path, tag: str) -> tuple[Path, Path]:
    package_dir = output_base / f"{PACKAGE_NAME}_{tag}"
    zip_path = output_base / f"{PACKAGE_NAME}_{tag}.zip"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    docs_and_scripts = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "setup_topstep2.ps1",
        ROOT / "LaunchFilterlessWorkspace.bat",
        ROOT / "config_secrets.example.py",
        ROOT / "logo.gif",
    ]
    seed_files = [
        ROOT / "launch_filterless_workspace.py",
        ROOT / "launch_filterless_live.py",
        ROOT / "tools" / "filterless_dashboard_bridge.py",
        ROOT / "tools" / "filterless_static_server.py",
    ]
    python_files = build_python_closure(seed_files)

    copied_python: list[str] = []
    for path in python_files:
        copy_file(path, package_dir)
        copied_python.append(str(rel(path)).replace("\\", "/"))

    for path in docs_and_scripts:
        if path.name == "setup_topstep2.ps1":
            copy_text(path, package_dir, patch_setup_script)
        else:
            copy_file(path, package_dir)

    # Safe local secrets loader for public use.
    (package_dir / "config_secrets.py").write_text(SECRET_TEMPLATE, encoding="utf-8")
    (package_dir / ".gitignore").write_text(PUBLIC_GITIGNORE, encoding="utf-8")
    (package_dir / "PUBLIC_SETUP.md").write_text(PUBLIC_SETUP, encoding="utf-8")

    copy_tree_filtered(
        ROOT / "montecarlo" / "Backtest-Simulator-main",
        package_dir,
        exclude_dirs={"node_modules", "dist", "__pycache__"},
        exclude_files={".env.local", "filterless_live_state.json"},
    )
    copy_tree_filtered(
        ROOT / "dist_bracket_ml",
        package_dir,
        exclude_dirs={"__pycache__", "tests"},
        exclude_files=set(),
    )

    required_json_files = [
        ROOT / LIVE_PATHS["de3_v3_member_db"],
        ROOT / LIVE_PATHS["de3_v3_family_db"],
        ROOT / LIVE_PATHS["de3_v3_family_inventory"],
        ROOT / LIVE_PATHS["de3_v4_member_db"],
        ROOT / LIVE_PATHS["de3_v4_bundle"],
        ROOT / LIVE_PATHS["volatility_thresholds"],
        ROOT / LIVE_PATHS["ml_physics_thresholds"],
        ROOT / LIVE_PATHS["ml_physics_metrics"],
        ROOT / LIVE_PATHS["ml_physics_regimes"],
        ROOT / "de3_context_veto_models.json",
        ROOT / LIVE_PATHS["regimeadaptive_artifact"],
        ROOT / LIVE_PATHS["aetherflow_thresholds"],
        ROOT / LIVE_PATHS["aetherflow_metrics"],
    ]
    required_binary_files = [
        ROOT / LIVE_PATHS["regimeadaptive_gate_model"],
        ROOT / LIVE_PATHS["aetherflow_model"],
        ROOT / "model_asia.joblib",
        ROOT / "model_asia_low.joblib",
        ROOT / "model_asia_high.joblib",
        ROOT / "model_london.joblib",
        ROOT / "model_london_low.joblib",
        ROOT / "model_london_high.joblib",
        ROOT / "model_ny_am.joblib",
        ROOT / "model_ny_am_low.joblib",
        ROOT / "model_ny_am_normal.joblib",
        ROOT / "model_ny_am_high.joblib",
        ROOT / "model_ny_pm.joblib",
        ROOT / "model_ny_pm_low.joblib",
        ROOT / "model_ny_pm_normal.joblib",
        ROOT / "model_ny_pm_high.joblib",
    ]

    for src in required_json_files:
        sanitize_json_file(src, package_dir)
    for src in required_binary_files:
        copy_file(src, package_dir)

    ml_run_rel = Path(LIVE_PATHS["ml_physics_run_dir"].replace("\\", "/"))
    ml_run_src = ROOT / ml_run_rel
    ml_files, ml_bytes = build_mlphysics_inference_subset(ml_run_src, package_dir, ml_run_rel)

    manifest = {
        "package_name": package_dir.name,
        "folder_name": package_dir.name,
        "zip_name": zip_path.name,
        "copied_python_files": copied_python,
        "mlphysics_run_dir": str(ml_run_rel).replace("\\", "/"),
        "mlphysics_inference_files": ml_files,
        "mlphysics_inference_bytes": ml_bytes,
    }
    write_manifest(package_dir, manifest)
    create_zip(package_dir, zip_path)
    return package_dir, zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GitHub-safe public filterless bot package.")
    parser.add_argument(
        "--output-base",
        default=str(DEFAULT_OUTPUT_BASE),
        help="Directory where the public folder and zip should be created.",
    )
    parser.add_argument(
        "--tag",
        default=os.environ.get("PUBLIC_RELEASE_TAG") or dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Suffix tag for the package folder and zip.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_base = Path(args.output_base).expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    package_dir, zip_path = build_package(output_base, str(args.tag))
    print(f"PACKAGE_DIR={package_dir}")
    print(f"ZIP_PATH={zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
