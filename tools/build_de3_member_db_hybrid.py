import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _resolve_path(path_arg: str, *, must_exist: bool) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if must_exist and not path.is_file():
        raise SystemExit(f"File not found: {path_arg}")
    return path


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("strategies") or payload.get("rows") or []
    else:
        rows = []
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list-like payload at {path}")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _is_family_row(row: Dict[str, Any]) -> bool:
    family_tag = str(row.get("FamilyTag", row.get("family_tag", "")) or "").strip()
    if family_tag:
        return True
    strategy_id = str(row.get("strategy_id", "") or "").strip()
    return any(tag in strategy_id for tag in ("BullClimax", "BullTrap", "BearReject", "BearDrive", "BearTrend", "BearImpulse", "BullExhaust", "BullStall"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a DE3 member DB hybrid by appending family-tagged rows to a baseline member DB."
    )
    parser.add_argument("--baseline", required=True, help="Baseline DE3 member DB JSON.")
    parser.add_argument("--overlay", required=True, help="Overlay DE3 member DB JSON.")
    parser.add_argument("--output", required=True, help="Output hybrid member DB JSON.")
    args = parser.parse_args()

    baseline_path = _resolve_path(str(args.baseline), must_exist=True)
    overlay_path = _resolve_path(str(args.overlay), must_exist=True)
    output_path = _resolve_path(str(args.output), must_exist=False)

    baseline_rows = _load_rows(baseline_path)
    overlay_rows = _load_rows(overlay_path)
    family_rows = [dict(row) for row in overlay_rows if _is_family_row(row)]
    if not family_rows:
        raise SystemExit(f"No family-tagged rows found in {overlay_path}")

    seen_strategy_ids = {
        str(row.get("strategy_id", "") or "").strip()
        for row in baseline_rows
        if str(row.get("strategy_id", "") or "").strip()
    }
    merged_rows = list(baseline_rows)
    appended = 0
    for row in family_rows:
        strategy_id = str(row.get("strategy_id", "") or "").strip()
        if strategy_id and strategy_id in seen_strategy_ids:
            continue
        merged_rows.append(dict(row))
        if strategy_id:
            seen_strategy_ids.add(strategy_id)
        appended += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged_rows, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"output={output_path}")
    print(f"baseline_rows={len(baseline_rows)}")
    print(f"overlay_rows={len(overlay_rows)}")
    print(f"family_rows={len(family_rows)}")
    print(f"appended_rows={appended}")
    print(f"total_rows={len(merged_rows)}")


if __name__ == "__main__":
    main()
