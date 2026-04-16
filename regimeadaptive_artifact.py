import json
import logging
import math
from pathlib import Path
from typing import Optional


_WILDCARD = "ALL"


def _normalize_policy(value) -> str:
    policy = str(value or "skip").strip().lower()
    if policy not in {"normal", "reversed", "skip"}:
        return "skip"
    return policy


def _coerce_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _coerce_optional_probability(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    if out < 0.0 or out > 1.0:
        return None
    return float(out)


def _normalize_side(value: str) -> Optional[str]:
    side = str(value or "").strip().upper()
    if side in {"LONG", "SHORT"}:
        return side
    return None


def _normalize_group_policy_priority(value) -> str:
    text = str(value or "fill_only").strip().lower()
    if text not in {"fill_only", "override_skip"}:
        return "fill_only"
    return text


def _split_combo_pattern(value) -> Optional[tuple[str, str, str, str]]:
    parts = [str(part or "").strip().upper() for part in str(value or "").split("_")]
    if len(parts) < 4 or any(not part for part in parts):
        return None
    quarter, week, day = parts[:3]
    session = "_".join(parts[3:])
    if not session:
        return None
    return quarter, week, day, session


def _normalize_combo_pattern(value) -> Optional[str]:
    split = _split_combo_pattern(value)
    if split is None:
        return None
    return "_".join(split)


def _pattern_specificity(pattern: str) -> int:
    split = _split_combo_pattern(pattern)
    if split is None:
        return 0
    return sum(1 for part in split if part != _WILDCARD)


def _pattern_matches(pattern: str, combo_key: str) -> bool:
    combo_parts = _split_combo_pattern(combo_key)
    pattern_parts = _split_combo_pattern(pattern)
    if combo_parts is None or pattern_parts is None:
        return False
    return all(pattern_part == _WILDCARD or pattern_part == combo_part for pattern_part, combo_part in zip(pattern_parts, combo_parts))


def _clean_side_payload(payload) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    try:
        sl = float(payload.get("sl", 0.0) or 0.0)
        tp = float(payload.get("tp", 0.0) or 0.0)
    except Exception:
        return None
    if sl <= 0 or tp <= 0:
        return None
    return {"sl_dist": sl, "tp_dist": tp}


def _clean_rule_payload(payload) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    try:
        rule_type = str(payload.get("rule_type", "pullback") or "pullback").strip().lower()
        sma_fast = int(payload.get("sma_fast", 0) or 0)
        sma_slow = int(payload.get("sma_slow", 0) or 0)
        atr_period = int(payload.get("atr_period", 0) or 0)
        max_hold_bars = int(payload.get("max_hold_bars", 0) or 0)
        cross_atr_mult = float(payload.get("cross_atr_mult", 0.0) or 0.0)
        pattern_lookback = int(payload.get("pattern_lookback", 0) or 0)
        touch_atr_mult = float(payload.get("touch_atr_mult", 0.25) or 0.25)
    except Exception:
        return None
    if rule_type not in {"pullback", "continuation", "breakout"}:
        return None
    if sma_fast <= 0 or sma_slow <= 0 or sma_fast >= sma_slow:
        return None
    if atr_period <= 0 or max_hold_bars <= 0 or cross_atr_mult < 0.0:
        return None
    cleaned = {
        "rule_type": rule_type,
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "atr_period": atr_period,
        "max_hold_bars": max_hold_bars,
        "cross_atr_mult": cross_atr_mult,
    }
    if rule_type in {"continuation", "breakout"}:
        if pattern_lookback <= 0:
            return None
        cleaned["pattern_lookback"] = pattern_lookback
    if rule_type == "continuation":
        cleaned["touch_atr_mult"] = max(0.0, touch_atr_mult)
    return cleaned


class RegimeAdaptiveArtifact:
    def __init__(self, payload: dict, path: Path, default_policy_override: Optional[str] = None):
        self.path = Path(path)
        self.payload = payload if isinstance(payload, dict) else {}
        self.group_policy_priority = _normalize_group_policy_priority(
            self.payload.get("group_policy_priority")
        )
        payload_default_policy = _normalize_policy(self.payload.get("default_unlisted_policy"))
        override_policy = _normalize_policy(default_policy_override)
        self.default_policy = (
            override_policy
            if override_policy in {"normal", "reversed", "skip"}
            else payload_default_policy
        )
        cleaned_base_rule = _clean_rule_payload(self.payload.get("base_rule", {}))
        self.base_rule = cleaned_base_rule or {}
        raw_rule_catalog = self.payload.get("rule_catalog", {}) if isinstance(self.payload.get("rule_catalog", {}), dict) else {}
        self.rule_catalog: dict[str, dict] = {}
        for rule_id, rule_payload in raw_rule_catalog.items():
            cleaned_rule = _clean_rule_payload(rule_payload)
            if cleaned_rule:
                self.rule_catalog[str(rule_id)] = cleaned_rule
        default_rule_id = str(self.payload.get("default_rule_id", "") or "").strip()
        self.default_rule_id = default_rule_id if default_rule_id in self.rule_catalog else None
        raw_combo = self.payload.get("combo_policies", {}) if isinstance(self.payload.get("combo_policies", {}), dict) else {}
        self.combo_policies = {
            str(key): value
            for key, value in raw_combo.items()
            if isinstance(value, dict)
        }
        raw_signal = self.payload.get("signal_policies", {}) if isinstance(self.payload.get("signal_policies", {}), dict) else {}
        self.signal_policies: dict[str, dict[str, dict]] = {}
        for combo_key, side_map in raw_signal.items():
            if not isinstance(side_map, dict):
                continue
            cleaned_side_map: dict[str, dict] = {}
            for side in ("LONG", "SHORT"):
                record = side_map.get(side)
                if not isinstance(record, dict):
                    continue
                cleaned_record = dict(record)
                cleaned_record["policy"] = _normalize_policy(record.get("policy"))
                early_exit_enabled = _coerce_optional_bool(record.get("early_exit_enabled"))
                if early_exit_enabled is None:
                    cleaned_record.pop("early_exit_enabled", None)
                else:
                    cleaned_record["early_exit_enabled"] = bool(early_exit_enabled)
                rule_id = str(record.get("rule_id", "") or "").strip()
                if rule_id and rule_id in self.rule_catalog:
                    cleaned_record["rule_id"] = rule_id
                else:
                    cleaned_record.pop("rule_id", None)
                min_gate_threshold = _coerce_optional_probability(record.get("min_gate_threshold"))
                if min_gate_threshold is None:
                    cleaned_record.pop("min_gate_threshold", None)
                else:
                    cleaned_record["min_gate_threshold"] = float(min_gate_threshold)
                cleaned_side_map[side] = cleaned_record
            if cleaned_side_map:
                self.signal_policies[str(combo_key)] = cleaned_side_map
        raw_group_signal = self.payload.get("group_signal_policies", {}) if isinstance(self.payload.get("group_signal_policies", {}), dict) else {}
        self.group_signal_policies: dict[str, dict[str, dict]] = {}
        for group_key, side_map in raw_group_signal.items():
            pattern = _normalize_combo_pattern(group_key)
            if pattern is None or not isinstance(side_map, dict):
                continue
            cleaned_side_map: dict[str, dict] = {}
            for side in ("LONG", "SHORT"):
                record = side_map.get(side)
                if not isinstance(record, dict):
                    continue
                cleaned_record = dict(record)
                cleaned_record["policy"] = _normalize_policy(record.get("policy"))
                early_exit_enabled = _coerce_optional_bool(record.get("early_exit_enabled"))
                if early_exit_enabled is None:
                    cleaned_record.pop("early_exit_enabled", None)
                else:
                    cleaned_record["early_exit_enabled"] = bool(early_exit_enabled)
                rule_id = str(record.get("rule_id", "") or "").strip()
                if rule_id and rule_id in self.rule_catalog:
                    cleaned_record["rule_id"] = rule_id
                else:
                    cleaned_record.pop("rule_id", None)
                min_gate_threshold = _coerce_optional_probability(record.get("min_gate_threshold"))
                if min_gate_threshold is None:
                    cleaned_record.pop("min_gate_threshold", None)
                else:
                    cleaned_record["min_gate_threshold"] = float(min_gate_threshold)
                cleaned_side_map[side] = cleaned_record
            if cleaned_side_map:
                self.group_signal_policies[pattern] = cleaned_side_map
        self._group_signal_entries = [
            (pattern, _pattern_specificity(pattern), side_map)
            for pattern, side_map in self.group_signal_policies.items()
        ]
        self._group_signal_entries.sort(key=lambda item: (-int(item[1]), str(item[0])))
        raw_session = self.payload.get("session_defaults", {}) if isinstance(self.payload.get("session_defaults", {}), dict) else {}
        self.session_defaults: dict[str, dict[str, dict]] = {}
        for session_name, side_map in raw_session.items():
            if not isinstance(side_map, dict):
                continue
            cleaned = {}
            for side in ("LONG", "SHORT"):
                side_payload = _clean_side_payload(side_map.get(side))
                if side_payload:
                    cleaned[side] = side_payload
            if cleaned:
                self.session_defaults[str(session_name).upper()] = cleaned
        global_defaults = self.payload.get("global_default", {}) if isinstance(self.payload.get("global_default", {}), dict) else {}
        self.global_default = {}
        for side in ("LONG", "SHORT"):
            side_payload = _clean_side_payload(global_defaults.get(side))
            if side_payload:
                self.global_default[side] = side_payload
        raw_signal_gate = self.payload.get("signal_gate", {}) if isinstance(self.payload.get("signal_gate", {}), dict) else {}
        gate_model_path = str(raw_signal_gate.get("model_path", "") or "").strip()
        gate_threshold = None
        try:
            gate_threshold = float(raw_signal_gate.get("threshold")) if raw_signal_gate.get("threshold") is not None else None
        except Exception:
            gate_threshold = None
        feature_columns = raw_signal_gate.get("feature_columns", [])
        raw_session_thresholds = raw_signal_gate.get("session_thresholds", {})
        session_thresholds = {}
        if isinstance(raw_session_thresholds, dict):
            for session_name, raw_threshold in raw_session_thresholds.items():
                try:
                    threshold = float(raw_threshold)
                except Exception:
                    continue
                if math.isfinite(threshold):
                    session_thresholds[str(session_name).upper()] = float(threshold)
        raw_policy_thresholds = raw_signal_gate.get("policy_thresholds", {})
        policy_thresholds = {}
        if isinstance(raw_policy_thresholds, dict):
            for policy_key, raw_threshold in raw_policy_thresholds.items():
                try:
                    threshold = float(raw_threshold)
                except Exception:
                    continue
                if math.isfinite(threshold):
                    policy_thresholds[str(policy_key).upper()] = float(threshold)
        self.signal_gate = {
            "enabled": bool(raw_signal_gate.get("enabled", False)) and bool(gate_model_path),
            "model_path": gate_model_path,
            "threshold": gate_threshold,
            "feature_columns": [str(col) for col in feature_columns] if isinstance(feature_columns, list) else [],
            "session_thresholds": session_thresholds,
            "policy_thresholds": policy_thresholds,
        }
        combo_keys = set(self.combo_policies.keys()) | set(self.signal_policies.keys()) | set(self.group_signal_policies.keys())
        self.combo_count = len(combo_keys)
        self.reverted_count = 0
        for value in self.combo_policies.values():
            if _normalize_policy(value.get("policy")) == "reversed":
                self.reverted_count += 1
        for side_map in self.signal_policies.values():
            for side_record in side_map.values():
                if _normalize_policy(side_record.get("policy")) == "reversed":
                    self.reverted_count += 1
        for side_map in self.group_signal_policies.values():
            for side_record in side_map.values():
                if _normalize_policy(side_record.get("policy")) == "reversed":
                    self.reverted_count += 1

    def _group_signal_policy_record(self, combo_key: str, original_side: Optional[str]) -> dict:
        side_key = _normalize_side(original_side or "")
        if side_key is None:
            return {}
        combo_text = _normalize_combo_pattern(combo_key)
        if combo_text is None:
            return {}
        for pattern, _, side_map in self._group_signal_entries:
            if not _pattern_matches(pattern, combo_text):
                continue
            record = side_map.get(side_key)
            if isinstance(record, dict):
                return record
        return {}

    def signal_policy_record(self, combo_key: str, original_side: Optional[str]) -> dict:
        combo_text = str(combo_key)
        side_key = _normalize_side(original_side or "")
        if side_key is not None:
            side_map = self.signal_policies.get(combo_text, {})
            if isinstance(side_map, dict):
                record = side_map.get(side_key)
                if isinstance(record, dict):
                    if self.group_policy_priority == "override_skip":
                        exact_policy = _normalize_policy(record.get("policy"))
                        if exact_policy == "skip":
                            group_record = self._group_signal_policy_record(combo_text, side_key)
                            if isinstance(group_record, dict) and group_record:
                                group_policy = _normalize_policy(group_record.get("policy"))
                                if group_policy in {"normal", "reversed"}:
                                    return group_record
                    return record
            group_record = self._group_signal_policy_record(combo_text, side_key)
            if isinstance(group_record, dict) and group_record:
                return group_record
        legacy = self.combo_policies.get(combo_text, {})
        return legacy if isinstance(legacy, dict) else {}

    def combo_policy(self, combo_key: str, original_side: Optional[str] = None) -> str:
        if original_side is not None:
            record = self.signal_policy_record(combo_key, original_side)
            if isinstance(record, dict) and record:
                return _normalize_policy(record.get("policy"))
            return str(self.default_policy)
        resolved_policies = {
            _normalize_policy(record.get("policy"))
            for side_key in ("LONG", "SHORT")
            for record in [self.signal_policy_record(str(combo_key), side_key)]
            if isinstance(record, dict) and record
        }
        if len(resolved_policies) == 1:
            return next(iter(resolved_policies))
        if resolved_policies and resolved_policies != {"skip"}:
            return "skip"
        group_side_records = []
        for side_key in ("LONG", "SHORT"):
            record = self._group_signal_policy_record(str(combo_key), side_key)
            if isinstance(record, dict) and record:
                group_side_records.append(_normalize_policy(record.get("policy")))
        group_policies = {policy for policy in group_side_records if policy in {"normal", "reversed", "skip"}}
        if len(group_policies) == 1:
            return next(iter(group_policies))
        if group_policies and group_policies != {"skip"}:
            return "skip"
        record = self.combo_policies.get(str(combo_key), {})
        if isinstance(record, dict) and record:
            return _normalize_policy(record.get("policy"))
        return str(self.default_policy)

    def should_skip(self, combo_key: str, original_side: Optional[str] = None) -> bool:
        return self.combo_policy(combo_key, original_side=original_side) == "skip"

    def should_revert(self, combo_key: str, original_side: Optional[str] = None) -> bool:
        return self.combo_policy(combo_key, original_side=original_side) == "reversed"

    def get_early_exit_enabled(self, combo_key: str, original_side: Optional[str] = None):
        if original_side is not None:
            record = self.signal_policy_record(combo_key, original_side)
            return _coerce_optional_bool(record.get("early_exit_enabled")) if isinstance(record, dict) else None
        side_map = self.signal_policies.get(str(combo_key), {})
        if isinstance(side_map, dict) and side_map:
            values = {
                _coerce_optional_bool(record.get("early_exit_enabled"))
                for record in side_map.values()
                if isinstance(record, dict) and _coerce_optional_bool(record.get("early_exit_enabled")) is not None
            }
            if len(values) == 1:
                return next(iter(values))
        record = self.combo_policies.get(str(combo_key), {})
        return _coerce_optional_bool(record.get("early_exit_enabled")) if isinstance(record, dict) else None

    def get_rule_id(self, combo_key: str, original_side: Optional[str] = None) -> Optional[str]:
        if original_side is not None:
            record = self.signal_policy_record(combo_key, original_side)
            if isinstance(record, dict):
                rule_id = str(record.get("rule_id", "") or "").strip()
                if rule_id in self.rule_catalog:
                    return rule_id
        if self.default_rule_id in self.rule_catalog:
            return self.default_rule_id
        return None

    def get_min_gate_threshold(self, combo_key: str, original_side: Optional[str] = None) -> Optional[float]:
        if original_side is not None:
            record = self.signal_policy_record(combo_key, original_side)
            if isinstance(record, dict):
                return _coerce_optional_probability(record.get("min_gate_threshold"))
        return None

    def get_rule(self, combo_key: str, original_side: Optional[str] = None) -> dict:
        rule_id = self.get_rule_id(combo_key, original_side)
        if rule_id and rule_id in self.rule_catalog:
            return dict(self.rule_catalog[rule_id])
        return dict(self.base_rule)

    def get_combo_sltp(self, side: str, combo_key: str) -> Optional[dict]:
        record = self.combo_policies.get(str(combo_key), {})
        if not isinstance(record, dict):
            return None
        return _clean_side_payload(record.get(str(side).upper()))

    def get_sltp(self, side: str, combo_key: str, session_name: str) -> dict:
        side_key = str(side).upper()
        combo_payload = self.get_combo_sltp(side_key, combo_key)
        if combo_payload:
            return combo_payload
        session_payload = self.session_defaults.get(str(session_name).upper(), {}).get(side_key)
        if session_payload:
            return session_payload
        global_payload = self.global_default.get(side_key)
        if global_payload:
            return global_payload
        return {"sl_dist": 2.0, "tp_dist": 3.0}


def load_regimeadaptive_artifact(
    path_text: str,
    *,
    default_policy_override: Optional[str] = None,
) -> Optional[RegimeAdaptiveArtifact]:
    path_raw = str(path_text or "").strip()
    if not path_raw:
        return None
    path = Path(path_raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("Failed to load RegimeAdaptive artifact %s: %s", path, exc)
        return None
    artifact = RegimeAdaptiveArtifact(payload, path, default_policy_override=default_policy_override)
    logging.info(
        "Loaded RegimeAdaptive artifact %s | combos=%s reverted=%s default_policy=%s group_policy_priority=%s",
        path,
        artifact.combo_count,
        artifact.reverted_count,
        artifact.default_policy,
        artifact.group_policy_priority,
    )
    return artifact
