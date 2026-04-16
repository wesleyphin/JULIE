from typing import Any, Dict, Optional, Tuple

from de3_v4_schema import build_family_id, safe_float


class DE3V4BracketModule:
    def __init__(
        self,
        *,
        bracket_defaults: Optional[Dict[str, Dict[str, Any]]],
        bracket_modes: Optional[Dict[str, Dict[str, Dict[str, float]]]],
        family_bracket_selector: Optional[Dict[str, Any]] = None,
        runtime_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.bracket_defaults = (
            dict(bracket_defaults) if isinstance(bracket_defaults, dict) else {}
        )
        self.bracket_modes = dict(bracket_modes) if isinstance(bracket_modes, dict) else {}
        self.family_bracket_selector = (
            dict(family_bracket_selector)
            if isinstance(family_bracket_selector, dict)
            else {}
        )
        self.runtime_cfg = dict(runtime_cfg) if isinstance(runtime_cfg, dict) else {}
        self.enable_adaptive_modes = bool(
            self.runtime_cfg.get("enable_adaptive_modes", False)
        )
        self.enable_family_bracket_selector = bool(
            self.runtime_cfg.get("enable_family_bracket_selector", True)
        )
        self.min_support_for_adaptive = max(
            0,
            int(
                safe_float(
                    self.runtime_cfg.get("min_support_for_adaptive_modes", 80),
                    80,
                )
            ),
        )
        self.min_support_for_conservative = max(
            0,
            int(
                safe_float(
                    self.runtime_cfg.get(
                        "min_support_for_conservative_modes",
                        self.min_support_for_adaptive,
                    ),
                    self.min_support_for_adaptive,
                )
            ),
        )
        self.min_support_for_aggressive = max(
            0,
            int(
                safe_float(
                    self.runtime_cfg.get(
                        "min_support_for_aggressive_modes",
                        self.min_support_for_adaptive,
                    ),
                    self.min_support_for_adaptive,
                )
            ),
        )
        self.aggressive_requires_expanding = bool(
            self.runtime_cfg.get("aggressive_requires_expanding", True)
        )
        self.allow_aggressive_in_high_vol = bool(
            self.runtime_cfg.get("allow_aggressive_in_high_vol", False)
        )
        low_bands = self.runtime_cfg.get(
            "adaptive_low_confidence_bands",
            ["low", "very_low"],
        )
        high_bands = self.runtime_cfg.get(
            "adaptive_high_confidence_bands",
            ["high", "very_high"],
        )
        self.low_confidence_bands = {
            str(v).strip().lower()
            for v in (
                low_bands if isinstance(low_bands, (list, tuple, set)) else []
            )
        }
        self.high_confidence_bands = {
            str(v).strip().lower()
            for v in (
                high_bands if isinstance(high_bands, (list, tuple, set)) else []
            )
        }

    @staticmethod
    def _norm_pair(sl: Any, tp: Any) -> Optional[Tuple[float, float]]:
        sl_val = round(safe_float(sl, 0.0), 4)
        tp_val = round(safe_float(tp, 0.0), 4)
        if sl_val <= 0.0 or tp_val <= 0.0:
            return None
        return (sl_val, tp_val)

    @staticmethod
    def _scope_key(scope: str, context: Dict[str, Any]) -> str:
        vol = str(context.get("volatility_regime", "normal") or "normal").strip().lower()
        comp = str(
            context.get("compression_expansion_regime", "neutral") or "neutral"
        ).strip().lower()
        conf = str(context.get("confidence_band", "mid") or "mid").strip().lower()
        substate = str(context.get("session_substate", "") or "").strip().lower()
        if not substate:
            substate = "unknown"
        if scope == "exact":
            return f"{vol}|{comp}|{conf}|{substate}"
        if scope == "regime_conf":
            return f"{vol}|{comp}|{conf}"
        if scope == "regime_substate":
            return f"{vol}|{comp}|{substate}"
        if scope == "session_substate":
            return str(substate)
        return ""

    @staticmethod
    def _family_id_from_choice(chosen_entry: Dict[str, Any]) -> str:
        family_id = str(chosen_entry.get("family_id", "") or "").strip()
        if family_id:
            return family_id
        cand = (
            chosen_entry.get("cand", {})
            if isinstance(chosen_entry.get("cand"), dict)
            else {}
        )
        return build_family_id(
            timeframe=cand.get("timeframe", ""),
            session=chosen_entry.get("session", cand.get("session", "")),
            strategy_type=cand.get("strategy_type", ""),
            threshold=cand.get("thresh", cand.get("threshold", 0.0)),
            family_tag=cand.get("family_tag", ""),
        )

    def _choose_mode(
        self,
        *,
        support_trades: int,
        mode_map: Dict[str, Dict[str, float]],
        context: Dict[str, Any],
    ) -> tuple[str, str]:
        mode = "canonical"
        reason = "canonical_default"
        if not self.enable_adaptive_modes:
            return mode, "adaptive_disabled"
        band = str(context.get("confidence_band", "") or "").strip().lower()
        vol_regime = str(context.get("volatility_regime", "") or "").strip().lower()
        comp_regime = (
            str(context.get("compression_expansion_regime", "") or "")
            .strip()
            .lower()
        )
        has_conservative = isinstance(mode_map.get("conservative"), dict)
        has_aggressive = isinstance(mode_map.get("aggressive"), dict)

        if (
            band in self.low_confidence_bands
            and has_conservative
            and support_trades >= self.min_support_for_conservative
        ):
            return "conservative", "adaptive_low_confidence_band"

        if (
            band in self.high_confidence_bands
            and has_aggressive
            and support_trades >= self.min_support_for_aggressive
        ):
            if (
                self.aggressive_requires_expanding
                and comp_regime not in {"expanding", "expansion"}
            ):
                return "canonical", "aggressive_blocked_not_expanding"
            if (not self.allow_aggressive_in_high_vol) and vol_regime == "high":
                return "canonical", "aggressive_blocked_high_vol"
            return "aggressive", "adaptive_high_confidence_band"

        if band in {"mid", "medium", "neutral"}:
            if (
                vol_regime == "high"
                and has_conservative
                and support_trades >= self.min_support_for_conservative
            ):
                return "conservative", "adaptive_mid_confidence_high_vol"
            if (
                comp_regime in {"expanding", "expansion"}
                and has_aggressive
                and support_trades >= self.min_support_for_aggressive
                and ((self.allow_aggressive_in_high_vol) or vol_regime != "high")
            ):
                return "aggressive", "adaptive_mid_confidence_expanding"

        return mode, reason

    def _select_family_override(
        self,
        *,
        canonical_pair: Tuple[float, float],
        chosen_entry: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.enable_family_bracket_selector:
            return None
        selector = self.family_bracket_selector
        if not isinstance(selector, dict) or not bool(selector.get("enabled", False)):
            return None
        family_id = self._family_id_from_choice(chosen_entry)
        if not family_id:
            return None
        family_payload = (selector.get("families", {}) or {}).get(family_id, {})
        if not isinstance(family_payload, dict):
            return None
        scope_priority = selector.get("scope_priority", [])
        if not isinstance(scope_priority, list):
            scope_priority = []

        for scope in scope_priority:
            scope_payloads = (family_payload.get("context_overrides", {}) or {}).get(
                str(scope),
                {},
            )
            if not isinstance(scope_payloads, dict):
                continue
            match_key = self._scope_key(str(scope), context)
            match_payload = scope_payloads.get(match_key, {})
            if not isinstance(match_payload, dict):
                continue
            pair = self._norm_pair(match_payload.get("sl"), match_payload.get("tp"))
            if pair is None or pair == canonical_pair:
                continue
            return {
                "pair": pair,
                "mode": f"family_context_{scope}",
                "reason": f"family_selector_{scope}",
                "family_id": family_id,
                "scope": str(scope),
                "match_key": str(match_key),
            }

        global_default = family_payload.get("global_default", {})
        pair = (
            self._norm_pair(global_default.get("sl"), global_default.get("tp"))
            if isinstance(global_default, dict)
            else None
        )
        if pair is None or pair == canonical_pair:
            return None
        return {
            "pair": pair,
            "mode": "family_context_global",
            "reason": "family_selector_global",
            "family_id": family_id,
            "scope": "global",
            "match_key": "global",
        }

    def select_bracket(
        self,
        *,
        lane: str,
        variant_id: str,
        chosen_entry: Dict[str, Any],
        context_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context_inputs if isinstance(context_inputs, dict) else {}
        default_row = (
            self.bracket_defaults.get(variant_id, {})
            if isinstance(self.bracket_defaults.get(variant_id, {}), dict)
            else {}
        )
        mode_map = (
            self.bracket_modes.get(variant_id, {})
            if isinstance(self.bracket_modes.get(variant_id, {}), dict)
            else {}
        )
        cand = (
            chosen_entry.get("cand", {})
            if isinstance(chosen_entry.get("cand"), dict)
            else {}
        )

        canonical_sl = safe_float(default_row.get("sl", cand.get("sl", 0.0)), 0.0)
        canonical_tp = safe_float(default_row.get("tp", cand.get("tp", 0.0)), 0.0)
        support_trades = int(safe_float(default_row.get("support_trades", 0), 0))
        canonical_pair = self._norm_pair(canonical_sl, canonical_tp) or (
            float(canonical_sl),
            float(canonical_tp),
        )

        family_override = self._select_family_override(
            canonical_pair=canonical_pair,
            chosen_entry=chosen_entry,
            context=context,
        )
        if isinstance(family_override, dict):
            pair = family_override.get("pair", canonical_pair)
            return {
                "lane": str(lane),
                "variant_id": str(variant_id),
                "bracket_mode": str(family_override.get("mode", "family_context_global")),
                "selected_sl": float(pair[0]),
                "selected_tp": float(pair[1]),
                "canonical_default_used": bool(pair == canonical_pair),
                "bracket_reason": str(
                    family_override.get("reason", "family_selector_global")
                ),
                "canonical_sl": float(canonical_pair[0]),
                "canonical_tp": float(canonical_pair[1]),
                "support_trades": int(support_trades),
                "family_bracket_selector_used": True,
                "family_bracket_family_id": str(
                    family_override.get("family_id", "")
                ),
                "family_bracket_scope": str(family_override.get("scope", "")),
                "family_bracket_match_key": str(
                    family_override.get("match_key", "")
                ),
            }

        mode, reason = self._choose_mode(
            support_trades=support_trades,
            mode_map=mode_map,
            context=context,
        )
        selected = (
            mode_map.get(mode, {})
            if isinstance(mode_map.get(mode, {}), dict)
            else {}
        )
        selected_sl = safe_float(selected.get("sl", canonical_pair[0]), canonical_pair[0])
        selected_tp = safe_float(selected.get("tp", canonical_pair[1]), canonical_pair[1])
        if selected_sl <= 0.0:
            selected_sl = canonical_pair[0]
        if selected_tp <= 0.0:
            selected_tp = canonical_pair[1]
        selected_pair = self._norm_pair(selected_sl, selected_tp) or canonical_pair

        return {
            "lane": str(lane),
            "variant_id": str(variant_id),
            "bracket_mode": str(mode),
            "selected_sl": float(selected_pair[0]),
            "selected_tp": float(selected_pair[1]),
            "canonical_default_used": bool(selected_pair == canonical_pair),
            "bracket_reason": str(reason),
            "canonical_sl": float(canonical_pair[0]),
            "canonical_tp": float(canonical_pair[1]),
            "support_trades": int(support_trades),
            "family_bracket_selector_used": False,
            "family_bracket_family_id": "",
            "family_bracket_scope": "",
            "family_bracket_match_key": "",
        }
