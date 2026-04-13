from typing import Any, Dict, List, Optional

from de3_v4_schema import safe_float


class DE3V4LaneSelector:
    def __init__(
        self,
        *,
        lane_inventory: Optional[Dict[str, List[str]]],
        lane_variant_quality: Optional[Dict[str, Dict[str, Any]]],
        lane_anchor_variants: Optional[Dict[str, str]],
        runtime_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.lane_inventory = dict(lane_inventory) if isinstance(lane_inventory, dict) else {}
        self.lane_variant_quality = (
            dict(lane_variant_quality) if isinstance(lane_variant_quality, dict) else {}
        )
        self.lane_anchor_variants = (
            dict(lane_anchor_variants) if isinstance(lane_anchor_variants, dict) else {}
        )
        self.runtime_cfg = dict(runtime_cfg) if isinstance(runtime_cfg, dict) else {}
        self.weights = (
            dict(self.runtime_cfg.get("weights", {}))
            if isinstance(self.runtime_cfg.get("weights"), dict)
            else {}
        )
        self.w_edge = safe_float(self.weights.get("edge_points", 0.60), 0.60)
        self.w_struct = safe_float(self.weights.get("structural_score", 0.20), 0.20)
        self.w_prior = safe_float(self.weights.get("variant_quality_prior", 0.20), 0.20)

    def _variant_prior(self, variant_id: str) -> float:
        quality = self.lane_variant_quality.get(variant_id, {})
        if not isinstance(quality, dict):
            return 0.0
        return safe_float(quality.get("satellite_quality_score", quality.get("quality_proxy", 0.0)), 0.0)

    def select(
        self,
        *,
        lane: str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        lane_set = set(self.lane_inventory.get(lane, []) or [])
        rows: List[Dict[str, Any]] = []
        rows_fallback_any: List[Dict[str, Any]] = []
        use_inventory_filter = bool(lane_set)
        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            cand = entry.get("cand", {}) if isinstance(entry.get("cand"), dict) else {}
            variant_id = str(entry.get("cand_id") or cand.get("strategy_id") or "").strip()
            if not variant_id:
                continue
            edge = safe_float(entry.get("edge_points", 0.0), 0.0)
            struct = safe_float(
                entry.get(
                    "structural_score",
                    cand.get("StructuralScore", cand.get("score_raw", 0.0)),
                ),
                0.0,
            )
            prior = self._variant_prior(variant_id)
            lane_score = float((self.w_edge * edge) + (self.w_struct * struct) + (self.w_prior * prior))
            row = dict(entry)
            row["selected_variant_id"] = variant_id
            row["lane_score"] = float(lane_score)
            row["lane_score_components"] = {
                "edge_component": float(self.w_edge * edge),
                "structural_component": float(self.w_struct * struct),
                "variant_quality_prior_component": float(self.w_prior * prior),
            }
            row["lane_score_inputs"] = {
                "edge_points": float(edge),
                "structural_score": float(struct),
                "variant_quality_prior": float(prior),
            }
            rows_fallback_any.append(row)
            if not use_inventory_filter or variant_id in lane_set:
                rows.append(row)

        if use_inventory_filter and not rows:
            rows = rows_fallback_any

        if not rows:
            return {
                "selected_variant_id": "",
                "chosen_entry": None,
                "runner_up_entry": None,
                "lane_scores": [],
                "lane_selection_reason": "no_lane_candidates",
                "lane_candidate_count": 0,
            }

        rows.sort(key=lambda r: float(r.get("lane_score", 0.0)), reverse=True)
        chosen = rows[0]
        runner_up = rows[1] if len(rows) > 1 else None
        return {
            "selected_variant_id": str(chosen.get("selected_variant_id", "")),
            "chosen_entry": dict(chosen),
            "runner_up_entry": dict(runner_up) if isinstance(runner_up, dict) else None,
            "lane_scores": [
                {
                    "variant_id": str(row.get("selected_variant_id", "")),
                    "lane_score": float(row.get("lane_score", 0.0)),
                    "components": dict(row.get("lane_score_components", {})),
                    "inputs": dict(row.get("lane_score_inputs", {})),
                }
                for row in rows
            ],
            "lane_selection_reason": "ranked_lane_local_variant",
            "lane_candidate_count": int(len(rows)),
        }
