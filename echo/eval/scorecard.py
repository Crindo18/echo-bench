"""Model scorecard: turn all the measured factors into a defensible choice.

Picking an encoder is a multi-criteria decision, not a single accuracy number.
This module gathers, per model, everything the benchmark measured — footprint,
budget compliance, few-shot accuracy and adaptation gain, accuracy on the
*hardest* severity tier, latency / RTF / RAM on the target, and cross-validation
stability — and produces a ranked table plus a recommendation.

It is deliberately transparent and tunable rather than a black box:

  * every raw number is shown;
  * each criterion is min-max normalised across the compared models to [0,1]
    (cost criteria like size and latency are inverted so higher is always
    better), then combined with **weights you control**;
  * a model that misses the ≤16 MB budget is flagged and disqualified from the
    recommendation (but still shown), because it cannot deploy;
  * the recommendation is "highest weighted score among budget-compliant
    models", and the weights are printed alongside so the ranking is auditable.

Change the weights to change the priorities — there is no hidden objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Default priorities. Higher weight = matters more. Tune freely.
DEFAULT_WEIGHTS = {
    "accuracy": 0.30,       # post@K on all test speakers
    "hardest_tier": 0.25,   # post@K on the most-impaired tier (atypical focus)
    "gain": 0.10,           # personalization benefit (post − SI)
    "footprint": 0.15,      # INT8 size (smaller better)
    "latency": 0.15,        # ms per utterance on target (faster better)
    "stability": 0.05,      # cross-fold CI (tighter better)
}

# Which tiers count as "hardest" per corpus severity naming (most impaired first).
_HARDEST_ORDER = ["severe", "very_low", "moderate", "low", "mild", "mid", "high"]


@dataclass
class ModelCard:
    name: str
    params_M: float
    int8_mb: float
    budget_ok: bool
    # accuracy block (at the chosen primary K)
    post_acc: float
    gain: float
    hardest_tier: str
    hardest_tier_acc: float
    stability_ci: float           # CI width of post@K across folds (smaller better)
    # hardware block (optional; NaN if not measured)
    latency_ms: float = float("nan")
    rtf: float = float("nan")
    peak_ram_mb: float = float("nan")
    extra: Dict = field(default_factory=dict)


def card_from_measurements(name, footprint: Dict, accuracy: Dict,
                           latency: Optional[Dict] = None,
                           primary_k: int = 5) -> ModelCard:
    """Assemble a ModelCard from footprint_report + cross_validate (+ latency)."""
    if primary_k not in accuracy.get("post", {}):
        avail = sorted(accuracy.get("post", {}))
        raise KeyError(f"primary_k={primary_k} was not measured (available K: "
                       f"{avail}). Pass a primary_k in --ks.")
    # hardest available tier
    tiers = accuracy.get("per_tier", {})
    hardest, h_acc = "n/a", float("nan")
    for t in _HARDEST_ORDER:
        if t in tiers and primary_k in tiers[t]:
            hardest, h_acc = t, tiers[t][primary_k]["mean"]
            break
    post = accuracy["post"][primary_k]
    return ModelCard(
        name=name,
        params_M=footprint.get("encoder_params_M", float("nan")),
        int8_mb=footprint.get("onnx_int8_MB", float("nan")),
        budget_ok=bool(footprint.get("params_within_budget", False)),
        post_acc=post["mean"], gain=accuracy["gain"][primary_k]["mean"],
        hardest_tier=hardest, hardest_tier_acc=h_acc,
        stability_ci=post["ci95"],
        latency_ms=(latency or {}).get("latency_ms_mean", float("nan")),
        rtf=(latency or {}).get("rtf", float("nan")),
        peak_ram_mb=(latency or {}).get("peak_ram_mb", float("nan")),
    )


def weights_without_severity(weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Return a weight set with the hardest-tier criterion removed (its weight
    redistributed across the others by ``build_scorecard``'s renormalisation).
    Use this together with an unstratified plan when severity is toggled off —
    the per-tier accuracy is still shown, it just no longer drives the ranking."""
    w = dict(weights or DEFAULT_WEIGHTS)
    w["hardest_tier"] = 0.0
    return w


def _norm(values: List[float], invert: bool) -> List[float]:
    xs = [v if v == v else None for v in values]
    present = [v for v in xs if v is not None]
    if not present:
        return [0.0] * len(values)
    lo, hi = min(present), max(present)
    out = []
    for v in xs:
        if v is None:
            out.append(0.0)
        elif hi == lo:
            out.append(1.0)
        else:
            s = (v - lo) / (hi - lo)
            out.append(1.0 - s if invert else s)
    return out


def build_scorecard(cards: List[ModelCard],
                    weights: Optional[Dict[str, float]] = None) -> Dict:
    """Score and rank models. Returns a dict with per-model scores and a pick."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    total_w = sum(w.values()) or 1.0
    w = {k: v / total_w for k, v in w.items()}          # normalise weights

    crit = {
        "accuracy":     ([c.post_acc for c in cards], False),
        "hardest_tier": ([c.hardest_tier_acc for c in cards], False),
        "gain":         ([c.gain for c in cards], False),
        "footprint":    ([c.int8_mb for c in cards], True),   # smaller better
        "latency":      ([c.latency_ms for c in cards], True),
        "stability":    ([c.stability_ci for c in cards], True),  # tighter better
    }
    normed = {k: _norm(vals, inv) for k, (vals, inv) in crit.items()}

    rows = []
    for i, c in enumerate(cards):
        contrib = {k: w.get(k, 0.0) * normed[k][i] for k in crit}
        score = sum(contrib.values())
        rows.append(dict(name=c.name, card=c, score=score,
                         contributions=contrib))
    rows.sort(key=lambda r: r["score"], reverse=True)

    eligible = [r for r in rows if r["card"].budget_ok]
    pick = eligible[0]["name"] if eligible else None
    return dict(weights=w, rows=rows, recommendation=pick,
                disqualified=[r["name"] for r in rows if not r["card"].budget_ok])


def print_scorecard(sc: Dict, primary_k: int = 5) -> None:
    print("\n" + "=" * 78)
    print(f"MODEL SCORECARD  (primary K={primary_k}; weights: " +
          ", ".join(f"{k}={v:.2f}" for k, v in sc["weights"].items()) + ")")
    print("=" * 78)
    hdr = (f"{'model':<15}{'params':>8}{'INT8MB':>8}{'budget':>7}"
           f"{'acc':>7}{'hard':>7}{'gain':>7}{'lat_ms':>8}{'score':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in sc["rows"]:
        c = r["card"]
        budget = "ok" if c.budget_ok else "OVER"
        lat = f"{c.latency_ms:.0f}" if c.latency_ms == c.latency_ms else "-"
        print(f"{c.name:<15}{c.params_M:>7.1f}M{c.int8_mb:>7.1f}{budget:>7}"
              f"{c.post_acc*100:>6.1f}%{c.hardest_tier_acc*100:>6.1f}%"
              f"{c.gain*100:>+6.1f}{lat:>8}{r['score']:>7.3f}")
    print("-" * len(hdr))
    if sc["recommendation"]:
        top = sc["rows"][0]["card"]
        print(f"RECOMMENDATION: {sc['recommendation']}  "
              f"(best weighted score among budget-compliant models; "
              f"hardest tier '{top.hardest_tier}' {top.hardest_tier_acc*100:.1f}%)")
    else:
        print("RECOMMENDATION: none — no model met the ≤16 MB budget.")
    if sc["disqualified"]:
        print(f"Disqualified (over budget): {', '.join(sc['disqualified'])}")
    print("Weights are tunable — re-run build_scorecard(weights=...) to reprioritise.")
