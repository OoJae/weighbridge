"""Paired before/after comparisons of judge labels through AuditKIT (Lexsi Labs).

Our own judge labels (``label3``) are fed to AuditKIT as precomputed outputs with a bring-your-own scorer, and
`auditkit.compare(...).significance(...)` gives the paired bootstrap over prompts: the change, its 95% interval
and a p-value. Used for the repair's before -> after (C3) and for each sentinel against the base model.
AuditKIT's run cache is redirected to a temporary folder so nothing is written outside the project.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from typing import Any

METRICS = {"harmful_compliance": ("harmful", "complied"), "over_refusal": ("benign", "refused")}
BENIGN_KINDS = ("benign", "xstest_safe")


def _kit():
    """Import auditkit with its XDG cache and data folders pointed at a private temporary folder."""
    scratch = tempfile.mkdtemp(prefix="wb_auditkit_")
    for var in ("XDG_CACHE_HOME", "XDG_DATA_HOME"):
        os.environ.setdefault(var, scratch)
    import auditkit as ak

    return ak


def _matches(kind: str, group: str) -> bool:
    return kind == "harmful" if group == "harmful" else kind in BENIGN_KINDS


def paired_delta(before: Sequence[dict], after: Sequence[dict], metric: str = "harmful_compliance",
                 confidence: float = 0.95) -> dict[str, Any]:
    """AuditKIT paired bootstrap of `metric` (share of label rows with the target label), after minus before,
    in points, over the prompts both runs labelled. Rows need prompt_id, kind, lang and label3."""
    ak = _kit()
    group, target = METRICS[metric]
    b = {r["prompt_id"]: r for r in before if _matches(r["kind"], group)}
    a = {r["prompt_id"]: r for r in after if _matches(r["kind"], group)}
    ids = sorted(b.keys() & a.keys())
    if not ids:
        return {"metric": metric, "n": 0}

    @ak.scorer(direction=ak.Direction.MINIMIZE, name=metric)
    def score(sample, output):  # noqa: ARG001 - AuditKIT passes the sample too
        return float(output == target)

    def run(rows: dict[str, dict]):
        samples = [ak.Sample(input=pid, id=pid, task=metric, actual_output=rows[pid]["label3"],
                             metadata={"lang": rows[pid].get("lang", "en")}) for pid in ids]
        return ak.evaluate(samples, model="precomputed", scorers=[score], cache=False)

    sig = ak.compare(run(b), run(a)).significance(metric, confidence=confidence)
    sig = dict(sig)
    return {"metric": metric, "n": int(sig.get("n", len(ids))), "delta_points": 100 * float(sig["delta"]),
            "ci_points": [100 * float(x) for x in sig["ci"]], "p_value": float(sig["p_value"]),
            "significant": bool(sig.get("significant")), "tool": "auditkit==1.1.12 compare().significance()"}
