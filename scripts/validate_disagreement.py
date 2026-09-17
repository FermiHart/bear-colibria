"""Prospective validation of the disagreement-uncertainty finding (read-only).

Finding 1 (glm_original_findings_v1.md): expert disagreement at issuance
d = |pred_weekly_ridge - pred_adjusted| predicted the best-expert error in 3/4
consumed windows. This validator recomputes d from the STORED forecast payloads
of the running observers and joins it with arriving labels. It opens the live
databases read-only, never perturbs writers, computes no gates and promotes
nothing. With enough accumulated points the pooled terciles become the
prospective test; before that it honestly reports insufficient.
"""
import argparse
import json
import math
import sqlite3
import sys
import time
import uuid
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from scripts.public_electricity import write_once

DEFAULT_OBSERVERS = (
    ("v1-weekly_ridge", "data/future_electricity_live_v1/future.sqlite"),
    ("risk-blend50", "data/risk_electricity_live_v1/future.sqlite"),
)


def scored_points(db_path):
    if not Path(db_path).is_file():
        return None
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = []
        for target, raw, available in db.execute(
                "SELECT target,payload,available FROM forecasts ORDER BY target"):
            if available is None:
                continue
            payload = json.loads(raw)
            preds = payload.get("predictions") or {}
            if not {"weekly_ridge", "adjusted"} <= set(preds):
                continue
            observation = db.execute("SELECT value FROM observations WHERE stamp=?", (target,)).fetchone()
            if observation is None:
                continue
            d = abs(preds["weekly_ridge"]-preds["adjusted"])
            out.append({"target_end": target, "horizon": payload.get("horizon"), "d": d,
                        "best_err_mw": min(abs(preds["weekly_ridge"]-observation[0]),
                                           abs(preds["adjusted"]-observation[0])),
                        "pred_weekly_ridge": preds["weekly_ridge"],
                        "pred_adjusted": preds["adjusted"], "value": observation[0]})
        return out
    finally:
        db.close()


def tercile_check(points):
    if len(points) < 9:
        return {"status": "insufficient", "n": len(points),
                "note": "terciles require at least9 scored points"}
    ordered = sorted(points, key=lambda p: p["d"])
    n = len(ordered)
    summary = []
    for label, part in zip(("low", "mid", "high"),
                           (ordered[:n//3], ordered[n//3:2*n//3], ordered[2*n//3:])):
        summary.append({"tercile": label, "n": len(part),
                        "mean_d": math.fsum(p["d"] for p in part)/len(part),
                        "mean_best_err_mw": math.fsum(p["best_err_mw"] for p in part)/len(part)})
    return {"status": "descriptive", "n": n, "terciles": summary,
            "high_above_low": summary[2]["mean_best_err_mw"] > summary[0]["mean_best_err_mw"]}


def validate(root=LAB, observers=DEFAULT_OBSERVERS):
    observers_out, pooled = [], []
    for name, relative in observers:
        pts = scored_points(Path(root)/relative)
        if pts is None:
            observers_out.append({"observer": name, "status": "missing_database"})
            continue
        pooled.extend(pts)
        observers_out.append({"observer": name, "status": "ok", "n": len(pts), "points": pts})
    return {"schema": 1, "as_of": time.time(), "scope": "read-only prospective validation; no gates, no promotion",
            "observers": observers_out, "pooled": tercile_check(pooled)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=LAB)
    parser.add_argument("--out-directory", type=Path, default=LAB/"data/future_electricity_live_v1/monitoring")
    args = parser.parse_args(argv)
    report = validate(args.root)
    args.out_directory.mkdir(parents=True, exist_ok=True)
    write_once(args.out_directory/(f"disagreement-{uuid.uuid4().hex}.json"),
               json.dumps(report, sort_keys=True, allow_nan=False).encode()+b"\n")
    view = [{"observer": o["observer"], "status": o["status"], "n": o.get("n", 0)} for o in report["observers"]]
    print(json.dumps({"observers": view, "pooled": report["pooled"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
