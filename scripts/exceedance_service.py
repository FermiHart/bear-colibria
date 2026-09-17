"""CEO Wave4: provable pre-event exceedance records from the running observers.

Reads the live observer databases READ-ONLY, computes P(load > contracted C)
for each confirmed, still-unobserved forecast using the frozen dev-residual CDF
(Wave1 adjusted winner), and appends a hash-chained ledger record linking the
probability to the forecast's payload hash — committed BEFORE the target hour.
The ledger enforces one record per forecast hash and fails closed on tampering.
Contract levels are SYNTHETIC scenarios (P75/P90 of the consumed Jun-Jul train),
honestly labeled; the CDF source report hash is pinned in every record.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from bisect import bisect_right
from pathlib import Path
import sqlite3
import sys
import time
import uuid

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from scripts.public_electricity import write_once

PLAN = LAB/"experiments/exceedance_d_v1.json"  # dataset sources (Wave1/3 lineage)
LEDGER_MAX = 32*1024**2


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    return json.loads((LAB/path).read_bytes())


def exceedance_probability(residuals, pred, C, lo=0.005, hi=0.995):
    """1 - F(C - pred) from the empirical residual CDF, clipped. Stdlib only."""
    ordered = sorted(residuals)
    rank = bisect_right(ordered, C-pred)
    return float(min(hi, max(lo, 1.0-rank/len(ordered))))


def quantile(values, q):
    """Linear-interpolation quantile, matching numpy's default, stdlib only."""
    ordered = sorted(values)
    k = q*(len(ordered)-1)
    f, i = math.modf(k)
    i = int(i)
    if i+1 >= len(ordered):
        return float(ordered[-1])
    return float(ordered[i]*(1-f)+ordered[i+1]*f)


def frozen_cdf():
    """Prefer the frozen CDF artifact; fall back to the archived dev report.

    The artifact (in the bundle and workspace) stores residuals rounded to 4
    decimals; records issued before it existed used exact values. Its source
    hash is verified when the source file is available. Returns (residuals,
    source_sha256, contract_levels) — levels from the artifact, or computed
    from the dataset when falling back.
    """
    artifact_path = LAB/"experiments/exceedance_cdf_v1.json"
    if artifact_path.is_file():
        artifact = load_json("experiments/exceedance_cdf_v1.json")
        source = LAB/artifact["source_report"]
        if source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() != artifact["source_sha256"]:
            raise ValueError("CDF source changed since the artifact was frozen")
        return sorted(artifact["residuals_mw"]), artifact["source_sha256"], artifact["contract_levels_mw"]
    plan = load_json("experiments/exceedance_electricity_v1.json")
    dev_rows = load_json(plan["dataset"]["dev_report"])["result"]["rows"]
    residuals = sorted(r["predictions"]["adjusted"]-r["value"] for r in dev_rows)
    values = load_json("data/risk_electricity_dataset_v1/dataset.json")["values"]
    levels = contract_levels(values, 720)
    return residuals, hashlib.sha256(
        (LAB/plan["dataset"]["dev_report"]).read_bytes()).hexdigest(), levels


def contract_levels(values, train):
    return {"P75": quantile(values[:train], .75),
            "P90": quantile(values[:train], .90)}


def pending_forecasts(db_path):
    if not Path(db_path).is_file():
        return []
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = []
        for target, raw, checksum, available in db.execute(
                "SELECT target,payload,checksum,available FROM forecasts ORDER BY target"):
            if available is None:
                continue
            payload = json.loads(raw)
            if digest(raw) != checksum:
                raise ValueError("forecast checksum differs")
            if db.execute("SELECT 1 FROM observations WHERE stamp=?", (target,)).fetchone():
                continue  # already observed; only pre-event records belong here
            if not {"weekly_ridge", "adjusted"} <= set(payload.get("predictions", {})):
                continue
            out.append({"target_end": target, "forecast_sha256": checksum,
                        "pred_adjusted": payload["predictions"]["adjusted"],
                        "pred_weekly_ridge": payload["predictions"]["weekly_ridge"],
                        "horizon": payload.get("horizon")})
        return out
    finally:
        db.close()


def load_ledger(path):
    rows, previous = [], None
    if not path.exists():
        return rows, previous
    with path.open("rb") as f:
        for raw in f:
            row = json.loads(raw)
            body = {k: row[k] for k in ("previous", "record")}
            if row["sha256"] != digest(json.dumps(body, sort_keys=True).encode()):
                raise ValueError("ledger checksum differs")
            if row["previous"] != previous:
                raise ValueError("ledger chain differs")
            previous, _ = row["sha256"], rows.append(row)
    return rows, previous


def issue(root, observers, out_path):
    residuals, cdf_source, levels = frozen_cdf()
    values = load_json("data/risk_electricity_dataset_v1/dataset.json")["values"] if not (
        LAB/"experiments/exceedance_cdf_v1.json").is_file() else None
    fd = os.open(out_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, previous = load_ledger(out_path)
        if os.fstat(fd).st_size > LEDGER_MAX:
            raise ValueError("ledger full; archival continuation required")
        seen = {r["record"]["forecast_sha256"] for r in rows}
        appended = 0
        for name, relative in observers:
            for forecast in pending_forecasts(Path(root)/relative):
                if forecast["forecast_sha256"] in seen:
                    continue
                record = {**forecast, "levels": {
                    name_level: exceedance_probability(residuals, forecast["pred_adjusted"], C)
                    for name_level, C in levels.items()},
                    "observer": name, "cdf_source_sha256": cdf_source,
                    "contract_scope": "synthetic scenario levels from consumed Jun-Jul train; research only",
                    "issued_at": time.time()}
                body = {"previous": previous, "record": record}
                sha = digest(json.dumps(body, sort_keys=True).encode())
                line = json.dumps({**body, "sha256": sha}, sort_keys=True)+"\n"
                if os.fstat(fd).st_size+len(line) > LEDGER_MAX:
                    raise ValueError("ledger full; stop before continuing")
                os.lseek(fd, 0, os.SEEK_END)
                os.write(fd, line.encode())
                os.fsync(fd)
                previous, seen = sha, seen | {forecast["forecast_sha256"]}
                appended += 1
        return {"appended": appended, "ledger_records": len(rows)+appended,
                "ledger_sha256": previous, "levels": levels, "cdf_source_sha256": cdf_source}
    finally:
        os.close(fd)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=LAB/"data")
    p.add_argument("--observers", nargs=2, action="append",
                   default=[["v1-weekly_ridge", "future_electricity_live_v1/future.sqlite"],
                            ["risk-blend50", "risk_electricity_live_v1/future.sqlite"]])
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--report", type=Path)
    args = p.parse_args(argv)
    result = issue(args.root, [tuple(o) for o in args.observers], args.ledger)
    if args.report:
        if args.report.exists() or args.report.is_symlink() or not args.report.parent.is_dir():
            p.error("new report path in existing parent required")
        write_once(args.report, json.dumps(result, sort_keys=True, allow_nan=False).encode()+b"\n")
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
