"""Issue and score complete FUTURE hours using the existing public collector."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from lck.future_electricity import FutureElectricity
from lck.electricity_field import ElectricityField, timestamp
from lck.online_reference import _json
from lck.public_electricity import METER, UNIT
from scripts.monitor_public_electricity import receipts
from scripts.public_electricity import write_once, sync_dir, adapter_hashes
from scripts.future_electricity_study import SOURCE_PATHS
from lck.electricity_session import SOURCE_FILES

RUNTIME_FILES = tuple(sorted(set(SOURCE_PATHS+tuple("lck/"+p for p in SOURCE_FILES)+(
    "lck/future_electricity.py", "lck/electricity_field.py", "lck/electricity_capture.py",
    "scripts/future_electricity_pilot.py", "scripts/qualify_future_electricity.py",
    "scripts/monitor_public_electricity.py", "scripts/public_electricity.py"))))


def source_snapshot(root):
    lock = os.open(root/"POLL.lock", os.O_RDONLY)
    archive = None
    try:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        archive = os.open(root/"receipts.jsonl", os.O_RDONLY)
        fcntl.flock(archive, fcntl.LOCK_SH | fcntl.LOCK_NB)
        pilot = json.loads((root/"pilot.json").read_bytes())
        if pilot["adapter_sha256"] != adapter_hashes():
            raise ValueError("public source adapter changed")
        rows, receipt_hash = receipts(archive)
        with ElectricityField.open(root/"journal", METER, UNIT, min_events=len(rows)-1,
                                   expected_config_hash=pilot["config_sha256"]) as field:
            status = field.status()
        if status["events"] != len(rows)-1 or status["state_sha256"] != rows[-1]["state_sha256"]:
            raise ValueError("source collector reconciliation required")
        raw = (root/"training.csv").read_bytes()
        if hashlib.sha256(raw).hexdigest() != status["config"]["train_sha256"]:
            raise ValueError("source training changed")
        import io
        records = [(timestamp(r["timestamp"]), float(r["value"])) for r in csv.DictReader(io.StringIO(raw.decode()))]
        records.extend((timestamp(r["last_observation"]["timestamp"]), r["last_observation"]["value"]) for r in rows[1:])
        return records, status["config"], {"config_sha256": pilot["config_sha256"], "receipt_archive_sha256": receipt_hash,
                                           "last_state_sha256": status["state_sha256"]}
    finally:
        if archive is not None:
            os.close(archive)
        os.close(lock)


def verify_release(path):
    release = json.loads(path.read_bytes())
    if set(release["source_sha256"]) != set(RUNTIME_FILES):
        raise ValueError("incomplete frozen future source set")
    for name, expected in release["source_sha256"].items():
        if hashlib.sha256((LAB/name).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen future source changed")
    if release["scope"] != "research_only" or release["operational_decision"] != "OPERATIONAL_PASS":
        raise ValueError("qualified research release required")
    return release


def cycle(root, public, release_path, *, initialize=False):
    release = verify_release(release_path)
    records, config, provenance = source_snapshot(public)
    if initialize:
        service = FutureElectricity.create(root, records, config["mean"], config["std"], release["policy"],
            source_config_hash=provenance["config_sha256"], protocol_hash=release["plan_fingerprint"])
        try:
            (root/"runs").mkdir()
            sync_dir(root)
            write_once(root/"release.json", _json(release)+b"\n")
        except BaseException:
            service.close()
            raise
    else:
        stored = json.loads((root/"release.json").read_bytes())
        if stored != release:
            raise ValueError("deployment release differs")
        service = FutureElectricity.open(root, source_config_hash=provenance["config_sha256"])
    with service:
        consumption = service.ingest(records)
        # Reconcile any prepared forecast only if its original deadline is still met.
        for row in service._forecasts():
            if row["available_at"] is None:
                service.confirm(row["target_end"])
        try:
            target = service.prepare()
            issued = service.confirm(target)
            issue_status = "confirmed_future_hour" if issued.get("eligible") else "deadline_not_met"
        except ValueError as exc:
            # Source staleness suppresses issuance; accepted delayed labels still score.
            issued, issue_status = {"reason": str(exc)}, "issuance_paused"
        report = {"status": "ok", "as_of": time.time(), "scope": "research_only", "consumption": consumption,
                  "source_provenance": provenance, "issue_status": issue_status, "issued": issued,
                  "report": service.report(), "retrospective_decision": release["retrospective_decision"]}
    write_once(root/"runs"/(uuid.uuid4().hex+".json"), _json(report)+b"\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("initialize", "tick", "status"))
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--public-directory", type=Path, default=LAB/"data/public_electricity_de_v1")
    p.add_argument("--release", type=Path, default=LAB/"reports/future_electricity_release_v1.json")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    if args.out and (args.out.exists() or not args.out.parent.is_dir()):
        p.error("new report required")
    if args.mode == "status":
        verify_release(args.release)
        with FutureElectricity.open(args.directory) as service:
            report = {"status": "ok", "report": service.report()}
    else:
        report = cycle(args.directory, args.public_directory, args.release, initialize=args.mode == "initialize")
    if args.out:
        write_once(args.out, _json(report)+b"\n")
    print(json.dumps({"status": report["status"], "issue_status": report.get("issue_status"), "issued": report.get("issued"),
        "prepared": report["report"]["prepared"], "confirmed": report["report"]["confirmed"],
        "scored_future_hours": report["report"]["scored_future_hours"], "decision": report["report"]["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
