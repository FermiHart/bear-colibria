"""Initialize and poll an actual public electricity stream, with archived vintages."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import uuid

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from lck.electricity_capture import FieldCapture
from lck.electricity_field import ElectricityField, iso, timestamp
from lck.online_reference import _json
from lck.public_electricity import fetch, hourly, recent_training, SourceError, METER, UNIT, ENDPOINT, DOCS


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once(path, data):
    """Exclusive durable publication; incomplete temporary files aren't inputs."""
    temp = path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp")
    with temp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()
    sync_dir(path.parent)


def adapter_hashes():
    return {name: hashlib.sha256((LAB/name).read_bytes()).hexdigest()
            for name in ("lck/public_electricity.py", "scripts/public_electricity.py")}


def event(stamp, value):
    return {"timestamp": iso(stamp), "meter_id": METER, "unit": UNIT, "value": value}


def save_snapshot(root, raw, metadata, values, diagnostics, before=None):
    body = root / "responses" / (metadata["sha256"]+".json")
    if body.exists():
        if hashlib.sha256(body.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("cached source response changed")
    else:
        write_once(body, raw)
    name = uuid.uuid4().hex+".json"
    record = {**metadata, "diagnostics": diagnostics, "hours": {str(k): v for k, v in values.items()},
              "forecast_before_request": before}
    write_once(root/"requests"/name, _json(record)+b"\n")
    return name


def history(root):
    seen = set()
    for path in sorted((root/"requests").glob("*.json")):
        row = json.loads(path.read_bytes())
        seen.update(int(k) for k in row["hours"])
    return seen


def accepted_values(root):
    values = {}
    with (root/"training.csv").open() as f:
        for row in csv.DictReader(f):
            values[timestamp(row["timestamp"])] = float(row["value"])
    with (root/"receipts.jsonl").open() as f:
        for line in f:
            sample = json.loads(line)["receipt"]["last_observation"]
            if sample is not None:
                values[timestamp(sample["timestamp"])] = sample["value"]
    return values


def save_run(root, report):
    report = {"adapter_sha256": adapter_hashes(), "recorded_at": time.time(), **report}
    write_once(root/"runs"/(uuid.uuid4().hex+".json"), _json(report)+b"\n")
    return report


def initialize(root, *, downloader=fetch):
    if root.exists() or root.is_symlink() or not root.parent.is_dir():
        raise ValueError("new pilot directory in existing parent required")
    now = int(time.time())
    raw, metadata = downloader(now-35*86400, now)
    values, diagnostics = hourly(raw, metadata["received_at"])
    train, evaluation = recent_training(values)
    root.mkdir(mode=0o700)
    sync_dir(root.parent)
    for child in ("responses", "requests", "runs"):
        (root/child).mkdir()
    sync_dir(root)
    snapshot = save_snapshot(root, raw, metadata, values, diagnostics)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=("timestamp", "meter_id", "unit", "value"), lineterminator="\n")
    writer.writeheader()
    writer.writerows(event(stamp, value) for stamp, value in train)
    write_once(root/"training.csv", text.getvalue().encode())
    with ElectricityField.create(root/"journal", root/"training.csv", METER, UNIT, source_kind="live") as field:
        prepared = field.status()
    pilot = {"schema": 1, "source": ENDPOINT, "country": "de", "series": "Load", "unit": UNIT,
             "timestamp_semantics": "end of hour; mean of4 complete15-minute MW intervals",
             "scope": "public aggregate grid load; not an individual customer meter",
             "license": "CC-BY-4.0", "attribution": "Energy-Charts.info / Fraunhofer ISE", "documentation": DOCS,
             "adapter_sha256": adapter_hashes(), "config_sha256": prepared["config_sha256"],
             "initial_snapshot": snapshot, "created_at": time.time(), "bootstrap_evaluation_hours": len(evaluation)}
    write_once(root/"pilot.json", _json(pilot)+b"\n")
    with FieldCapture.open(root/"journal", root/"receipts.jsonl", METER, UNIT,
                           config_sha256=pilot["config_sha256"]) as capture:
        for stamp, value in evaluation:
            capture.ingest(event(stamp, value))
        report = capture.field.report()
    return save_run(root, {"status": "initialized", "source_snapshot": snapshot,
        "public_source": pilot, "training_hours": len(train), "retrospective_bootstrap_targets": len(evaluation),
        "forecast_before_first_local_fetch_targets": 0, "latest_complete_hour": iso(max(values)),
        "source_lag_seconds": metadata["received_at"]-max(values), "field_report": report})


def poll(root, *, downloader=fetch):
    pilot = json.loads((root/"pilot.json").read_bytes())
    if pilot["adapter_sha256"] != adapter_hashes():
        raise ValueError("public adapter changed; versioned migration required")
    fd = os.open(root/"POLL.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Preserve all vintages. Stop at a bounded quota rather than deleting evidence.
        if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) >= 512*1024**2:
            raise ValueError("pilot cache reached512MiB; archival continuation required")
        for path in (root/"runs").glob("*.json"):
            previous = json.loads(path.read_bytes())
            if previous.get("retry_not_before", 0) > time.time():
                return {"status": "source_backoff", "retry_not_before": previous["retry_not_before"]}
        known = history(root)
        with FieldCapture.open(root/"journal", root/"receipts.jsonl", METER, UNIT,
                               config_sha256=pilot["config_sha256"]) as capture:
            prior = capture.field.status()
            expected = timestamp(prior["issued"]["target_timestamp"])
            now = int(time.time())
            start = min(now-48*3600, expected-48*3600)
            if now-start > 45*86400:
                raise ValueError("outage exceeds45-day catch-up window")
            try:
                raw, metadata = downloader(start, now)
            except SourceError as exc:
                return save_run(root, {"status": "source_unavailable", "http_status": exc.status,
                    "retry_not_before": time.time()+max(900, exc.retry_after), "error": str(exc),
                    "events_unchanged": prior["events"]})
            values, diagnostics = hourly(raw, metadata["received_at"])
            snapshot = save_snapshot(root, raw, metadata, values, diagnostics, before=prior)
            accepted = accepted_values(root)
            revisions = [{"timestamp": iso(k), "accepted": accepted[k], "latest": v}
                         for k, v in values.items() if k in accepted and accepted[k] != v]
            consumed, prefetch = [], 0
            next_stamp = expected
            while next_stamp in values and len(consumed) < 48:
                # Only the forecast already held BEFORE this HTTP request can
                # qualify. Forecasts made while consuming the same batch cannot.
                eligible = (not consumed and next_stamp not in known
                            and prior["forecast_available_at"] <= metadata["request_started_at"])
                receipt = capture.ingest(event(next_stamp, values[next_stamp]))
                consumed.append({"timestamp": iso(next_stamp), "value": values[next_stamp],
                                 "state_sha256": receipt["state_sha256"],
                                 "forecast_before_first_local_fetch": bool(eligible)})
                prefetch += int(eligible)
                next_stamp += 3600
            result = capture.field.report()
            status = "updated" if consumed else "waiting_for_source"
            if next_stamp not in values and values and max(values) > next_stamp:
                status = "blocked_source_gap"
            return save_run(root, {"status": status, "source_snapshot": snapshot,
                "latest_complete_hour": iso(max(values)) if values else None,
                "source_lag_seconds": metadata["received_at"]-max(values) if values else None,
                "consumed": consumed, "forecast_before_first_local_fetch_targets": prefetch,
                "revisions_preserved_without_relearning": revisions, "field_report": result})
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("initialize", "poll", "status"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.out is not None and (args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir()):
        parser.error("new report path in existing parent required")
    if args.mode == "initialize":
        report = initialize(args.directory)
    elif args.mode == "poll":
        report = poll(args.directory)
    else:
        pilot = json.loads((args.directory/"pilot.json").read_bytes())
        with ElectricityField.open(args.directory/"journal", METER, UNIT,
                                   expected_config_hash=pilot["config_sha256"]) as field:
            report = {"status": "public_pilot_status", "public_source": pilot, "field_report": field.report()}
    if args.out:
        write_once(args.out, _json(report)+b"\n")
    view = {k: v for k, v in report.items() if k not in ("field_report", "public_source", "adapter_sha256")}
    if "field_report" in report:
        field = report["field_report"]
        view.update(events=field["events"], physical_prospective_targets=field["metrics"]["prospective"]["selector"]["n"],
                    next_forecast=field["issued"], metrics=field["metrics"]["all"]["selector"])
    print(json.dumps(view, allow_nan=False))
    return 1 if report["status"] in ("source_unavailable", "blocked_source_gap") else 0


if __name__ == "__main__":
    raise SystemExit(main())
