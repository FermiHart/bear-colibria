"""Bounded, credential-free Energy-Charts Load ingestion (Germany, MW).

Attribution: Energy-Charts.info / Fraunhofer ISE, CC BY4.0.
API timestamps label interval starts. Four complete15-minute Load values are
averaged to an hourly MW value labeled by the interval END for the field API.
"""
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://api.energy-charts.info/public_power"
DOCS = "https://api.energy-charts.info/openapi.json"
METER = "energycharts-de-load"
UNIT = "MW"
MAX_BODY = 8*1024**2


class SourceError(RuntimeError):
    def __init__(self, message, *, status=None, retry_after=0):
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


def fetch(start, end):
    if type(start) is not int or type(end) is not int or not 0 < end-start <= 45*86400:
        raise ValueError("source requests must span0..45 days")
    url = ENDPOINT+"?"+urllib.parse.urlencode({"country": "de", "start": start, "end": end})
    request = urllib.request.Request(url, headers={"Accept": "application/json",
        "User-Agent": "Bear-ColibrIA-public-electricity/1 (local research; CC-BY attribution retained)"})
    begun = time.time()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(MAX_BODY+1)
            if len(raw) > MAX_BODY:
                raise SourceError("source response exceeds8MiB")
            headers = {name: response.headers.get(name) for name in ("Date", "ETag", "Last-Modified", "Content-Type")}
            if response.status != 200:
                raise SourceError("unexpected source status", status=response.status)
    except urllib.error.HTTPError as exc:
        # No busy retries. The caller persists a not-before time for later polls.
        delay = exc.headers.get("Retry-After", "900")
        try:
            seconds = max(60, int(delay))
        except ValueError:
            from email.utils import parsedate_to_datetime
            try:
                seconds = max(60, int(parsedate_to_datetime(delay).timestamp()-time.time()))
            except (ValueError, TypeError, OverflowError):
                seconds = 900
        raise SourceError(f"source HTTP {exc.code}", status=exc.code, retry_after=seconds) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SourceError("source connection failed", retry_after=900) from exc
    finished = time.time()
    return raw, {"url": url, "request_started_at": begun, "received_at": finished,
                 "headers": headers, "sha256": hashlib.sha256(raw).hexdigest(),
                 "attribution": "Energy-Charts.info / Fraunhofer ISE", "license": "CC-BY-4.0"}


def hourly(raw, as_of):
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get("deprecated") is not False:
        raise ValueError("unexpected or deprecated Energy-Charts schema")
    times, series = document["unix_seconds"], document["production_types"]
    if not isinstance(times, list) or not 1 <= len(times) <= 100000:
        raise ValueError("invalid source timestamp vector")
    loads = [s["data"] for s in series if s.get("name") == "Load"]
    if len(loads) != 1 or len(loads[0]) != len(times):
        raise ValueError("exactly one aligned Load series required")
    groups, previous, nulls = {}, None, 0
    for stamp, value in zip(times, loads[0]):
        if type(stamp) is not int or stamp % 900 or (previous is not None and stamp <= previous):
            raise ValueError("source timestamps must be unique ordered UTC quarter-hours")
        previous = stamp
        if value is None:
            nulls += 1
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e9:
            raise ValueError("invalid Load value")
        start = stamp//3600*3600
        groups.setdefault(start, {})[stamp-start] = float(value)
    values = {start+3600: math.fsum(points[k] for k in (0, 900, 1800, 2700))/4
              for start, points in groups.items()
              if set(points) == {0, 900, 1800, 2700} and start+3600 <= as_of}
    return values, {"quarter_hour_points": len(times), "null_points": nulls,
                    "complete_hours": len(values), "incomplete_or_unfinished_hours": len(groups)-len(values)}


def recent_training(values, warmup=720, evaluation=24):
    if not values:
        raise ValueError("no complete public hours available")
    end = max(values)
    chosen = [(end-i*3600, values.get(end-i*3600)) for i in reversed(range(warmup+evaluation))]
    if any(value is None for _, value in chosen):
        raise ValueError("latest public window has gaps;744 contiguous hours required")
    return chosen[:warmup], chosen[warmup:]
