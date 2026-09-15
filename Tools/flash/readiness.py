"""Bounded cooldown shared by serial Flash performance studies."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import thermal_readiness
from common import EvidenceError


def wait_for_nominal(*, stable_seconds=30, max_seconds=900, interval=5,
                     reader=thermal_readiness.observe, clock=time.monotonic, sleep=time.sleep):
    """Require a quiet window of nominal observations, outside model timing."""
    if not 0 < stable_seconds <= max_seconds or not 0 < interval <= 30:
        raise EvidenceError("invalid benchmark cooldown policy")
    started = clock()
    nominal_since = None
    observations = []
    previous = None
    while True:
        observation = reader()
        now = clock()
        observations.append({"elapsed_seconds": now - started, **observation})
        conditions = observation.get("conditions", {})
        ready = conditions.get("thermalState") == "nominal" and conditions.get("lowPowerModeEnabled") is False
        if conditions != previous:
            print("BENCH READINESS", conditions, flush=True)
            previous = dict(conditions)
        if ready:
            if nominal_since is None: nominal_since = now
            if now - nominal_since >= stable_seconds:
                return {"stable_seconds": stable_seconds, "interval_seconds": interval,
                        "elapsed_seconds": now - started, "observations": observations}
        else:
            nominal_since = None
        if now - started >= max_seconds:
            raise EvidenceError("benchmark cooldown did not reach nominal conditions")
        sleep(min(interval, max_seconds - (now - started)))

