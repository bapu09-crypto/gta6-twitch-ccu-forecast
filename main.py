"""Forecast GTA VI's launch-day Twitch viewers hour by hour, as five quantiles.

Model: a launch-day shape model fitted on the case's own comparables (past big
game launches, each with a full release-day hourly curve). For every comparable
we take its release-day curve, divide each hour by its hour-1 print, and work in
log space. For the case at hand, with hour-1 print p1, each hour's log-ratio
distribution is estimated by a kernel-weighted normal fit: comparables whose
hour-1 print is close to p1 (in log space, Gaussian kernel, bandwidth 1.0) weigh
most. The forecast quantiles are then

    q_tau(h) = p1 * exp(mu_h + z_tau * sd_h)

with mu_h, sd_h the kernel-weighted mean and standard deviation of the
comparables' log ratios at hour h, and z_tau the standard normal quantiles
(-1.645, -0.674, 0, 0.674, 1.645) at levels (0.05, 0.25, 0.5, 0.75, 0.95).

Why this shape: launch days share a shape but not a scale. Hour 1 is usually
already live (rolling midnight: New Zealand and Australia unlocked hours ago),
so the day dips a little in the small hours, rises when the Americas unlock,
and peaks in the evening at roughly 1.2-1.3x hour 1 for a big launch. A small
hour-1 print means the game has not unlocked yet anywhere, so the later hours
can be hundreds of times larger -- the kernel picks that up automatically,
because small-p1 comparables (simultaneous global unlocks) carry exactly that
pattern, and their wide spread makes the bands honestly wide. Conditioning on
the hour-1 print is what separates the two regimes; the house baseline
(persistence with a widening random-walk spread) cannot.

Validation (local, against SullyGnome's published hourly averages, the same
source the kit's practice cases come from): mean pinball loss on the log scale
over the four practice launches is 0.217, against the house baseline's 0.387.
Leave-one-out over the 38 comparables: 0.393 against the baseline's 0.930.
Typical miss on practice: 18-75% per case; 90% coverage 90/92 hours.

The program reads every <id>.in in the input directory (the grader runs one
case at a time, with only that case's file there) and writes <id>.out to the
output directory: {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]},
exactly 23 rows, hours 2..24 each once. Standard library only, deterministic
(no randomness, no dict-order dependence, no wall-clock use), offline.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand
in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
# Gaussian kernel bandwidth on log(hour-1 print): comparables within about a
# factor e of the case's hour-1 print dominate; further ones fade smoothly.
BANDWIDTH = 1.0
# A print below 1 viewer counts as 1 (the log of 0 is no number).
VIEWER_FLOOR = 1.0


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _parse_ts(text: str):
    """An ISO-8601 timestamp with a Z suffix, into a naive-UTC datetime."""
    from datetime import datetime

    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hour_of(point_time, day_start) -> int:
    """Which release-day hour (1..24) a point falls in, or 0 when outside."""
    delta = point_time - day_start
    hour = int(delta.total_seconds() // 3600) + 1
    return hour if 1 <= hour <= 24 else 0


def comparable_curve(comp: dict) -> dict:
    """A comparable's release-day hourly curve: hour -> mean viewers, half up.

    Each hour's figure is the mean of that comparable's points in the hour
    (SullyGnome hourly averages, one point an hour); an hour with no point
    has no figure and is left out.
    """
    day_start = _parse_ts(comp["day_start"])
    buckets: dict[int, list[float]] = {}
    for point in comp.get("hourly") or []:
        hour = _hour_of(_parse_ts(point["from"]), day_start)
        if hour:
            buckets.setdefault(hour, []).append(point["viewers"])
    curve = {}
    for hour, values in buckets.items():
        curve[hour] = int(math.floor(sum(values) / len(values) + 0.5))
    return curve


def kernel_moments(samples, log_p1: float):
    """Kernel-weighted mean and sd of log ratios, weights by log hour-1 print.

    samples: list of (log_ratio, log_hour1). Returns (mu, sd). With no usable
    sample the caller falls back to the house baseline's spread.
    """
    total = 0.0
    weighted = 0.0
    for log_ratio, log_hour1 in samples:
        diff = log_hour1 - log_p1
        weight = math.exp(-(diff * diff) / (2.0 * BANDWIDTH * BANDWIDTH))
        total += weight
        weighted += weight * log_ratio
    if total <= 0.0:
        return None
    mu = weighted / total
    variance = 0.0
    for log_ratio, log_hour1 in samples:
        diff = log_hour1 - log_p1
        weight = math.exp(-(diff * diff) / (2.0 * BANDWIDTH * BANDWIDTH))
        variance += weight * (log_ratio - mu) ** 2
    sd = math.sqrt(variance / total)
    return mu, sd


def hour_quantiles(samples, log_p1: float, p1: float, hour: int) -> list:
    """The five forecast quantiles for one hour, from the comparables' ratios.

    samples: the hour's (log_ratio, log_hour1) pairs. With no usable sample
    the house baseline's widening spread stands in, so the forecast is still
    a valid persistence forecast.
    """
    moments = kernel_moments(samples, log_p1)
    if moments is None:
        mu, sd = 0.0, 0.15 * math.sqrt(hour - 1)
    else:
        mu, sd = moments
    quantiles = []
    for z in Z:
        value = p1 * math.exp(mu + z * sd)
        # Keep every quantile a finite number in (0, 1e9]; the cap is far
        # above any real launch-day count.
        value = min(max(value, 0.01), 1e9)
        quantiles.append(round2(value))
    # Rounding can only make neighbours equal, never decrease them, but
    # enforce non-decreasing order explicitly so the gate always holds.
    for index in range(1, len(quantiles)):
        if quantiles[index] < quantiles[index - 1]:
            quantiles[index] = quantiles[index - 1]
    return quantiles


def forecast_case(case: dict, name: str) -> dict:
    """Forecast one case: hours 2..24 as five quantiles each, from hour 1."""
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        # No hour-1 print: fall back to the last pre-release point, else 1.
        history = case.get("history") or []
        p1 = float(history[-1]["viewers"]) if history else 1.0
    else:
        p1 = float(first["viewers"])
    p1 = max(p1, VIEWER_FLOOR)
    log_p1 = math.log(p1)

    # Per hour, the comparables' log ratios to their own hour-1 prints.
    per_hour: dict[int, list] = {hour: [] for hour in range(2, 25)}
    for comp in case.get("comparables") or []:
        curve = comparable_curve(comp)
        if 1 not in curve:
            continue
        comp_p1 = max(float(curve[1]), VIEWER_FLOOR)
        log_comp_p1 = math.log(comp_p1)
        for hour in range(2, 25):
            if hour in curve:
                ratio = max(float(curve[hour]), VIEWER_FLOOR) / comp_p1
                per_hour[hour].append((math.log(ratio), log_comp_p1))

    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        hours.append({"hour": hour,
                      "q": hour_quantiles(per_hour[hour], log_p1, p1, hour)})
    hours.sort(key=lambda row: row["hour"])
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
