#!/usr/bin/env python3
"""
Build compact replay files for the Pit Wall app.

For every finished Race and Sprint, this downloads the session once through FastF1
(which reads Formula 1's own timing archive) and writes one JSON file with:
  - car positions on a fixed 4 Hz grid (delta-encoded, in metres, rotated to the official map orientation)
  - laps with tyre compounds, pit stops, race control messages
  - live-timing positions and gaps over time
  - the classification, including why each retired car retired

Files that already exist are skipped, so each run only builds new races.

Usage:
  python build_races.py                  # last season and this season
  python build_races.py --years 2023,2024
  python build_races.py --force          # rebuild files that already exist
"""
import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1

HZ = 4                      # position samples per second in the output
OUT = Path("data")
CACHE = Path(".fastf1-cache")
SESSIONS = ("Sprint", "Race")
COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
OUT_CODES = {"R", "N", "E"}  # retired, not classified, excluded


def log(*a):
    print(*a, flush=True)


def clean(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s if s and s.lower() != "nan" else None


def clean_int(v):
    try:
        if v is None or pd.isna(v):
            return None
        return int(float(v))
    except (TypeError, ValueError):
        return None


def clean_num(v):
    try:
        if v is None or pd.isna(v):
            return None
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None


def secs(td):
    try:
        if td is None or pd.isna(td):
            return None
        return pd.Timedelta(td).total_seconds()
    except (TypeError, ValueError):
        return None


def r3(v):
    return None if v is None else round(float(v), 3)


def parse_gap(v):
    s = clean(v)
    if s is None:
        return None
    m = re.fullmatch(r"\+?(\d+(?:\.\d+)?)", s)
    return round(float(m.group(1)), 3) if m else s


def rotator(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return lambda x, y: (x * c - y * s, x * s + y * c)


def track_outline(session, laps, res, rot):
    lap = None
    try:
        winner = res.sort_values("Position").iloc[0]["DriverNumber"]
        lap = laps[laps["DriverNumber"] == str(winner)].pick_fastest()
    except Exception:
        lap = None
    if lap is None or (hasattr(lap, "empty") and lap.empty):
        lap = laps.pick_fastest()
    pdata = lap.get_pos_data()
    tx, ty = rot(pdata["X"].to_numpy(float), pdata["Y"].to_numpy(float))
    tx = np.round(tx / 10).astype(int).tolist()
    ty = np.round(ty / 10).astype(int).tolist()
    pts = []
    for a, b in zip(tx, ty):
        if not pts or pts[-1] != (a, b):
            pts.append((a, b))
    return {"x": [p[0] for p in pts], "y": [p[1] for p in pts]}


def build_session(year, rnd, ev, sname):
    session = fastf1.get_session(year, rnd, sname)
    session.load(laps=True, telemetry=True, weather=False, messages=True)
    laps = session.laps
    if laps is None or laps.empty:
        raise RuntimeError("no lap data")
    res = session.results
    t0 = session.t0_date

    try:
        rotation = float(session.get_circuit_info().rotation)
    except Exception:
        rotation = 0.0
    rot = rotator(rotation)

    # --- timeline bounds: 5 minutes before lights out to 3 minutes after the last lap
    l1 = laps[laps["LapNumber"] == 1]["LapStartTime"].dropna()
    lights = secs(l1.min()) if len(l1) else secs(laps["LapStartTime"].dropna().min())
    race_end = secs(laps["Time"].dropna().max())
    frame0 = max(0.0, lights - 300)
    frame_end = race_end + 180
    n_frames = int((frame_end - frame0) * HZ) + 1
    grid = frame0 + np.arange(n_frames) / HZ

    # --- drivers
    drivers = []
    for _, r in res.iterrows():
        n = clean_int(r.get("DriverNumber"))
        if n is None:
            continue
        color = clean(r.get("TeamColor"))
        drivers.append({
            "n": n,
            "code": clean(r.get("Abbreviation")) or str(n),
            "name": clean(r.get("FullName")) or "",
            "team": clean(r.get("TeamName")) or "",
            "color": ("#" + color.lstrip("#")) if color else "#9AA6BA",
        })

    # --- positions, resampled to a fixed grid and delta-encoded
    pos, last_move = {}, {}
    for drv, df in (session.pos_data or {}).items():
        d = df
        if "Status" in d:
            d = d[d["Status"] == "OnTrack"]
        d = d[~((d["X"] == 0) & (d["Y"] == 0))]
        if len(d) < 10:
            continue
        t = d["SessionTime"].dt.total_seconds().to_numpy()
        x = d["X"].to_numpy(float)
        y = d["Y"].to_numpy(float)
        t, uniq = np.unique(t, return_index=True)
        x, y = x[uniq], y[uniq]
        xi = np.interp(grid, t, x)
        yi = np.interp(grid, t, y)
        idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        valid = (grid >= t[0]) & (grid <= t[-1]) & ((t[idx] - t[idx - 1]) <= 3.0)
        rx, ry = rot(xi, yi)
        rx = np.round(rx / 10).astype(int)   # 1/10 m -> m
        ry = np.round(ry / 10).astype(int)
        segs, i = [], 0
        while i < n_frames:
            if not valid[i]:
                i += 1
                continue
            j = i
            while j < n_frames and valid[j]:
                j += 1
            sx, sy = rx[i:j], ry[i:j]
            segs.append([i, [int(sx[0])] + np.diff(sx).tolist(), [int(sy[0])] + np.diff(sy).tolist()])
            i = j
        if segs:
            pos[str(int(drv))] = segs
        # last moment the car was really moving (>= ~8 m/s), used to time retirements
        step = np.hypot(np.diff(rx), np.diff(ry))
        moving = np.where(valid[1:] & valid[:-1] & (step >= 2))[0]
        if len(moving):
            last_move[int(drv)] = round(float(grid[moving[-1] + 1]), 1)

    # --- laps (with tyre compound) and pit stops
    lap_rows, pits = {}, []
    for num, g in laps.groupby("DriverNumber"):
        g = g.sort_values("LapNumber")
        rows = list(g.itertuples(index=False))
        out = []
        for k, l in enumerate(rows):
            st = secs(l.LapStartTime)
            if st is not None:
                cmp_ = clean(getattr(l, "Compound", None))
                out.append([
                    int(l.LapNumber), round(st, 3), r3(secs(l.LapTime)),
                    0 if pd.isna(l.PitOutTime) else 1,
                    0 if pd.isna(l.PitInTime) else 1,
                    cmp_.upper() if cmp_ and cmp_.upper() in COMPOUNDS else None,
                ])
            if not pd.isna(l.PitInTime) and k + 1 < len(rows) and not pd.isna(rows[k + 1].PitOutTime):
                lane = secs(rows[k + 1].PitOutTime - l.PitInTime)
                pits.append([round(secs(l.PitInTime), 1), int(num), int(l.LapNumber), r3(lane)])
        lap_rows[str(int(num))] = out
    pits.sort()

    # --- timing tower data: positions and gaps over time
    timing = {}
    try:
        _, stream = fastf1.api.timing_data(session.api_path)
        stream = stream.sort_values("Time")
        for drv, g in stream.groupby("Driver"):
            rows, last = [], None
            for r in g.itertuples(index=False):
                t = secs(r.Time)
                if t is None:
                    continue
                row = [round(t, 1), clean_int(r.Position), parse_gap(r.GapToLeader), parse_gap(r.IntervalToPositionAhead)]
                if last is not None and row[1:] == last[1:]:
                    continue
                rows.append(row)
                last = row
            timing[str(int(drv))] = rows
    except Exception as e:
        log("  timing stream unavailable, using lap-end positions:", e)
    if not timing:
        for _, l in laps.iterrows():
            t, p = secs(l["Time"]), clean_int(l.get("Position"))
            if t is not None and p is not None:
                timing.setdefault(str(int(l["DriverNumber"])), []).append([round(t, 1), p, None, None])
        for v in timing.values():
            v.sort()

    # --- race control
    rc = []
    msgs = session.race_control_messages
    if msgs is not None and not msgs.empty:
        for r in msgs.itertuples(index=False):
            t = secs(pd.Timestamp(r.Time) - pd.Timestamp(t0))
            if t is None:
                continue
            rc.append([round(t, 1), clean_int(getattr(r, "Lap", None)), clean(r.Category), clean(r.Flag),
                       clean(r.Scope), clean(r.Message)])
        rc.sort(key=lambda m: m[0])

    # --- classification
    laps_done = laps.groupby("DriverNumber")["LapNumber"].max()
    results = []
    for _, r in res.iterrows():
        n = clean_int(r.get("DriverNumber"))
        if n is None:
            continue
        cls = clean(r.get("ClassifiedPosition"))
        results.append({
            "n": n,
            "pos": clean_int(r.get("Position")),
            "cls": cls,
            "status": clean(r.get("Status")) or "",
            "pts": clean_num(r.get("Points")),
            "time": r3(secs(r.get("Time"))),
            "laps": clean_int(laps_done.get(str(n))) or 0,
            "grid": clean_int(r.get("GridPosition")),
            "outT": last_move.get(n) if cls in OUT_CODES else None,
        })

    try:
        track = track_outline(session, laps, res, rot)
    except Exception as e:
        log("  track outline unavailable:", e)
        track = {"x": [], "y": []}

    return {
        "v": 1,
        "meta": {
            "year": year, "round": rnd, "event": clean(ev.get("EventName")),
            "country": clean(ev.get("Country")), "location": clean(ev.get("Location")),
            "session": sname, "date": pd.Timestamp(t0).strftime("%Y-%m-%d"),
            "lightsOut": round(lights, 1), "raceEnd": round(race_end, 1),
            "hz": HZ, "frame0": round(frame0, 3), "frames": n_frames,
            "totalLaps": int(laps["LapNumber"].max()), "rotation": rotation,
            "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "track": track, "drivers": drivers, "pos": pos, "laps": lap_rows,
        "timing": timing, "rc": rc, "pits": pits, "results": results,
    }


def write_index():
    events = {}
    for f in sorted(OUT.glob("*/*.json")):
        try:
            meta = json.loads(f.read_text())["meta"]
        except Exception:
            continue
        key = (meta["year"], meta["round"])
        e = events.setdefault(key, {
            "round": meta["round"], "name": meta["event"], "location": meta["location"],
            "country": meta["country"], "date": meta["date"], "sessions": [],
        })
        e["sessions"].append({"name": meta["session"], "file": f.relative_to(OUT).as_posix()})
        if meta["session"] == "Race":
            e["date"] = meta["date"]
    seasons = {}
    for (year, _), e in sorted(events.items()):
        e["sessions"].sort(key=lambda s: SESSIONS.index(s["name"]) if s["name"] in SESSIONS else 9)
        seasons.setdefault(str(year), []).append(e)
    OUT.mkdir(exist_ok=True)
    (OUT / "index.json").write_text(json.dumps({
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seasons": seasons,
    }, separators=(",", ":")))
    log(f"index.json: {sum(len(v) for v in seasons.values())} race weekends")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default=os.environ.get("YEARS", ""))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--budget-min", type=float, default=float(os.environ.get("BUDGET_MIN", "300")))
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    years = [int(y) for y in re.split(r"[,\s]+", args.years.strip()) if y] or [now.year - 1, now.year]
    CACHE.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    fastf1.set_log_level("WARNING")
    deadline = time.time() + args.budget_min * 60
    built = failed = 0
    stop = False

    for year in years:
        if stop:
            break
        try:
            sched = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            log(f"{year}: couldn't load the schedule: {e}")
            continue
        for _, ev in sched.iterrows():
            if stop:
                break
            rnd = int(ev["RoundNumber"])
            for i in range(1, 6):
                name, date = ev.get(f"Session{i}"), ev.get(f"Session{i}DateUtc")
                if name not in SESSIONS or date is None or pd.isna(date):
                    continue
                ts = pd.Timestamp(date)
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
                if ts > pd.Timestamp(now) - timedelta(hours=4):
                    continue  # not finished yet
                path = OUT / str(year) / f"{rnd:02d}_{name.lower()}.json"
                if path.exists() and not args.force:
                    continue
                if time.time() > deadline:
                    log("Time budget reached; the next run will continue from here.")
                    stop = True
                    break
                log(f"{year} round {rnd}: {ev['EventName']}, {name}")
                started = time.time()
                try:
                    data = build_session(year, rnd, ev, name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data, separators=(",", ":")))
                    built += 1
                    log(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {time.time() - started:.0f} s)")
                except Exception as e:
                    failed += 1
                    log(f"  failed: {e}")

    write_index()
    log(f"Done: {built} built, {failed} failed.")


if __name__ == "__main__":
    main()
