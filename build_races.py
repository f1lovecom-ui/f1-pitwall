#!/usr/bin/env python3
"""
Build compact replay files for the Pit Wall app from OpenF1's free historical API.

Formula 1's own timing archive refuses requests from cloud machines like GitHub Actions,
so this reads the same data through OpenF1 instead (available from 2023 onwards).
Retirement reasons (accident, engine, ...) come from the Jolpica/Ergast results API.

For every finished Race and Sprint it writes one JSON file with car positions on a
fixed 4 Hz grid, laps with tyre compounds, pit stops, race control messages, timing
tower positions and gaps, and the classification. Existing files are skipped.

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
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

API = "https://api.openf1.org/v1/"
HZ = 4
OUT = Path("data")
COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
SESSION_ORDER = ("Sprint", "Race")
UA = {"User-Agent": "pitwall-builder (github.com)"}


def log(*a):
    print(*a, flush=True)


class ApiError(Exception):
    pass


class OpenF1:
    """Polite client: stays under ~3 requests/second and 25/minute, retries on errors."""

    def __init__(self, per_sec=3, per_min=25):
        self.per_sec, self.per_min, self.hist = per_sec, per_min, []
        self.http = requests.Session()
        self.http.headers.update(UA)

    def _throttle(self):
        while True:
            now = time.time()
            self.hist = [t for t in self.hist if now - t < 60]
            waits = []
            if len(self.hist) >= self.per_min:
                waits.append(self.hist[0] + 60 - now)
            recent = [t for t in self.hist if now - t < 1]
            if len(recent) >= self.per_sec:
                waits.append(recent[0] + 1 - now)
            if not waits or max(waits) <= 0:
                self.hist.append(now)
                return
            time.sleep(max(waits) + 0.05)

    def get(self, path, query=""):
        url = API + path + ("?" + query if query else "")
        for attempt in range(6):
            self._throttle()
            try:
                r = self.http.get(url, timeout=90)
            except requests.RequestException as e:
                log(f"    network problem ({e.__class__.__name__}), retrying")
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.replace(".", "", 1).isdigit() else 20)
                continue
            if r.status_code == 404:
                return []
            if r.status_code in (401, 403):
                raise ApiError(f"OpenF1 refused access ({r.status_code}). This can happen while a session is live; try again later.")
            if r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            if not r.ok:
                raise ApiError(f"OpenF1 returned {r.status_code} for {path}")
            data = r.json()
            return data if isinstance(data, list) else []
        raise ApiError(f"OpenF1 kept failing for {path}")


def parse_dt(s):
    if not s:
        return None
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def num(v, nd=3):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return round(float(v), nd)
    return None


def gapval(v):
    if isinstance(v, list):
        v = next((x for x in reversed(v) if x is not None), None)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(float(v), 3)
    if isinstance(v, str) and v.strip():
        m = re.fullmatch(r"\+?(\d+(?:\.\d+)?)", v.strip())
        return round(float(m.group(1)), 3) if m else v.strip()
    return None


def rotator(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return lambda x, y: (x * c - y * s, x * s + y * c)


def circuit_rotation(circuit_key, year):
    try:
        r = requests.get(f"https://api.multiviewer.app/api/v1/circuits/{circuit_key}/{year}", headers=UA, timeout=30)
        if r.ok:
            return float(r.json().get("rotation") or 0)
    except Exception:
        pass
    return 0.0


def retirement_reasons(year, rnd, sprint):
    """Driver number -> status text such as 'Accident' or 'Engine' (Jolpica/Ergast)."""
    kind = "sprint" if sprint else "results"
    try:
        r = requests.get(f"https://api.jolpi.ca/ergast/f1/{year}/{rnd}/{kind}.json", headers=UA, timeout=30)
        races = r.json()["MRData"]["RaceTable"]["Races"] if r.ok else []
        rows = races[0].get("SprintResults" if sprint else "Results", []) if races else []
        return {int(x["number"]): x.get("status", "") for x in rows}
    except Exception:
        return {}


def fetch_location(api, sk, n, a, b, depth=0):
    q = f"session_key={sk}&driver_number={n}&date>={a.isoformat()}&date<{b.isoformat()}"
    try:
        return api.get("location", q)
    except ApiError:
        if depth >= 2:
            raise
        m = a + (b - a) / 2
        return fetch_location(api, sk, n, a, m, depth + 1) + fetch_location(api, sk, n, m, b, depth + 1)


def build_session(api, sess, meeting, rnd):
    sk, year = sess["session_key"], sess["year"]
    t0 = parse_dt(sess["date_start"])
    rel = lambda s: (parse_dt(s) - t0).total_seconds() if s else None
    k = f"session_key={sk}"

    drivers_raw = api.get("drivers", k)
    laps_raw = api.get("laps", k)
    if not laps_raw:
        raise ApiError("no lap data")
    position = api.get("position", k)
    intervals = api.get("intervals", k)
    pits_raw = api.get("pit", k)
    stints = api.get("stints", k)
    rc_raw = api.get("race_control", k)
    res_raw = api.get("session_result", k)

    # --- drivers
    drivers, seen = [], set()
    for d in drivers_raw:
        n = d.get("driver_number")
        if n is None or n in seen:
            continue
        seen.add(n)
        col = (d.get("team_colour") or "").strip().lstrip("#")
        drivers.append({"n": n, "code": d.get("name_acronym") or str(n), "name": d.get("full_name") or "",
                        "team": d.get("team_name") or "", "color": "#" + col if col else "#9AA6BA"})

    # --- laps per driver, with estimated lap-1 start and tyre compound
    by_drv = {}
    for l in laps_raw:
        by_drv.setdefault(l["driver_number"], []).append(l)
    pit_laps = {}
    for p in pits_raw:
        pit_laps.setdefault(p["driver_number"], set()).add(p.get("lap_number"))
    stint_by = {}
    for s in stints:
        stint_by.setdefault(s["driver_number"], []).append(s)

    durs = sorted(l["lap_duration"] for l in laps_raw if num(l.get("lap_duration")) and l.get("lap_number", 0) > 1)
    typical = durs[len(durs) // 2] if durs else 95.0

    lap_rows, lap1_starts, ends = {}, [], []
    for n, ls in by_drv.items():
        ls.sort(key=lambda l: l["lap_number"])
        starts = {l["lap_number"]: rel(l.get("date_start")) for l in ls}
        if starts.get(1) is None and starts.get(2) is not None:
            l1 = next((l for l in ls if l["lap_number"] == 1), None)
            d1 = num(l1.get("lap_duration")) if l1 else None
            starts[1] = starts[2] - (d1 if d1 else typical * 1.15)
        rows = []
        for l in ls:
            v, st, dur = l["lap_number"], starts.get(l["lap_number"]), num(l.get("lap_duration"))
            if st is None:
                continue
            cmp_ = None
            for s in stint_by.get(n, []):
                if (s.get("lap_start") or 0) <= v <= (s.get("lap_end") or 999):
                    c = (s.get("compound") or "").upper()
                    cmp_ = c if c in COMPOUNDS else None
            rows.append([v, round(st, 3), dur, 1 if l.get("is_pit_out_lap") else 0, 1 if v in pit_laps.get(n, ()) else 0, cmp_])
            ends.append(st + (dur or typical))
        if 1 in starts and starts[1] is not None:
            lap1_starts.append(starts[1])
        lap_rows[str(n)] = rows
    lights = min(lap1_starts) if lap1_starts else min(r[1] for rs in lap_rows.values() for r in rs)
    race_end = max(ends)
    frame0 = lights - 300
    n_frames = int((race_end + 180 - frame0) * HZ) + 1
    grid = frame0 + np.arange(n_frames) / HZ

    # --- car positions, one driver at a time
    circuit_key = sess.get("circuit_key") or meeting.get("circuit_key")
    rotation = circuit_rotation(circuit_key, year) if circuit_key else 0.0
    rot = rotator(rotation)
    a, b = t0 + timedelta(seconds=frame0), t0 + timedelta(seconds=race_end + 180)
    pos, last_move, raw_xy = {}, {}, {}
    for d in drivers:
        n = d["n"]
        rows = fetch_location(api, sk, n, a, b)
        pts = [(rel(r["date"]), r["x"], r["y"]) for r in rows if (r.get("x") or r.get("y"))]
        if len(pts) < 10:
            continue
        pts.sort()
        t = np.array([p[0] for p in pts], float)
        t, uniq = np.unique(t, return_index=True)
        x = np.array([p[1] for p in pts], float)[uniq]
        y = np.array([p[2] for p in pts], float)[uniq]
        xi, yi = np.interp(grid, t, x), np.interp(grid, t, y)
        idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        valid = (grid >= t[0]) & (grid <= t[-1]) & ((t[idx] - t[idx - 1]) <= 3.0)
        rx, ry = rot(xi, yi)
        rx, ry = np.round(rx / 10).astype(int), np.round(ry / 10).astype(int)
        raw_xy[n] = (t, x, y)
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
            pos[str(n)] = segs
        step = np.hypot(np.diff(rx), np.diff(ry))
        moving = np.where(valid[1:] & valid[:-1] & (step >= 2))[0]
        if len(moving):
            last_move[n] = round(float(grid[moving[-1] + 1]), 1)

    # --- timing tower: positions and gaps over time
    timing = {}
    for p in position:
        t = rel(p.get("date"))
        if t is not None and p.get("position"):
            timing.setdefault(str(p["driver_number"]), []).append([round(t, 1), int(p["position"]), None, None])
    for iv in intervals:
        t = rel(iv.get("date"))
        g, i_ = gapval(iv.get("gap_to_leader")), gapval(iv.get("interval"))
        if t is not None and (g is not None or i_ is not None):
            timing.setdefault(str(iv["driver_number"]), []).append([round(t, 1), None, g, i_])
    for v in timing.values():
        v.sort(key=lambda r: r[0])

    # --- race control and pits
    rc = [[round(rel(m["date"]), 1), m.get("lap_number"), m.get("category"), m.get("flag"), m.get("scope"), m.get("message")]
          for m in rc_raw if m.get("date")]
    rc.sort(key=lambda m: m[0])
    pits = []
    for p in pits_raw:
        lane = num(p.get("lane_duration")) or num(p.get("pit_duration"))
        if p.get("date") and lane:
            pits.append([round(rel(p["date"]), 1), p["driver_number"], p.get("lap_number"), lane])
    pits.sort()

    # --- classification, with retirement reasons where available
    reasons = retirement_reasons(year, rnd, sess.get("session_name") == "Sprint")
    laps_done = {int(k2): max((r[0] for r in v), default=0) for k2, v in lap_rows.items()}
    results = []
    for r in res_raw:
        n = r.get("driver_number")
        if n is None:
            continue
        p = r.get("position")
        cls = "D" if r.get("dsq") else "W" if r.get("dns") else "R" if r.get("dnf") else (str(p) if p else "N")
        g = gapval(r.get("gap_to_leader"))
        status = reasons.get(n) or (g if isinstance(g, str) else "")
        results.append({
            "n": n, "pos": p, "cls": cls, "status": status,
            "pts": num(r.get("points")),
            "time": gapval(r.get("duration")) if p == 1 else (g if isinstance(g, float) else None),
            "laps": r.get("number_of_laps") or laps_done.get(n, 0),
            "grid": None,
            "outT": last_move.get(n) if cls in ("R", "N") else None,
        })

    # --- track outline from the winner's fastest clean lap
    track = {"x": [], "y": []}
    order = [r["n"] for r in sorted(results, key=lambda r: r["pos"] or 99)] or list(raw_xy)
    for n in order[:3]:
        cand = [r for r in lap_rows.get(str(n), []) if r[0] >= 2 and r[2] and not r[3] and not r[4]]
        if n not in raw_xy or not cand:
            continue
        best = min(cand, key=lambda r: r[2])
        t, x, y = raw_xy[n]
        m = (t >= best[1]) & (t <= best[1] + best[2])
        if m.sum() < 50:
            continue
        tx, ty = rot(x[m], y[m])
        pts = []
        for p2 in zip(np.round(tx / 10).astype(int).tolist(), np.round(ty / 10).astype(int).tolist()):
            if not pts or pts[-1] != p2:
                pts.append(p2)
        track = {"x": [p2[0] for p2 in pts], "y": [p2[1] for p2 in pts]}
        break

    return {
        "v": 1,
        "meta": {
            "year": year, "round": rnd, "event": meeting.get("meeting_name"),
            "country": meeting.get("country_name"), "location": meeting.get("location"),
            "session": sess.get("session_name"), "date": t0.strftime("%Y-%m-%d"),
            "lightsOut": round(lights, 1), "raceEnd": round(race_end, 1),
            "hz": HZ, "frame0": round(frame0, 3), "frames": n_frames,
            "totalLaps": max((r[0] for rs in lap_rows.values() for r in rs), default=0),
            "rotation": rotation, "source": "openf1",
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
        e = events.setdefault((meta["year"], meta["round"]), {
            "round": meta["round"], "name": meta["event"], "location": meta["location"],
            "country": meta["country"], "date": meta["date"], "sessions": [],
        })
        e["sessions"].append({"name": meta["session"], "file": f.relative_to(OUT).as_posix()})
        if meta["session"] == "Race":
            e["date"] = meta["date"]
    seasons = {}
    for (year, _), e in sorted(events.items()):
        e["sessions"].sort(key=lambda s: SESSION_ORDER.index(s["name"]) if s["name"] in SESSION_ORDER else 9)
        seasons.setdefault(str(year), []).append(e)
    OUT.mkdir(exist_ok=True)
    (OUT / "index.json").write_text(json.dumps({
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "seasons": seasons,
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
    skipped = [y for y in years if y < 2023]
    if skipped:
        log(f"Skipping {skipped}: OpenF1 has data from 2023 onwards only.")
    years = [y for y in years if y >= 2023]

    api = OpenF1()
    deadline = time.time() + args.budget_min * 60
    built = failed = 0
    first_error = None
    stop = False

    for year in years:
        if stop:
            break
        try:
            meetings = sorted(api.get("meetings", f"year={year}"), key=lambda m: m["date_start"])
            sessions = api.get("sessions", f"year={year}&session_type=Race")
        except ApiError as e:
            log(f"{year}: couldn't load the calendar: {e}")
            continue
        races = [m for m in meetings if "test" not in (m.get("meeting_name") or "").lower()]
        rounds = {m["meeting_key"]: i + 1 for i, m in enumerate(races)}
        by_key = {m["meeting_key"]: m for m in races}
        for sess in sorted(sessions, key=lambda s: s["date_start"]):
            mk, name = sess.get("meeting_key"), sess.get("session_name")
            if mk not in rounds or name not in SESSION_ORDER:
                continue
            end = parse_dt(sess.get("date_end")) or parse_dt(sess["date_start"]) + timedelta(hours=3)
            if end > now - timedelta(hours=4):
                continue  # not finished yet
            rnd = rounds[mk]
            path = OUT / str(year) / f"{rnd:02d}_{name.lower()}.json"
            if path.exists() and not args.force:
                continue
            if time.time() > deadline:
                log("Time budget reached; the next run will continue from here.")
                stop = True
                break
            log(f"{year} round {rnd}: {by_key[mk].get('meeting_name')}, {name}")
            started = time.time()
            try:
                data = build_session(api, sess, by_key[mk], rnd)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, separators=(",", ":")))
                built += 1
                log(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {time.time() - started:.0f} s)")
            except Exception as e:
                failed += 1
                log(f"  failed: {type(e).__name__}: {e}")
                if first_error is None:
                    first_error = traceback.format_exc()
                    log(first_error)

    log(f"Done: {built} built, {failed} failed.")
    if failed and not built:
        log("Every race failed to build, so nothing will be published. See the first error above.")
        sys.exit(1)
    write_index()


if __name__ == "__main__":
    main()
