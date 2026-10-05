#!/usr/bin/env python3
"""
Build compact replay files for the Pit Wall app from OpenF1's free historical API.

Formula 1's own timing archive refuses requests from cloud machines like GitHub Actions,
so this reads the same data through OpenF1 instead (available from 2023 onwards).
Retirement reasons (accident, engine, ...) come from the Jolpica/Ergast results API.

For every finished Race, Sprint, Qualifying and Sprint Qualifying/Shootout it writes one
JSON file with car positions on a fixed 4 Hz grid, laps with tyre compounds, pit stops,
race control messages, timing tower positions and gaps, weather, team radio, the
classification and (for races) championship standings. Car telemetry (speed, gear,
throttle, brake, DRS) goes in one small file per driver next to it.

Existing files are skipped; files from an older version of this script are upgraded
by fetching only the missing parts.

Usage:
  python pitwall_build.py                  # last season and this season
  python pitwall_build.py --years 2023,2024
  python pitwall_build.py --force          # rebuild files that already exist
  TELEMETRY=all python pitwall_build.py    # telemetry for every season (default: this and last)
  TELEMETRY=0 python pitwall_build.py      # no telemetry at all (much smaller files)
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
SESSION_ORDER = ("Sprint Shootout", "Sprint Qualifying", "Sprint", "Qualifying", "Race")
RACE_TYPES = ("Sprint", "Race")
FILE_V = 3                 # bump when new data is added to the files
GEO_URL = "https://raw.githubusercontent.com/bacinger/f1-circuits/master/f1-circuits.geojson"
TEL_HZ = 2                 # telemetry samples per second
# Telemetry: "recent" (default) = this season and last, "all", or "0"/"none" to skip.
# It roughly doubles the data size, and GitHub Pages sites are limited to 1 GB.
TELEMETRY = os.environ.get("TELEMETRY", "recent").strip().lower() or "recent"


def want_telemetry(year):
    if TELEMETRY in ("0", "none", "off", "false"):
        return False
    if TELEMETRY == "all":
        return True
    return year >= datetime.now(timezone.utc).year - 1
UA = {"User-Agent": "pitwall-builder (github.com)"}


def log(*a):
    print(*a, flush=True)


class ApiError(Exception):
    pass


class NoData(Exception):
    """The session is on the calendar but has no data (for example, it was cancelled)."""


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


_circuit_cache = {}


def circuit_info(circuit_key, year):
    """Official map rotation and corner positions (MultiViewer), in the same raw units as car positions."""
    key = (circuit_key, year)
    if key not in _circuit_cache:
        info = {"rotation": 0.0, "corners": []}
        try:
            r = requests.get(f"https://api.multiviewer.app/api/v1/circuits/{circuit_key}/{year}", headers=UA, timeout=30)
            if r.ok:
                j = r.json()
                info["rotation"] = float(j.get("rotation") or 0)
                for c in j.get("corners") or []:
                    tp = c.get("trackPosition") or {}
                    if tp.get("x") is not None and tp.get("y") is not None:
                        info["corners"].append((c.get("number"), c.get("letter") or "", float(tp["x"]), float(tp["y"])))
        except Exception:
            pass
        _circuit_cache[key] = info
    return _circuit_cache[key]


def circuit_rotation(circuit_key, year):
    return circuit_info(circuit_key, year)["rotation"]


def rotated_corners(info, rotation):
    rot = rotator(rotation)
    out = []
    for numb, letter, x, y in info["corners"]:
        rx, ry = rot(x, y)
        out.append([numb, letter, int(round(rx / 10)), int(round(ry / 10))])
    return out


# ---------- placing the track on a real map ----------
_geo_features = None
R_EARTH = 6378137.0


def geo_features():
    """Real-world circuit outlines (bacinger/f1-circuits, MIT licence), downloaded once per run."""
    global _geo_features
    if _geo_features is None:
        _geo_features = []
        try:
            r = requests.get(GEO_URL, headers=UA, timeout=60)
            if r.ok:
                for f in r.json().get("features", []):
                    g = f.get("geometry") or {}
                    lines = [g["coordinates"]] if g.get("type") == "LineString" else g.get("coordinates", []) if g.get("type") == "MultiLineString" else []
                    pts = [pt for line in lines for pt in line]
                    if len(pts) >= 20:
                        lon = np.array([p[0] for p in pts], float)
                        lat = np.array([p[1] for p in pts], float)
                        mx = R_EARTH * np.radians(lon)
                        my = R_EARTH * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
                        props = f.get("properties") or {}
                        _geo_features.append({"id": props.get("id") or props.get("Name"), "name": props.get("Name"),
                                              "xy": np.stack([mx, my], 1)})
        except Exception as e:
            log("  couldn't download real-world circuit outlines:", e)
    return _geo_features


def resample_loop(P, n=256):
    P = np.asarray(P, float)
    Q = np.vstack([P, P[:1]])
    seg = np.hypot(*np.diff(Q, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(t, cum, Q[:, 0]), np.interp(t, cum, Q[:, 1])], 1)


def similarity(P, Q, reflect):
    """Best scale + rotation (or mirror) + shift taking P onto Q (Umeyama)."""
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(Y.T @ X / len(P))
    D = np.eye(2)
    det = np.linalg.det(U @ Vt)
    D[1, 1] = (-1 if det > 0 else 1) if reflect else (1 if det > 0 else -1)
    Rm = U @ D @ Vt
    s_ = np.trace(np.diag(S) @ D) / ((X ** 2).sum() / len(P))
    M = s_ * Rm
    t = mq - M @ mp
    rms = np.sqrt(((Q - (P @ M.T + t)) ** 2).sum(1).mean())
    return M, t, rms / np.sqrt((Y ** 2).sum(1).mean())


def fit_geo(track):
    """Match the track outline to a real circuit and return the map placement, or None."""
    xs, ys = track.get("x") or [], track.get("y") or []
    feats = geo_features()
    if len(xs) < 50 or not feats:
        return None
    P = resample_loop(np.stack([xs, ys], 1))
    best = None
    for f in feats:
        Q0 = resample_loop(f["xy"])
        for Qd in (Q0, Q0[::-1]):
            for k in range(0, len(Qd), 4):
                Qs = np.roll(Qd, -k, axis=0)
                for refl in (False, True):
                    M, t, err = similarity(P, Qs, refl)
                    if best is None or err < best[0]:
                        best = (err, f, Qd, k, refl)
    _, f, Qd, k0, refl = best
    err, bestM, bestT = None, None, None
    for k in range(k0 - 4, k0 + 5):   # refine around the best match
        M, t, e = similarity(P, np.roll(Qd, -(k % len(Qd)), axis=0), refl)
        if err is None or e < err:
            err, bestM, bestT = e, M, t
    if err > 0.08:
        return None   # no circuit matches well enough (new or changed layout)
    cx, cy = bestM @ P.mean(0) + bestT
    lon = float(np.degrees(cx / R_EARTH))
    lat = float(np.degrees(2 * np.arctan(np.exp(cy / R_EARTH)) - np.pi / 2))
    return {"a": float(bestM[0, 0]), "b": float(bestM[0, 1]), "c": float(bestM[1, 0]), "d": float(bestM[1, 1]),
            "tx": float(bestT[0]), "ty": float(bestT[1]), "lat": round(lat, 5), "lon": round(lon, 5),
            "fit": round(float(err), 4), "circuit": f["name"]}


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


def fetch_range(api, endpoint, sk, n, a, b, depth=0):
    q = f"session_key={sk}&driver_number={n}&date>={a.isoformat()}&date<{b.isoformat()}"
    try:
        return api.get(endpoint, q)
    except ApiError:
        if depth >= 2:
            raise
        m = a + (b - a) / 2
        return fetch_range(api, endpoint, sk, n, a, m, depth + 1) + fetch_range(api, endpoint, sk, n, m, b, depth + 1)


def slug(name):
    return name.lower().replace(" ", "-")


def build_session(api, sess, meeting, rnd):
    sk, year = sess["session_key"], sess["year"]
    t0 = parse_dt(sess["date_start"])
    rel = lambda s: (parse_dt(s) - t0).total_seconds() if s else None
    k = f"session_key={sk}"

    is_race = sess.get("session_name") in RACE_TYPES
    laps_raw = api.get("laps", k)
    if not laps_raw:
        raise NoData()
    drivers_raw = api.get("drivers", k)
    position = api.get("position", k)
    intervals = api.get("intervals", k) if is_race else []
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
        if is_race and starts.get(1) is None and starts.get(2) is not None:
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
    race_end = max(ends)
    if is_race:
        lights = min(lap1_starts) if lap1_starts else min(r[1] for rs in lap_rows.values() for r in rs)
        frame0 = lights - 300
        tail = 180
    else:
        lights, frame0, tail = 0.0, 0.0, 60   # qualifying: the clock runs from the session start
    n_frames = int((race_end + tail - frame0) * HZ) + 1
    grid = frame0 + np.arange(n_frames) / HZ

    # --- car positions, one driver at a time
    circuit_key = sess.get("circuit_key") or meeting.get("circuit_key")
    cinfo = circuit_info(circuit_key, year) if circuit_key else {"rotation": 0.0, "corners": []}
    rotation = cinfo["rotation"]
    rot = rotator(rotation)
    a, b = t0 + timedelta(seconds=frame0), t0 + timedelta(seconds=race_end + tail)
    pos, last_move, raw_xy = {}, {}, {}
    for d in drivers:
        n = d["n"]
        rows = fetch_range(api, "location", sk, n, a, b)
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
    reasons = retirement_reasons(year, rnd, sess.get("session_name") == "Sprint") if is_race else {}
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
            "outT": last_move.get(n) if is_race and cls in ("R", "N") else None,
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
            "rotation": rotation, "source": "openf1", "kind": "race" if is_race else "qualifying",
            "start": sess["date_start"],
            "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "track": track, "drivers": drivers, "pos": pos, "laps": lap_rows,
        "timing": timing, "rc": rc, "pits": pits, "results": results,
        "stints": [[st["driver_number"], st.get("lap_start"), st.get("lap_end"),
                    (st.get("compound") or "").upper() if (st.get("compound") or "").upper() in COMPOUNDS else None,
                    st.get("tyre_age_at_start")] for st in stints if st.get("driver_number") is not None],
    }


def jolpica(path):
    try:
        r = requests.get("https://api.jolpi.ca/ergast/f1/" + path, headers=UA, timeout=30)
        time.sleep(0.3)
        return r.json()["MRData"] if r.ok else None
    except Exception:
        return None


def standings(year, rnd):
    """Driver and constructor standings after this round, with the previous round for comparison."""
    def lists(kind, r):
        if r < 1:
            return []
        m = jolpica(f"{year}/{r}/{kind}Standings.json")
        sl = (m or {}).get("StandingsTable", {}).get("StandingsLists", [])
        return sl[0].get(kind[0].upper() + kind[1:] + "Standings", []) if sl else []

    def ipos(x):
        return int(x["position"]) if str(x.get("position", "")).isdigit() else None

    cur, prev = lists("driver", rnd), lists("driver", rnd - 1)
    if not cur:
        return None
    pmap = {x["Driver"]["driverId"]: x for x in prev}
    drivers = []
    for x in cur:
        d = x["Driver"]
        p = pmap.get(d["driverId"])
        drivers.append([d.get("code") or d.get("familyName", "")[:3].upper(), f'{d.get("givenName", "")} {d.get("familyName", "")}'.strip(),
                        (x.get("Constructors") or [{}])[-1].get("name", ""), ipos(x), float(x.get("points", 0)), int(x.get("wins", 0)),
                        ipos(p) if p else None, float(p["points"]) if p else 0.0])
    ccur, cprev = lists("constructor", rnd), lists("constructor", rnd - 1)
    cmap = {x["Constructor"]["constructorId"]: x for x in cprev}
    teams = []
    for x in ccur:
        p = cmap.get(x["Constructor"]["constructorId"])
        teams.append([x["Constructor"].get("name", ""), ipos(x), float(x.get("points", 0)), int(x.get("wins", 0)),
                      ipos(p) if p else None, float(p["points"]) if p else 0.0])
    return {"drivers": drivers, "teams": teams}


def build_telemetry(api, sess, meta, drivers, tel_dir):
    """Speed, gear, throttle, brake and DRS per driver on a 2 Hz grid, one file per driver."""
    sk, t0 = sess["session_key"], parse_dt(sess["date_start"])
    frame0 = meta["frame0"]
    end = frame0 + (meta["frames"] - 1) / meta["hz"]
    n_tel = int((end - frame0) * TEL_HZ) + 1
    grid = frame0 + np.arange(n_tel) / TEL_HZ
    a, b = t0 + timedelta(seconds=frame0), t0 + timedelta(seconds=end)
    tel_dir.mkdir(parents=True, exist_ok=True)
    have = []
    for d in drivers:
        n = d["n"]
        rows = fetch_range(api, "car_data", sk, n, a, b)
        pts = sorted(((parse_dt(r["date"]) - t0).total_seconds(), r) for r in rows if r.get("date"))
        if len(pts) < 10:
            continue
        t = np.array([p[0] for p in pts], float)
        t, uniq = np.unique(t, return_index=True)
        col = lambda key: np.array([float(pts[i][1].get(key) or 0) for i in uniq])
        spd = np.interp(grid, t, col("speed"))
        thr = np.interp(grid, t, col("throttle"))
        prev = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
        gear, brk, drs = col("n_gear")[prev], col("brake")[prev], col("drs")[prev]
        idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        valid = (grid >= t[0]) & (grid <= t[-1]) & ((t[idx] - t[idx - 1]) <= 3.0)
        chans = [
            np.round(spd).astype(int),
            np.clip(gear, 0, 8).astype(int),
            (np.clip(np.round(thr / 5) * 5, 0, 100)).astype(int),
            (brk > 0).astype(int),
            np.where(drs >= 10, 2, np.where(drs == 8, 1, 0)).astype(int),
        ]
        segs, i = [], 0
        while i < n_tel:
            if not valid[i]:
                i += 1
                continue
            j = i
            while j < n_tel and valid[j]:
                j += 1
            seg = [i]
            for c in chans:
                part = c[i:j]
                seg.append([int(part[0])] + np.diff(part).tolist())
            segs.append(seg)
            i = j
        if segs:
            (tel_dir / f"{n}.json").write_text(json.dumps({"hz": TEL_HZ, "frame0": frame0, "frames": n_tel, "segs": segs}, separators=(",", ":")))
            have.append(n)
    return have


def add_extras(api, sess, data, rnd, tel_dir):
    """Everything added after file version 1. Only the parts a file is missing are fetched,
    so upgrading an older file is quick."""
    sk, t0 = sess["session_key"], parse_dt(sess["date_start"])
    rel = lambda s: round((parse_dt(s) - t0).total_seconds(), 1)
    k = f"session_key={sk}"
    meta = data["meta"]
    meta.setdefault("start", sess["date_start"])
    if "weather" not in data:
        data["weather"] = [[rel(r["date"]), num(r.get("air_temperature"), 1), num(r.get("track_temperature"), 1),
                            1 if r.get("rainfall") else 0, num(r.get("wind_speed"), 1), num(r.get("humidity"), 0)]
                           for r in api.get("weather", k) if r.get("date")]
    if "radio" not in data:
        data["radio"] = sorted([rel(r["date"]), r["driver_number"], r["recording_url"]]
                               for r in api.get("team_radio", k) if r.get("date") and r.get("recording_url"))
    if "stints" not in data:
        data["stints"] = [[st["driver_number"], st.get("lap_start"), st.get("lap_end"),
                           (st.get("compound") or "").upper() if (st.get("compound") or "").upper() in COMPOUNDS else None,
                           st.get("tyre_age_at_start")] for st in api.get("stints", k) if st.get("driver_number") is not None]
    if meta.get("session") == "Race" and "standings" not in data:
        data["standings"] = standings(meta["year"], rnd)
    if "tel" not in meta:
        meta["tel"] = build_telemetry(api, sess, meta, data["drivers"], tel_dir) if want_telemetry(meta["year"]) else []
    track = data.setdefault("track", {"x": [], "y": []})
    if "corners" not in track:
        ck = sess.get("circuit_key")
        track["corners"] = rotated_corners(circuit_info(ck, meta["year"]), meta.get("rotation", 0.0)) if ck else []
    if "geo" not in meta:
        meta["geo"] = fit_geo(track)
        g = meta["geo"]
        log(f"  placed on the map: {g['circuit']} (match error {g['fit']:.3f})" if g else "  no real-world map match for this layout")
    data["v"] = FILE_V
    return data


def file_version(path):
    try:
        with open(path) as f:
            m = re.match(r'\{"v":(\d+)', f.read(12))
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def official_rounds(year):
    """Race date -> official round number, from Jolpica/Ergast (cancelled races aren't listed there)."""
    try:
        r = requests.get(f"https://api.jolpi.ca/ergast/f1/{year}.json?limit=100", headers=UA, timeout=30)
        races = r.json()["MRData"]["RaceTable"]["Races"] if r.ok else []
        return {x["date"]: int(x["round"]) for x in races}
    except Exception:
        return {}


def round_for(meeting, race_sessions, official):
    """Match a meeting to its official round by its race date (allowing a day either side)."""
    if not official:
        return None
    for s in race_sessions:
        d = parse_dt(s["date_start"]).date()
        for off in (0, -1, 1):
            key = (d + timedelta(days=off)).isoformat()
            if key in official:
                return official[key]
    return None


def existing_files(year):
    """(event name, session) -> path, for race files already built this season."""
    found = {}
    for f in (OUT / str(year)).glob("*.json"):
        try:
            meta = json.loads(f.read_text())["meta"]
            found[(meta["event"], meta["session"])] = f
        except Exception:
            pass
    return found


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

    log("Pit Wall builder v4 (real-world maps, pit lanes, corners, night lighting)")
    now = datetime.now(timezone.utc)
    years = [int(y) for y in re.split(r"[,\s]+", args.years.strip()) if y] or [now.year - 1, now.year]
    skipped = [y for y in years if y < 2023]
    if skipped:
        log(f"Skipping {skipped}: OpenF1 has data from 2023 onwards only.")
    years = [y for y in years if y >= 2023]

    api = OpenF1()
    deadline = time.time() + args.budget_min * 60
    built = failed = nodata = 0
    first_error = None
    stop = False

    for year in years:
        if stop:
            break
        try:
            meetings = sorted(api.get("meetings", f"year={year}"), key=lambda m: m["date_start"])
            sessions = api.get("sessions", f"year={year}&session_type=Race") + api.get("sessions", f"year={year}&session_type=Qualifying")
        except ApiError as e:
            log(f"{year}: couldn't load the calendar: {e}")
            continue
        races = [m for m in meetings if "test" not in (m.get("meeting_name") or "").lower()]
        by_key = {m["meeting_key"]: m for m in races}
        # Official round numbers, so cancelled races don't shift the numbering
        official = official_rounds(year)
        rounds = {}
        for i, m in enumerate(races):
            rs = [s for s in sessions if s.get("meeting_key") == m["meeting_key"] and s.get("session_name") == "Race"]
            rnd = round_for(m, rs, official)
            if official and rnd is None:
                continue  # not on the official calendar any more (cancelled)
            rounds[m["meeting_key"]] = rnd or i + 1
        have = existing_files(year)
        for sess in sorted(sessions, key=lambda s: s["date_start"]):
            mk, name = sess.get("meeting_key"), sess.get("session_name")
            if mk not in rounds or name not in SESSION_ORDER:
                continue
            end = parse_dt(sess.get("date_end")) or parse_dt(sess["date_start"]) + timedelta(hours=3)
            if end > now - timedelta(hours=4):
                continue  # not finished yet
            rnd = rounds[mk]
            path = OUT / str(year) / f"{rnd:02d}_{slug(name)}.json"
            tel_dir = path.with_suffix("")
            # A file built earlier under a different round number: move it to the right name
            old = have.get((by_key[mk].get("meeting_name"), name))
            if old and old != path and not path.exists():
                data = json.loads(old.read_text())
                data["meta"]["round"] = rnd
                path.write_text(json.dumps(data, separators=(",", ":")))
                old.unlink()
                if old.with_suffix("").is_dir() and not tel_dir.exists():
                    old.with_suffix("").rename(tel_dir)
                log(f"Renumbered {old.name} -> {path.name}")
            upgrade = path.exists() and not args.force
            if upgrade and file_version(path) >= FILE_V:
                continue
            if time.time() > deadline:
                log("Time budget reached; the next run will continue from here.")
                stop = True
                break
            log(f"{year} round {rnd}: {by_key[mk].get('meeting_name')}, {name}" + (" (adding new data)" if upgrade else ""))
            started = time.time()
            try:
                if upgrade:
                    data = add_extras(api, sess, json.loads(path.read_text()), rnd, tel_dir)
                else:
                    data = add_extras(api, sess, build_session(api, sess, by_key[mk], rnd), rnd, tel_dir)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data, separators=(",", ":")))
                built += 1
                log(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {time.time() - started:.0f} s)")
            except NoData:
                nodata += 1
                log("  no data for this session (it may have been cancelled), skipping")
            except Exception as e:
                failed += 1
                log(f"  failed: {type(e).__name__}: {e}")
                if first_error is None:
                    first_error = traceback.format_exc()
                    log(first_error)

    log(f"Done: {built} built, {failed} failed, {nodata} without data.")
    total = sum(f.stat().st_size for f in OUT.rglob("*.json")) if OUT.exists() else 0
    log(f"Race data now takes {total / 1e6:.0f} MB (GitHub Pages sites can be up to 1 GB).")
    if total > 850e6:
        log("WARNING: close to the 1 GB GitHub Pages limit. Avoid TELEMETRY=all, or build fewer seasons.")
    if failed and not built:
        log("Every race failed to build, so nothing will be published. See the first error above.")
        sys.exit(1)
    write_index()


if __name__ == "__main__":
    main()
