"""Synthetic ball-by-ball data in the shape of the Hampshire export.

The real data is covered by a data sharing agreement and cannot be published.
This module generates a stand-in with the same column names, the same kinds of
values, and the same three data quality defects documented in the dissertation:

1. The five-column export shift on the 2023 and 2024 seasons.
2. Deliveries with no bat contact that still carry FieldX / FieldY (the
   wicketkeeper's position).
3. Fielder coordinates that only exist for part of the 2025 season.

It is built so the pipeline has something realistic to run on and the repair
code has a real defect to fix. The cricket is plausible but invented: numbers
produced from this file say nothing about The Hundred.

Usage::

    python -m shotzones.synthetic --matches 12 --out data/sample/synthetic_raw.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

FIELD_CENTRE = np.array([175.0, 175.0])
BOUNDARY_RADIUS = 165.0
KEEPER_POSITION = np.array([175.0, 120.0])

TEAMS = ["Northern Owls", "Coastal Gulls", "City Foxes", "River Herons",
         "Valley Hawks", "Harbour Seals", "Forest Stags", "Moor Kites"]

BOWLER_TYPES = ["RF", "RFM", "RM", "LF", "LFM", "LM", "ROB", "LOB", "RLB"]
PACE_WEIGHTS = np.array([0.20, 0.25, 0.12, 0.08, 0.10, 0.05, 0.0, 0.0, 0.0])
SPIN_WEIGHTS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40, 0.30, 0.30])

LINES = ["Outside Off", "Off Stump", "Middle Stump", "Leg Stump", "Leg Wide", "Wide Outside Off"]
LINE_OFFSET = {"Wide Outside Off": -0.9, "Outside Off": -0.5, "Off Stump": -0.2,
               "Middle Stump": 0.0, "Leg Stump": 0.3, "Leg Wide": 0.8}
LENGTHS = ["Yorker", "Full", "Good", "Back Of Length", "Short", "Bouncer"]
LENGTH_METRES = {"Yorker": 1.0, "Full": 3.5, "Good": 6.0, "Back Of Length": 8.0,
                 "Short": 10.0, "Bouncer": 12.5}
PACE_DELIVERIES = ["No Movement", "Seam Away", "Seam In", "Swing Away", "Swing In"]
SPIN_DELIVERIES = ["No Turn", "Turn Away", "Turn In"]
CONNECTIONS_HIT = ["Middled", "Well Timed", "Edged", "Top Edge", "Mis-hit"]

N_FIELDERS = 9


def _column_layout() -> tuple[list[str], list[str]]:
    """Return (columns before the shift point, columns from the shift point on).

    The order mirrors the real export where it matters for the defect:
    ReleaseX/Y/Z sit five columns before BounceSpeed/StumpSpeed/DropSpeed, and
    Ball Type / Ball Colour sit five columns before Pitchside Fielder 2 Y / 3 X.
    Those adjacencies produce the same fingerprints described in Section 3.2.3.
    """
    front = [
        "Match", "Date", "Innings", "Innings Ball", "Batting Team", "Bowling Team",
        "Batter ID", "Bowler ID", "Batting Hand", "Bowler Type",
        "Around the Wicket", "Keeper Up", "Free Hit", "Legal Ball", "Power Play",
        "Line", "Length", "Delivery", "Feet", "Shot", "Connection",
        "Runs", "Extra Runs", "Wicket", "Ball Outcome",
        "Cumulative Batter Runs", "Cumulative Batter Balls", "Team Runs", "Team Wickets",
        "Req Run Rate At Start", "FieldX", "FieldY",
    ]
    shifted = [
        "PitchX", "PitchY", "ImpactX", "ImpactY", "ImpactZ", "PastX", "PastY", "PastZ",
        "ReleaseX", "ReleaseY", "ReleaseZ", "ReleaseAccelerationY", "ReleaseAccelerationZ",
        "BounceSpeed", "StumpSpeed", "DropSpeed",
        "Speed", "SwingAngle", "DropAngle", "BounceAngle", "ReleaseAngle", "Deviation",
        "Ball Type", "Ball Colour",
    ]
    for i in range(1, N_FIELDERS + 1):
        shifted += [f"Pitchside Fielder {i} X", f"Pitchside Fielder {i} Y"]
    return front, shifted


def _polar_to_field(angle: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Angle 0 points straight down the ground (increasing Y)."""
    x = FIELD_CENTRE[0] + radius * np.sin(angle)
    y = FIELD_CENTRE[1] + radius * np.cos(angle)
    return np.clip(np.column_stack([x, y]), 0, 350)


def _simulate_innings(rng, match_id, date, innings, bat, bowl, target, season):
    """Simulate one innings of legal and illegal deliveries."""
    batters = [f"{bat[:3].upper()}{rng.integers(100, 999)}" for _ in range(8)]
    left_handed = {b: rng.random() < 0.3 for b in batters}
    bowlers = [f"{bowl[:3].upper()}B{i}" for i in range(6)]
    bowler_type = {}
    for i, b in enumerate(bowlers):
        w = PACE_WEIGHTS if i < 4 else SPIN_WEIGHTS
        bowler_type[b] = rng.choice(BOWLER_TYPES, p=w / w.sum())

    striker_idx, next_in = 0, 2
    bat_runs = {b: 0 for b in batters}
    bat_balls = {b: 0 for b in batters}
    team_runs = team_wkts = 0
    rows = []
    ball = 0
    while ball < 100 and team_wkts < 10:
        # Five-ball sets, ten-ball ends: pick bowler per ten.
        bowler = bowlers[(ball // 10) % len(bowlers)]
        btype = bowler_type[bowler]
        is_pace = btype in config.PACE_TYPES
        batter = batters[striker_idx]
        lhb = left_handed[batter]
        balls_remaining = 100 - (ball + 1)

        legal = rng.random() > 0.04
        line = rng.choice(LINES, p=[0.28, 0.2, 0.17, 0.13, 0.1, 0.12])
        length = rng.choice(LENGTHS, p=[0.07, 0.24, 0.3, 0.2, 0.13, 0.06] if is_pace
                            else [0.02, 0.3, 0.45, 0.18, 0.05, 0.0])
        delivery = rng.choice(PACE_DELIVERIES if is_pace else SPIN_DELIVERIES)

        # Batting intent rises through the innings and with a chase.
        phase_aggr = 0.30 if ball < 25 else 0.25 if ball < 80 else 0.75
        good_ball = length in ("Good", "Back Of Length") and abs(LINE_OFFSET[line]) < 0.4
        aggression = np.clip(phase_aggr + rng.normal(0, 0.15) - 0.15 * good_ball
                             + (0.1 if length in ("Short", "Full") else 0), 0.02, 0.95)

        # Contact
        p_miss = 0.10 + 0.08 * good_ball + (0.05 if is_pace and ball < 25 else 0)
        hit = rng.random() > p_miss
        mirror = -1.0 if lhb else 1.0
        off = LINE_OFFSET[line]

        if hit:
            attacking = rng.random() < aggression
            # Shot angle: full balls go straight, short balls square, line pushes
            # the ball to the off or leg side; left-handers mirror.
            square = np.clip((LENGTH_METRES[length] - 4) / 8, 0, 1)
            side = np.sign(off + rng.normal(0, 0.35)) or 1.0
            angle = mirror * side * (0.35 + 1.4 * square + rng.normal(0, 0.35))
            if attacking:
                radius = rng.uniform(95, 172)
            else:
                radius = np.abs(rng.normal(45, 22))
            pos = _polar_to_field(np.array([angle]), np.array([radius]))[0]
            if radius > 150:
                runs = 6 if (attacking and rng.random() < 0.35 + 0.2 * (ball >= 80)) else 4
                radius = rng.uniform(155, 168)
                pos = _polar_to_field(np.array([angle]), np.array([radius]))[0]
            elif radius > 95:
                runs = rng.choice([1, 2, 3, 4], p=[0.45, 0.4, 0.03, 0.12])
            elif radius > 35:
                runs = rng.choice([0, 1, 2], p=[0.3, 0.62, 0.08])
            else:
                runs = rng.choice([0, 1], p=[0.8, 0.2])
            connection = rng.choice(CONNECTIONS_HIT, p=[0.35, 0.3, 0.15, 0.1, 0.1])
            p_out = 0.015 + 0.06 * attacking * (95 < radius <= 150) + 0.03 * (radius < 30)
            shot = ("Drive" if square < 0.3 else "Pull" if side * mirror > 0 else "Cut") \
                if attacking else ("Defence" if radius < 35 else "Push" if square < 0.5 else "Glance")
        else:
            angle, radius = 0.0, 0.0
            pos = KEEPER_POSITION + rng.normal(0, 6, 2)
            runs = 0
            connection = rng.choice(config.NOT_HIT_CONNECTIONS, p=[0.35, 0.35, 0.2, 0.1])
            p_out = 0.04
            shot = "Leave" if connection == "Left" else "No Shot"

        wicket = rng.random() < p_out
        if wicket:
            runs = 0
        extra = 0
        if not legal:
            extra = 1
        elif not hit and rng.random() < 0.08:
            extra = rng.choice([1, 4], p=[0.8, 0.2])  # byes / leg byes

        # Tracking measurements. PastY and ImpactX carry the most signal, loosely
        # echoing the real importance ranking.
        speed = rng.normal(132, 6) if is_pace else rng.normal(84, 5)
        pitch_x = 0.1 * mirror + off * 0.6 + rng.normal(0, 0.08)
        pitch_y = LENGTH_METRES[length] + rng.normal(0, 0.6)
        past_y = off * 0.55 + 0.25 * np.sin(angle) * hit + rng.normal(0, 0.12)
        impact_x = 0.3 + 0.2 * np.cos(angle) * hit + rng.normal(0, 0.1)
        impact_y = off * 0.4 + 0.15 * np.sin(angle) * hit + rng.normal(0, 0.1)

        row = {
            "Match": match_id,
            "Date": date,
            "Innings": innings,
            "Innings Ball": ball + 1,
            "Batting Team": bat,
            "Bowling Team": bowl,
            "Batter ID": batter,
            "Bowler ID": bowler,
            "Batting Hand": "LHB" if lhb else "RHB",
            "Bowler Type": btype,
            "Around the Wicket": "Yes" if rng.random() < 0.2 else "No",
            "Keeper Up": "Yes" if (not is_pace and rng.random() < 0.7) else "No",
            "Free Hit": "No",
            "Legal Ball": "Yes" if legal else "No",
            "Power Play": "PP" if ball < 25 else np.nan,
            "Line": line,
            "Length": length,
            "Delivery": delivery if rng.random() > 0.16 else np.nan,
            "Feet": rng.choice(["Front", "Back", "Across", "Down The Pitch"]),
            "Shot": shot,
            "Connection": connection,
            "Runs": int(runs),
            "Extra Runs": int(extra),
            "Wicket": "Caught" if wicket and hit else ("Bowled" if wicket else np.nan),
            "Ball Outcome": "W" if wicket else str(runs),
            # Match-state counters are stored as the state BEFORE the ball.
            "Cumulative Batter Runs": bat_runs[batter],
            "Cumulative Batter Balls": bat_balls[batter],
            "Team Runs": team_runs,
            "Team Wickets": team_wkts,
            "Req Run Rate At Start": (max(target - team_runs, 0) / max(balls_remaining + 1, 1) * 5)
            if target else np.nan,
            "FieldX": round(pos[0], 1),
            "FieldY": round(pos[1], 1),
            "PitchX": round(pitch_x, 3),
            "PitchY": round(pitch_y, 3),
            "ImpactX": round(impact_x, 3),
            "ImpactY": round(impact_y, 3),
            "ImpactZ": round(0.6 + rng.normal(0, 0.15), 3),
            "PastX": round(rng.normal(-0.5, 0.2), 3),
            "PastY": round(past_y, 3),
            "PastZ": round(0.7 + 0.03 * (12 - LENGTH_METRES[length]) + rng.normal(0, 0.1), 3),
            "ReleaseX": round((0.8 if btype.startswith("L") else -0.8) + rng.normal(0, 0.15), 3),
            "ReleaseY": round(rng.normal(19.5, 0.3), 3),
            "ReleaseZ": round(rng.normal(2.1 if is_pace else 1.9, 0.08), 3),
            "ReleaseAccelerationY": round(rng.normal(-1.5, 0.4), 3),
            "ReleaseAccelerationZ": round(rng.normal(-9.5, 0.6), 3),
            "BounceSpeed": np.nan,
            "StumpSpeed": np.nan,
            "DropSpeed": np.nan,
            "Speed": round(speed, 1),
            "SwingAngle": round(rng.normal(0.14, 0.6), 3),
            "DropAngle": round(rng.normal(-13.9, 2.5), 3),
            "BounceAngle": round(rng.normal(12, 3), 3),
            "ReleaseAngle": round(rng.normal(-6, 2), 3),
            "Deviation": round(rng.normal(0, 0.8 if not is_pace else 0.4), 3),
            "Ball Type": "Kookaburra",
            "Ball Colour": "White",
        }
        # Fielders: 2025 only, and only on about a third of deliveries.
        has_fielders = season == 2025 and rng.random() < 0.335
        for i in range(1, N_FIELDERS + 1):
            if has_fielders:
                a, r = rng.uniform(-np.pi, np.pi), rng.uniform(40, 160)
                fx, fy = _polar_to_field(np.array([a]), np.array([r]))[0]
                row[f"Pitchside Fielder {i} X"] = round(fx, 1)
                row[f"Pitchside Fielder {i} Y"] = round(fy, 1)
            else:
                row[f"Pitchside Fielder {i} X"] = np.nan
                row[f"Pitchside Fielder {i} Y"] = np.nan
        rows.append(row)

        team_runs += runs + extra
        if legal:
            ball += 1
            bat_runs[batter] += runs
            bat_balls[batter] += 1
            if wicket:
                team_wkts += 1
                striker_idx = next_in if next_in < len(batters) else striker_idx
                next_in += 1
            elif runs % 2 == 1:
                striker_idx = 1 - striker_idx if striker_idx < 2 else striker_idx
        if target and team_runs > target:
            break
    return rows, team_runs


def _sprinkle_missing(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Add the scattered missing values the cleaning pipeline has to deal with."""
    rates = {"FieldX": 0.05, "Shot": 0.006, "SwingAngle": 0.001, "ReleaseX": 0.015,
             "Feet": 0.003, "ImpactZ": 0.02, "ImpactX": 0.03, "Deviation": 0.02}
    for col, rate in rates.items():
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
        if col == "FieldX":
            df.loc[mask, "FieldY"] = np.nan
    return df


def _apply_export_shift(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Reproduce the 2023/24 export defect: insert five columns, push the rest right.

    The last five values fall off the end of the row, as in the real export.
    """
    front, shifted = _column_layout()
    start = len(front)
    width = config.SHIFT_WIDTH
    bad = df["Date"].dt.year.isin([2023, 2024]).to_numpy()
    values = df.to_numpy(dtype=object, copy=True)
    n_bad = bad.sum()
    junk = rng.normal(0, 1, size=(n_bad, width)).round(3).astype(object)
    tail = values[bad, start:]
    new_tail = np.concatenate([junk, tail[:, :-width]], axis=1)
    values[bad, start:] = new_tail
    return pd.DataFrame(values, columns=df.columns)


def generate(n_matches: int = 12, seed: int = config.SEED, with_defects: bool = True) -> pd.DataFrame:
    """Generate a raw synthetic export.

    Matches are spread across the 2023, 2024 and 2025 seasons so the column
    shift affects roughly two thirds of rows, as in the real data.
    """
    rng = np.random.default_rng(seed)
    front, shifted = _column_layout()
    all_rows = []
    seasons = np.array([2023, 2024, 2025])
    for m in range(n_matches):
        season = int(seasons[m % 3])
        date = pd.Timestamp(f"{season}-08-01") + pd.Timedelta(days=int(m // 3))
        home, away = rng.choice(TEAMS, size=2, replace=False)
        match_id = f"{date:%Y-%m-%d} {home} v {away}"
        rows1, score = _simulate_innings(rng, match_id, date, 1, home, away, None, season)
        rows2, _ = _simulate_innings(rng, match_id, date, 2, away, home, score, season)
        all_rows += rows1 + rows2

    df = pd.DataFrame(all_rows)[front + shifted]
    df = _sprinkle_missing(df, rng)
    if with_defects:
        df = _apply_export_shift(df, rng)
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Hundred ball-by-ball data")
    parser.add_argument("--matches", type=int, default=12)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out", type=Path, default=Path("data/sample/synthetic_raw.csv"))
    parser.add_argument("--no-defects", action="store_true",
                        help="Write clean data without the export shift")
    args = parser.parse_args(argv)
    df = generate(args.matches, args.seed, with_defects=not args.no_defects)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} columns to {args.out}")


if __name__ == "__main__":
    main()
