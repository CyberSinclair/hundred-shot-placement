"""Central configuration.

Every constant that shaped a result in the dissertation lives here, so a reader can
see the whole experimental setup in one place and a reviewer can change one value
and re-run.
"""

from __future__ import annotations

SEED = 42

# ---------------------------------------------------------------------------
# Raw export defect (dissertation Section 3.2.3)
# ---------------------------------------------------------------------------
# On the 2023 and 2024 seasons five extra columns were inserted at a fixed
# position, pushing every later value five columns to the right. The rows are
# fingerprinted by a text value that belongs in "Ball Type" turning up in a
# fielder coordinate column.
SHIFT_SENTINEL_COLUMN = "Pitchside Fielder 2 Y"
SHIFT_SENTINEL_VALUE = "Kookaburra"
SHIFT_WIDTH = 5
# Position of the first displaced column. 105 in the real Hampshire export; the
# synthetic sample uses its own (smaller) layout, so the pipeline locates the
# start column by name rather than hard-coding the index.
SHIFT_START_COLUMN = "PitchX"

# Columns that must be text after repair; everything else from the shift point on
# is coerced to numeric.
SHIFT_TEXT_COLUMNS = ["Ball Type", "Ball Colour"]

# ---------------------------------------------------------------------------
# Cleaning (dissertation Section 3.5, Table 3.1)
# ---------------------------------------------------------------------------
NOT_HIT_CONNECTIONS = ["Missed", "Left", "Hit Pad", "Hit Body"]

# (label, columns) pairs applied in order with dropna. The order matters because
# the audit log records the rows removed at each step.
DROPNA_STEPS = [
    ("Drop null FieldX / FieldY (target)", ["FieldX", "FieldY"]),
    ("Drop null core delivery features", ["PitchX", "PitchY", "Bowler Type", "Line", "Length"]),
    ("Drop null Shot", ["Shot"]),
]
TRACKING_DROPNA_STEPS = [
    ("Drop null SwingAngle / DropAngle", ["SwingAngle", "DropAngle"]),
    ("Drop null Release coordinates", ["ReleaseX", "ReleaseY", "ReleaseZ"]),
    ("Drop null Speed", ["Speed"]),
    ("Drop null BounceAngle / ReleaseAngle", ["BounceAngle", "ReleaseAngle"]),
    ("Drop null Feet", ["Feet"]),
]

# ---------------------------------------------------------------------------
# Shot zones (dissertation Section 4.2)
# ---------------------------------------------------------------------------
K_ZONES = 8
K_SEARCH_RANGE = range(2, 16)
SILHOUETTE_SAMPLE = 5000

# ---------------------------------------------------------------------------
# Features (dissertation Sections 3.6 and 3.7)
# ---------------------------------------------------------------------------
# The original notebook built features by *excluding* a blocklist, which meant
# any column not on the list (for example "Extra Runs") was silently used. The
# refactor uses an explicit allowlist instead: a column is a feature only if it
# is named here.
TRACKING_FEATURES = [
    "PitchX", "PitchY",
    "ImpactX", "ImpactY", "ImpactZ",
    "PastX", "PastY", "PastZ",
    "ReleaseX", "ReleaseY", "ReleaseZ",
    "ReleaseAccelerationY", "ReleaseAccelerationZ",
    "Speed", "SwingAngle", "DropAngle", "BounceAngle", "ReleaseAngle", "Deviation",
]

# Match-state counters. These are only valid features if they describe the state
# *before* the ball is bowled. Use features.check_cumulative_timing() on the real
# export to confirm; if a column turns out to be post-ball, convert it with
# features.to_pre_delivery() before modelling.
GAME_STATE_FEATURES = [
    "Cumulative Batter Runs",
    "Cumulative Batter Balls",
    "Team Runs",
    "Team Wickets",
    "Req Run Rate At Start",
]

ENGINEERED_FEATURES = ["balls_remaining", "is_second_innings", "is_powerplay", "is_left_handed"]

CATEGORICAL_FEATURES = ["Bowler Type", "Line", "Length", "Delivery"]
YES_NO_FEATURES = ["Around the Wicket", "Keeper Up", "Free Hit"]

# Columns describing what happened on the ball. None may reach the feature matrix.
LEAKAGE_COLUMNS = [
    "FieldX", "FieldY", "Runs", "Extra Runs", "Shot", "Connection", "Ball Outcome",
    "Wicket", "Dismissed Batter", "Run Rate After", "Req Run Rate After",
    "Result", "Winning Team", "Losing Team", "Feet",
]

# Median-imputed tracking columns (Section 3.5). Medians are learned on the
# training fold only, never on the held-out match.
MEDIAN_IMPUTE = [
    "ImpactX", "ImpactY", "ImpactZ", "Deviation",
    "PastX", "PastY", "PastZ", "ReleaseAccelerationY", "ReleaseAccelerationZ",
]
FIELDER_SENTINEL = -1.0

# ---------------------------------------------------------------------------
# Model (dissertation Section 4.3). Conventional values, deliberately not tuned.
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="multi:softmax",
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)
MIN_MATCH_DELIVERIES = 20  # matches smaller than this are not evaluated

# ---------------------------------------------------------------------------
# Backtesting and off-policy evaluation (Sections 4.6 and 4.8)
# ---------------------------------------------------------------------------
RUN_VALUES = [0, 1, 2, 3, 4, 6]
N_PERMUTATIONS = 2000
N_BOOTSTRAP = 10_000
PROPENSITY_FLOOR = 0.01
REWARD_SHRINK_ZONE = 50
REWARD_SHRINK_CELL = 25

PACE_TYPES = ["RF", "RFM", "RM", "LF", "LFM", "LM"]
SPIN_TYPES = {"ROB": "Right-arm off spin", "LOB": "Left-arm orthodox", "RLB": "Leg spin"}

# Innings phases by balls remaining (balls_remaining = 100 - Innings Ball).
# The dissertation used pd.cut(bins=[0, 20, 50, 100]), which is right-inclusive
# and so silently dropped the 100th ball (balls_remaining == 0) from every phase
# table. The lower edge is -1 here so that ball is kept. The dissertation also
# labels the first band "Powerplay", but it spans balls 1-49, of which the actual
# powerplay is only the first 25. The labels below say what each band contains.
PHASE_BINS = [-1, 20, 50, 100]
PHASE_LABELS = ["Death (balls 80-100)", "Middle (balls 50-79)", "Opening (balls 1-49)"]
PHASE_ORDER = ["Opening (balls 1-49)", "Middle (balls 50-79)", "Death (balls 80-100)"]
