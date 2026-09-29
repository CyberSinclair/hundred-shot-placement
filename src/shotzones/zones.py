"""Shot zones from k-means on landing coordinates (dissertation Section 4.2).

Zones are learned from where batters actually hit the ball rather than imposed
from named fielding positions. The same seed, scaler and ``n_init`` as the
dissertation are used so that zone numbering reproduces on the real data.

Note on scope: zones are fitted once on the full cleaned dataset, as in the
dissertation. Only FieldX / FieldY are used, and they define the target rather
than any feature, so no outcome information reaches the classifier. It does
mean the zone boundaries have seen the held-out matches; that is recorded as a
limitation rather than changed, because per-fold zones would give every fold a
different target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class ZoneModel:
    scaler: StandardScaler
    kmeans: KMeans

    @property
    def k(self) -> int:
        return self.kmeans.n_clusters

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.kmeans.predict(self.scaler.transform(df[["FieldX", "FieldY"]].to_numpy()))

    def centres(self) -> np.ndarray:
        """Cluster centres in the original 0-350 coordinate grid."""
        return self.scaler.inverse_transform(self.kmeans.cluster_centers_)


def fit_zones(df: pd.DataFrame, k: int = config.K_ZONES, seed: int = config.SEED) -> ZoneModel:
    coords = df[["FieldX", "FieldY"]].to_numpy(dtype=float)
    scaler = StandardScaler().fit(coords)
    km = KMeans(n_clusters=k, random_state=seed, n_init=20).fit(scaler.transform(coords))
    return ZoneModel(scaler=scaler, kmeans=km)


def k_search(df: pd.DataFrame, k_range=config.K_SEARCH_RANGE, seed: int = config.SEED,
             sample: int = config.SILHOUETTE_SAMPLE) -> pd.DataFrame:
    """Inertia and silhouette across k (Table 4.1)."""
    coords = StandardScaler().fit_transform(df[["FieldX", "FieldY"]].to_numpy(dtype=float))
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(coords)
        sil = silhouette_score(coords, labels, sample_size=min(sample, len(coords)),
                               random_state=seed)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def zone_profile(df: pd.DataFrame, zone_col: str = "shot_zone",
                 centres: np.ndarray | None = None) -> pd.DataFrame:
    """Per-zone size, mean runs, boundary %, dismissal rate (Tables 5.2 and 5.3)."""
    g = df.groupby(zone_col)
    prof = pd.DataFrame({
        "n": g.size(),
        "mean_runs": g["Runs"].mean(),
        "boundary_pct": g["is_boundary"].mean() * 100,
        "dismissal_pct": g["Wicket"].apply(lambda s: s.notna().mean() * 100),
    })
    if centres is not None:
        prof["centre_x"] = centres[prof.index, 0]
        prof["centre_y"] = centres[prof.index, 1]
    return prof
