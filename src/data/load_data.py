from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

League = Literal["M", "W"]


@dataclass
class DataPaths:
    base_dir: Path

    def csv_path(self, filename: str) -> Path:
        return self.base_dir / filename


def _regular_season_detailed_name(league: League) -> str:
    return f"{league}RegularSeasonDetailedResults.csv"


def _regular_season_compact_name(league: League) -> str:
    return f"{league}RegularSeasonCompactResults.csv"


def _tourney_compact_name(league: League) -> str:
    return f"{league}NCAATourneyCompactResults.csv"


def _tourney_seeds_name(league: League) -> str:
    return f"{league}NCAATourneySeeds.csv"


def _massey_ordinals_name(league: League) -> str:
    return f"{league}MasseyOrdinals.csv"


def _teams_name(league: League) -> str:
    return f"{league}Teams.csv"


def load_csv(paths: DataPaths, filename: str) -> pd.DataFrame:
    """Load a CSV from the data directory with basic type inference."""
    path = paths.csv_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {path}")
    df = pd.read_csv(path)
    return df


def load_regular_season_detailed(paths: DataPaths, league: League) -> pd.DataFrame:
    """Load detailed regular season results for a league."""
    df = load_csv(paths, _regular_season_detailed_name(league))
    df["League"] = league
    return df


def load_regular_season_compact(paths: DataPaths, league: League) -> pd.DataFrame:
    df = load_csv(paths, _regular_season_compact_name(league))
    df["League"] = league
    return df


def load_tourney_results(paths: DataPaths, league: League) -> pd.DataFrame:
    df = load_csv(paths, _tourney_compact_name(league))
    df["League"] = league
    return df


def load_tourney_seeds(paths: DataPaths, league: League) -> pd.DataFrame:
    df = load_csv(paths, _tourney_seeds_name(league))
    df["League"] = league
    return df


def load_massey_ordinals(paths: DataPaths, league: League) -> pd.DataFrame:
    """Load Massey Ordinals.

    Note: Only available for men's data (MMasseyOrdinals.csv) in this competition.
    For the women's league, this will return an empty DataFrame.
    """
    filename = _massey_ordinals_name(league)
    path = paths.csv_path(filename)
    if league == "W" and not path.exists():
        # No Massey data for women; return empty frame so downstream code
        # can safely treat ranking features as missing.
        return pd.DataFrame()

    df = load_csv(paths, filename)
    df["League"] = league
    return df


def load_teams(paths: DataPaths, league: League) -> pd.DataFrame:
    df = load_csv(paths, _teams_name(league))
    df["League"] = league
    return df


def filter_seasons(df: pd.DataFrame, min_season: int | None = None, max_season: int | None = None) -> pd.DataFrame:
    """Filter a DataFrame with a Season column to a given range."""
    out = df.copy()
    if min_season is not None:
        out = out[out["Season"] >= min_season]
    if max_season is not None:
        out = out[out["Season"] <= max_season]
    return out.reset_index(drop=True)

