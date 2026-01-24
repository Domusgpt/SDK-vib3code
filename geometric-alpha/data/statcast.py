"""
Statcast Data Client

Integration with pybaseball for fetching pitch-level telemetry data.
Provides geometric feature extraction from ball tracking data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

try:
    import pybaseball as pb
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False

from config.settings import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class StatcastPitchData:
    """Structured pitch telemetry data with kinematic vectors."""

    # Identification
    game_pk: int
    at_bat_number: int
    pitch_number: int
    pitcher_id: int
    batter_id: int

    # Kinematic vectors (at 50 ft release)
    vx0: float  # Velocity X component
    vy0: float  # Velocity Y component
    vz0: float  # Velocity Z component
    ax: float   # Acceleration X
    ay: float   # Acceleration Y
    az: float   # Acceleration Z

    # Release point
    release_pos_x: float
    release_pos_z: float
    release_extension: float

    # Spin physics
    release_speed: float
    release_spin_rate: float
    spin_axis: Optional[float]

    # Movement (pfx = pitch f(x) movement)
    pfx_x: float  # Horizontal movement (inches)
    pfx_z: float  # Vertical movement (inches)

    # Plate location
    plate_x: float
    plate_z: float

    # Outcome
    pitch_type: str
    description: str
    zone: int
    type: str  # Ball, Strike, In-play

    # Context
    game_date: datetime
    umpire_id: Optional[int] = None


class StatcastClient:
    """
    Client for fetching and managing Statcast pitch telemetry data.

    Uses pybaseball for data acquisition and implements:
    - Parquet-based caching for efficient storage
    - KNN imputation for missing kinematic data
    - Season partitioning for scalable queries
    """

    # Kinematic columns required for geometric analysis
    KINEMATIC_COLS = [
        'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
        'release_pos_x', 'release_pos_z', 'release_extension',
        'release_speed', 'release_spin_rate', 'spin_axis',
        'pfx_x', 'pfx_z', 'plate_x', 'plate_z'
    ]

    # Identification columns
    ID_COLS = [
        'game_pk', 'at_bat_number', 'pitch_number',
        'pitcher', 'batter', 'game_date'
    ]

    # Outcome columns
    OUTCOME_COLS = [
        'pitch_type', 'description', 'zone', 'type',
        'events', 'balls', 'strikes', 'outs_when_up',
        'on_1b', 'on_2b', 'on_3b'
    ]

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize Statcast client with optional cache directory."""
        if not PYBASEBALL_AVAILABLE:
            logger.warning("pybaseball not installed. Using mock data mode.")

        self.cache_dir = cache_dir or CONFIG.data.parquet_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_season(
        self,
        year: int,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch all Statcast data for a season.

        Args:
            year: Season year
            use_cache: Whether to use cached data if available
            force_refresh: Force re-fetch even if cached

        Returns:
            DataFrame with pitch-level telemetry
        """
        cache_path = self.cache_dir / f"statcast_{year}.parquet"

        # Check cache
        if use_cache and cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached Statcast data for {year}")
            return pd.read_parquet(cache_path)

        # Fetch from source
        logger.info(f"Fetching Statcast data for {year} season...")

        if PYBASEBALL_AVAILABLE:
            # Fetch in chunks to avoid timeout
            start_date = f"{year}-04-01"
            end_date = f"{year}-10-01"
            df = pb.statcast(start_dt=start_date, end_dt=end_date)
        else:
            # Mock data for testing without pybaseball
            df = self._generate_mock_data(year)

        # Clean and validate
        df = self._clean_statcast_data(df)

        # Save to cache
        if use_cache:
            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached {len(df)} pitches to {cache_path}")

        return df

    def fetch_date_range(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch Statcast data for a specific date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached season data

        Returns:
            DataFrame filtered to date range
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Determine which seasons to load
        years = list(range(start.year, end.year + 1))

        dfs = []
        for year in years:
            season_df = self.fetch_season(year, use_cache=use_cache)
            dfs.append(season_df)

        combined = pd.concat(dfs, ignore_index=True)

        # Filter to date range
        combined['game_date'] = pd.to_datetime(combined['game_date'])
        mask = (combined['game_date'] >= start) & (combined['game_date'] <= end)

        return combined[mask].reset_index(drop=True)

    def fetch_pitcher(
        self,
        pitcher_id: int,
        year: Optional[int] = None,
        min_pitches: int = 100
    ) -> pd.DataFrame:
        """
        Fetch all pitches for a specific pitcher.

        Args:
            pitcher_id: MLB pitcher ID
            year: Optional season filter
            min_pitches: Minimum pitches required for valid analysis

        Returns:
            DataFrame of pitcher's pitches
        """
        if year:
            df = self.fetch_season(year)
            pitcher_df = df[df['pitcher'] == pitcher_id].copy()
        else:
            # Load all available seasons
            dfs = []
            for year in range(CONFIG.data.statcast_start_year,
                             CONFIG.data.statcast_end_year + 1):
                season_df = self.fetch_season(year)
                dfs.append(season_df[season_df['pitcher'] == pitcher_id])

            pitcher_df = pd.concat(dfs, ignore_index=True)

        if len(pitcher_df) < min_pitches:
            logger.warning(
                f"Pitcher {pitcher_id} has only {len(pitcher_df)} pitches "
                f"(min: {min_pitches})"
            )

        return pitcher_df

    def _clean_statcast_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate Statcast data.

        - Filters out invalid velocity readings
        - Imputes missing kinematic data using KNN
        - Standardizes column names and types
        """
        original_len = len(df)

        # Filter by velocity bounds
        velocity_mask = (
            (df['release_speed'] >= CONFIG.data.min_pitch_velocity) &
            (df['release_speed'] <= CONFIG.data.max_pitch_velocity)
        )
        df = df[velocity_mask].copy()

        logger.info(
            f"Filtered {original_len - len(df)} pitches with invalid velocity"
        )

        # Handle missing kinematic data with KNN imputation
        df = self._impute_kinematics(df)

        # Ensure proper types
        df['game_date'] = pd.to_datetime(df['game_date'])

        for col in ['pitcher', 'batter', 'game_pk']:
            if col in df.columns:
                df[col] = df[col].astype('Int64')  # Nullable int

        return df

    def _impute_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing kinematic data using K-Nearest Neighbors.

        Preserves local geometric structure by imputing from similar
        pitches by the same pitcher with similar velocity and release point.
        """
        from sklearn.impute import KNNImputer

        # Group by pitcher and pitch type for local imputation
        kinematic_cols = [c for c in self.KINEMATIC_COLS if c in df.columns]

        # Check for missing data
        missing_mask = df[kinematic_cols].isnull().any(axis=1)
        if not missing_mask.any():
            return df

        logger.info(f"Imputing {missing_mask.sum()} pitches with missing data")

        # Apply KNN imputation within pitcher groups
        imputer = KNNImputer(n_neighbors=CONFIG.data.knn_impute_neighbors)

        def impute_group(group):
            if len(group) < CONFIG.data.knn_impute_neighbors * 2:
                return group  # Not enough data for KNN

            # Extract kinematic features
            features = group[kinematic_cols].values

            # Impute
            imputed = imputer.fit_transform(features)

            # Update group
            for i, col in enumerate(kinematic_cols):
                group[col] = imputed[:, i]

            return group

        df = df.groupby('pitcher', group_keys=False).apply(impute_group)

        return df

    def _generate_mock_data(self, year: int, n_pitches: int = 10000) -> pd.DataFrame:
        """Generate mock Statcast data for testing."""
        np.random.seed(year)

        n_games = n_pitches // 250

        data = {
            'game_pk': np.random.randint(1, n_games + 1, n_pitches),
            'at_bat_number': np.random.randint(1, 80, n_pitches),
            'pitch_number': np.random.randint(1, 10, n_pitches),
            'pitcher': np.random.randint(100000, 200000, n_pitches),
            'batter': np.random.randint(100000, 200000, n_pitches),
            'game_date': pd.date_range(f'{year}-04-01', f'{year}-09-30',
                                       periods=n_pitches),

            # Kinematic vectors
            'vx0': np.random.normal(-8, 3, n_pitches),
            'vy0': np.random.normal(-130, 10, n_pitches),
            'vz0': np.random.normal(-15, 5, n_pitches),
            'ax': np.random.normal(-10, 8, n_pitches),
            'ay': np.random.normal(25, 5, n_pitches),
            'az': np.random.normal(-20, 8, n_pitches),

            # Release point
            'release_pos_x': np.random.normal(-1, 1, n_pitches),
            'release_pos_z': np.random.normal(5.5, 0.5, n_pitches),
            'release_extension': np.random.normal(6.5, 0.5, n_pitches),

            # Spin
            'release_speed': np.random.normal(93, 5, n_pitches),
            'release_spin_rate': np.random.normal(2300, 400, n_pitches),
            'spin_axis': np.random.uniform(0, 360, n_pitches),

            # Movement
            'pfx_x': np.random.normal(-5, 8, n_pitches),
            'pfx_z': np.random.normal(5, 8, n_pitches),

            # Plate location
            'plate_x': np.random.normal(0, 0.8, n_pitches),
            'plate_z': np.random.normal(2.5, 0.8, n_pitches),

            # Outcomes
            'pitch_type': np.random.choice(
                ['FF', 'SL', 'CH', 'CU', 'SI', 'FC', 'KC'],
                n_pitches,
                p=[0.35, 0.18, 0.15, 0.12, 0.10, 0.06, 0.04]
            ),
            'description': np.random.choice(
                ['called_strike', 'ball', 'swinging_strike', 'foul',
                 'hit_into_play'],
                n_pitches,
                p=[0.15, 0.35, 0.12, 0.20, 0.18]
            ),
            'zone': np.random.randint(1, 14, n_pitches),
            'type': np.random.choice(['S', 'B', 'X'], n_pitches,
                                     p=[0.45, 0.37, 0.18]),
        }

        return pd.DataFrame(data)

    def get_trajectory(
        self,
        pitch: pd.Series,
        t: float
    ) -> Tuple[float, float, float]:
        """
        Calculate pitch position at time t using kinematics.

        Physics: x(t) = x0 + v0*t + 0.5*a*t^2

        Args:
            pitch: Series with kinematic data
            t: Time in seconds from release

        Returns:
            Tuple of (x, y, z) position
        """
        x = pitch['release_pos_x'] + pitch['vx0'] * t + 0.5 * pitch['ax'] * t**2
        y = 60.5 + pitch['vy0'] * t + 0.5 * pitch['ay'] * t**2  # 60.5 ft release
        z = pitch['release_pos_z'] + pitch['vz0'] * t + 0.5 * pitch['az'] * t**2

        return (x, y, z)

    def get_trajectory_at_decision_point(
        self,
        pitch: pd.Series
    ) -> Tuple[float, float, float]:
        """Get pitch position at the batter's decision point (~23.8 ft)."""
        return self.get_trajectory(pitch, CONFIG.geometric.decision_point_time)

    def get_trajectory_at_plate(
        self,
        pitch: pd.Series
    ) -> Tuple[float, float]:
        """Get pitch position at the plate (2D)."""
        return (pitch['plate_x'], pitch['plate_z'])


def synchronize_with_odds(
    pitches_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    time_col: str = 'game_date'
) -> pd.DataFrame:
    """
    Synchronize pitch data with odds using merge_asof.

    Ensures no look-ahead bias by matching each pitch with the most
    recent odds update BEFORE the game started.

    Args:
        pitches_df: Statcast pitch data
        odds_df: Odds history data
        time_col: Column to use for time matching

    Returns:
        Merged DataFrame with odds context
    """
    # Ensure both are sorted by time
    pitches_sorted = pitches_df.sort_values(time_col).copy()
    odds_sorted = odds_df.sort_values('commence_time').copy()

    # Merge asof - backward looking
    merged = pd.merge_asof(
        pitches_sorted,
        odds_sorted,
        left_on=time_col,
        right_on='commence_time',
        direction='backward',
        suffixes=('', '_odds')
    )

    return merged
