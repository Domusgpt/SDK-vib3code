"""
Pitch Tunneling Analysis

Manifold intersection analysis for pitch trajectory deception.
Tunneling is the ability of a pitcher to make two different pitches
travel along the same trajectory for as long as possible before diverging.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

from config.settings import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class TunnelScore:
    """Tunnel score between two pitch types."""

    pitch_type_a: str
    pitch_type_b: str

    # Distances
    distance_at_tunnel: float  # Distance at decision point
    distance_at_plate: float   # Distance at plate

    # Derived score
    tunnel_score: float  # plate_distance / tunnel_distance

    # Sample size
    n_pairs: int

    # Velocity differential
    velocity_diff: float

    def is_elite(self) -> bool:
        """Check if tunnel score is elite (top 10%)."""
        return self.tunnel_score > 3.0


class TunnelAnalyzer:
    """
    Analyzer for pitch tunneling geometry.

    Models pitch trajectories as parametric curves in 3D space
    and computes the divergence between pitch pairs at critical
    decision points.
    """

    def __init__(self, config=None):
        """Initialize tunnel analyzer with configuration."""
        self.config = config or CONFIG.geometric
        self.t_decision = self.config.decision_point_time
        self.t_plate = self.config.plate_time
        self.epsilon = self.config.tunnel_epsilon

    def compute_trajectory(
        self,
        vx0: float, vy0: float, vz0: float,
        ax: float, ay: float, az: float,
        release_x: float, release_z: float,
        t: float
    ) -> Tuple[float, float, float]:
        """
        Compute pitch position at time t using kinematics.

        Physics: pos(t) = pos0 + v0*t + 0.5*a*t^2

        Args:
            vx0, vy0, vz0: Initial velocity components (ft/s)
            ax, ay, az: Acceleration components (ft/s^2)
            release_x, release_z: Release point
            t: Time in seconds

        Returns:
            Tuple of (x, y, z) position in feet
        """
        # Y starts at release point (~55 ft from plate)
        y0 = 55.0

        x = release_x + vx0 * t + 0.5 * ax * t**2
        y = y0 + vy0 * t + 0.5 * ay * t**2
        z = release_z + vz0 * t + 0.5 * az * t**2

        return (x, y, z)

    def compute_tunnel_scores(
        self,
        pitches_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute tunnel scores for all consecutive pitch pairs.

        This is the vectorized implementation for performance.

        Args:
            pitches_df: DataFrame with pitch telemetry

        Returns:
            DataFrame with tunnel scores for each pitch
        """
        required_cols = ['vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
                        'release_pos_x', 'release_pos_z',
                        'plate_x', 'plate_z', 'pitch_type']

        missing = [c for c in required_cols if c not in pitches_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = pitches_df.copy()

        # Compute position at decision point
        t = self.t_decision

        df['pos_decision_x'] = (
            df['release_pos_x'] +
            df['vx0'] * t +
            0.5 * df['ax'] * t**2
        )
        df['pos_decision_z'] = (
            df['release_pos_z'] +
            df['vz0'] * t +
            0.5 * df['az'] * t**2
        )

        # Position at plate is already available
        df['pos_plate_x'] = df['plate_x']
        df['pos_plate_z'] = df['plate_z']

        # Shift to compare with previous pitch
        for col in ['pos_decision_x', 'pos_decision_z',
                    'pos_plate_x', 'pos_plate_z', 'pitch_type']:
            df[f'prev_{col}'] = df[col].shift(1)

        # Compute Euclidean distances
        df['dist_tunnel'] = np.sqrt(
            (df['pos_decision_x'] - df['prev_pos_decision_x'])**2 +
            (df['pos_decision_z'] - df['prev_pos_decision_z'])**2
        )

        df['dist_plate'] = np.sqrt(
            (df['pos_plate_x'] - df['prev_pos_plate_x'])**2 +
            (df['pos_plate_z'] - df['prev_pos_plate_z'])**2
        )

        # Tunnel score = divergence ratio
        df['tunnel_score'] = df['dist_plate'] / (df['dist_tunnel'] + self.epsilon)

        # Flag same-pitcher sequences
        if 'pitcher' in df.columns:
            df['same_pitcher'] = df['pitcher'] == df['pitcher'].shift(1)
            df.loc[~df['same_pitcher'], 'tunnel_score'] = np.nan

        return df

    def compute_arsenal_tunnels(
        self,
        pitches_df: pd.DataFrame,
        min_pairs: int = 50
    ) -> Dict[str, TunnelScore]:
        """
        Compute tunnel scores for all pitch type pairs in an arsenal.

        Args:
            pitches_df: DataFrame with single pitcher's pitches
            min_pairs: Minimum number of pairs for reliable score

        Returns:
            Dict mapping (type_a, type_b) to TunnelScore
        """
        df = self.compute_tunnel_scores(pitches_df)

        # Group by pitch type pairs
        df['pair'] = df['prev_pitch_type'] + '-' + df['pitch_type']

        tunnel_scores = {}

        for pair, group in df.groupby('pair'):
            if len(group) < min_pairs:
                continue

            type_a, type_b = pair.split('-')

            # Skip same pitch type
            if type_a == type_b:
                continue

            tunnel_scores[pair] = TunnelScore(
                pitch_type_a=type_a,
                pitch_type_b=type_b,
                distance_at_tunnel=group['dist_tunnel'].mean(),
                distance_at_plate=group['dist_plate'].mean(),
                tunnel_score=group['tunnel_score'].mean(),
                n_pairs=len(group),
                velocity_diff=0  # Would need velocity data to compute
            )

        return tunnel_scores

    def compute_pitcher_tunnel_matrix(
        self,
        pitches_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute the full tunnel matrix for a pitcher.

        Returns a matrix where entry (i, j) is the tunnel score
        from pitch type i to pitch type j.

        Args:
            pitches_df: DataFrame with single pitcher's pitches

        Returns:
            DataFrame matrix of tunnel scores
        """
        tunnel_scores = self.compute_arsenal_tunnels(pitches_df)

        # Get unique pitch types
        pitch_types = sorted(set(
            list(pitches_df['pitch_type'].unique())
        ))

        # Create matrix
        matrix = pd.DataFrame(
            index=pitch_types,
            columns=pitch_types,
            dtype=float
        )

        for pair, score in tunnel_scores.items():
            type_a, type_b = pair.split('-')
            if type_a in pitch_types and type_b in pitch_types:
                matrix.loc[type_a, type_b] = score.tunnel_score

        return matrix

    def compute_arsenal_graph_connectivity(
        self,
        tunnel_matrix: pd.DataFrame,
        threshold: float = 2.0
    ) -> Dict:
        """
        Analyze the arsenal as a graph where edges are strong tunnels.

        A highly connected arsenal presents a "single geometric front"
        to the batter, making it hard to differentiate pitch types.

        Args:
            tunnel_matrix: Matrix of tunnel scores
            threshold: Minimum tunnel score for "strong" connection

        Returns:
            Dict with graph metrics
        """
        # Count strong connections
        strong_tunnels = (tunnel_matrix > threshold).sum().sum()
        possible_tunnels = tunnel_matrix.notna().sum().sum()

        # Average tunnel score
        avg_tunnel = tunnel_matrix.mean().mean()

        # Max tunnel score
        max_tunnel = tunnel_matrix.max().max()

        # Connectivity ratio
        connectivity = strong_tunnels / possible_tunnels if possible_tunnels > 0 else 0

        return {
            'strong_tunnel_count': strong_tunnels,
            'possible_tunnels': possible_tunnels,
            'connectivity_ratio': connectivity,
            'avg_tunnel_score': avg_tunnel,
            'max_tunnel_score': max_tunnel
        }

    def identify_elite_tunnel_pairs(
        self,
        pitches_df: pd.DataFrame,
        top_n: int = 3
    ) -> List[TunnelScore]:
        """
        Identify the pitcher's best tunnel combinations.

        Args:
            pitches_df: DataFrame with single pitcher's pitches
            top_n: Number of top pairs to return

        Returns:
            List of top TunnelScore objects
        """
        tunnel_scores = self.compute_arsenal_tunnels(pitches_df)

        # Sort by tunnel score
        sorted_scores = sorted(
            tunnel_scores.values(),
            key=lambda x: x.tunnel_score,
            reverse=True
        )

        return sorted_scores[:top_n]


def compute_tunnel_score_vectorized(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorized tunnel score computation for entire dataset.

    Optimized for GPU acceleration (compatible with cudf).

    Args:
        df: DataFrame with pitch data

    Returns:
        Array of tunnel scores
    """
    t_dec = CONFIG.geometric.decision_point_time
    epsilon = CONFIG.geometric.tunnel_epsilon

    # Extract arrays
    vx0 = df['vx0'].values
    vy0 = df['vy0'].values
    vz0 = df['vz0'].values
    ax = df['ax'].values
    ay = df['ay'].values
    az = df['az'].values
    release_x = df['release_pos_x'].values
    release_z = df['release_pos_z'].values
    plate_x = df['plate_x'].values
    plate_z = df['plate_z'].values

    # Compute decision point positions
    pos_dec_x = release_x + vx0 * t_dec + 0.5 * ax * t_dec**2
    pos_dec_z = release_z + vz0 * t_dec + 0.5 * az * t_dec**2

    # Shift for consecutive comparison
    prev_pos_dec_x = np.roll(pos_dec_x, 1)
    prev_pos_dec_z = np.roll(pos_dec_z, 1)
    prev_plate_x = np.roll(plate_x, 1)
    prev_plate_z = np.roll(plate_z, 1)

    # Euclidean distances
    dist_tunnel = np.sqrt(
        (pos_dec_x - prev_pos_dec_x)**2 +
        (pos_dec_z - prev_pos_dec_z)**2
    )

    dist_plate = np.sqrt(
        (plate_x - prev_plate_x)**2 +
        (plate_z - prev_plate_z)**2
    )

    # Tunnel score
    tunnel_score = dist_plate / (dist_tunnel + epsilon)

    # First row is invalid (no previous pitch)
    tunnel_score[0] = np.nan

    return tunnel_score
