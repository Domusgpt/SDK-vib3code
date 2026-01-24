"""
Pitcher Arsenal Polytope Analysis

Models a pitcher's complete pitch repertoire as a high-dimensional
polytope in kinematic space. Analyzes the "shape" of skill.
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import logging

from config.settings import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ArsenalPolytope:
    """Representation of a pitcher's arsenal as a polytope."""

    pitcher_id: int

    # Hull properties (in high-dimensional space)
    hull_volume: float
    hull_vertices: int
    hull_facets: int

    # Per-pitch-type clusters
    pitch_clusters: Dict[str, Dict] = field(default_factory=dict)

    # Dimensionality analysis
    explained_variance_ratio: List[float] = field(default_factory=list)
    effective_dimensionality: float = 0.0

    # Stability metrics
    release_point_variance: float = 0.0
    velocity_variance: float = 0.0
    movement_variance: float = 0.0

    # Arsenal composition
    pitch_type_distribution: Dict[str, float] = field(default_factory=dict)

    # Clustering quality
    cluster_separation: float = 0.0  # Inter-cluster distance
    cluster_tightness: float = 0.0   # Intra-cluster variance

    # Sample info
    n_pitches: int = 0

    def is_stable(self) -> bool:
        """Check if arsenal shows consistent release and movement."""
        return (
            self.release_point_variance < 0.15 and
            self.cluster_tightness < 0.5
        )

    def is_diverse(self) -> bool:
        """Check if arsenal has diverse pitch types."""
        return len(self.pitch_type_distribution) >= 4


class ArsenalAnalyzer:
    """
    Analyzer for pitcher arsenal polytopes.

    Uses convex hull analysis in high-dimensional kinematic space
    to characterize a pitcher's capabilities.
    """

    def __init__(self, config=None):
        """Initialize arsenal analyzer."""
        self.config = config or CONFIG.geometric
        self.feature_cols = self.config.arsenal_dimensions
        self.scaler = StandardScaler()

    def compute_arsenal_polytope(
        self,
        pitches_df: pd.DataFrame,
        pitcher_id: int = 0
    ) -> Optional[ArsenalPolytope]:
        """
        Compute the polytope representation of a pitcher's arsenal.

        Args:
            pitches_df: DataFrame with single pitcher's pitches
            pitcher_id: Pitcher identifier

        Returns:
            ArsenalPolytope object
        """
        if len(pitches_df) < self.config.arsenal_min_pitches:
            logger.warning(
                f"Insufficient pitches ({len(pitches_df)}) for arsenal analysis"
            )
            return None

        # Extract and validate features
        available_cols = [c for c in self.feature_cols if c in pitches_df.columns]
        if len(available_cols) < 4:
            logger.warning("Insufficient feature columns for arsenal analysis")
            return None

        df = pitches_df[available_cols].dropna().copy()

        if len(df) < self.config.arsenal_min_pitches:
            return None

        # Normalize features
        features = self.scaler.fit_transform(df.values)

        # Compute convex hull
        hull_metrics = self._compute_hull_metrics(features)

        # Dimensionality analysis
        pca_results = self._analyze_dimensionality(features)

        # Cluster analysis per pitch type
        pitch_clusters = {}
        if 'pitch_type' in pitches_df.columns:
            valid_indices = df.index
            pitch_types = pitches_df.loc[valid_indices, 'pitch_type']

            for pt in pitch_types.unique():
                pt_mask = pitch_types == pt
                pt_features = features[pt_mask]

                if len(pt_features) >= 10:
                    pitch_clusters[pt] = self._analyze_cluster(pt_features)

        # Stability analysis
        stability = self._analyze_stability(df)

        # Pitch type distribution
        if 'pitch_type' in pitches_df.columns:
            distribution = pitches_df['pitch_type'].value_counts(normalize=True).to_dict()
        else:
            distribution = {}

        # Cluster quality
        cluster_separation, cluster_tightness = self._compute_cluster_quality(
            features, pitches_df.get('pitch_type')
        )

        return ArsenalPolytope(
            pitcher_id=pitcher_id,
            hull_volume=hull_metrics['volume'],
            hull_vertices=hull_metrics['vertices'],
            hull_facets=hull_metrics['facets'],
            pitch_clusters=pitch_clusters,
            explained_variance_ratio=pca_results['variance_ratio'],
            effective_dimensionality=pca_results['effective_dim'],
            release_point_variance=stability['release_variance'],
            velocity_variance=stability['velocity_variance'],
            movement_variance=stability['movement_variance'],
            pitch_type_distribution=distribution,
            cluster_separation=cluster_separation,
            cluster_tightness=cluster_tightness,
            n_pitches=len(df)
        )

    def _compute_hull_metrics(
        self,
        features: np.ndarray
    ) -> Dict:
        """Compute convex hull metrics in high-dimensional space."""
        try:
            # For high dimensions, use PCA to reduce before hull
            if features.shape[1] > 4:
                pca = PCA(n_components=min(4, features.shape[1]))
                reduced = pca.fit_transform(features)
            else:
                reduced = features

            hull = ConvexHull(reduced)

            return {
                'volume': hull.volume,
                'vertices': len(hull.vertices),
                'facets': len(hull.simplices)
            }
        except Exception as e:
            logger.warning(f"Hull computation failed: {e}")
            return {'volume': 0.0, 'vertices': 0, 'facets': 0}

    def _analyze_dimensionality(
        self,
        features: np.ndarray
    ) -> Dict:
        """Analyze effective dimensionality using PCA."""
        pca = PCA()
        pca.fit(features)

        variance_ratio = pca.explained_variance_ratio_.tolist()

        # Effective dimensionality: number of components for 90% variance
        cumulative = np.cumsum(variance_ratio)
        effective_dim = np.searchsorted(cumulative, 0.9) + 1

        return {
            'variance_ratio': variance_ratio,
            'effective_dim': float(effective_dim)
        }

    def _analyze_cluster(
        self,
        cluster_features: np.ndarray
    ) -> Dict:
        """Analyze a single pitch type cluster."""
        centroid = cluster_features.mean(axis=0)
        variance = cluster_features.var(axis=0).mean()

        # Distances from centroid
        distances = np.linalg.norm(cluster_features - centroid, axis=1)

        return {
            'centroid': centroid.tolist(),
            'variance': float(variance),
            'mean_distance': float(distances.mean()),
            'max_distance': float(distances.max()),
            'n_samples': len(cluster_features)
        }

    def _analyze_stability(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """Analyze release point and movement stability."""
        release_cols = [c for c in ['release_pos_x', 'release_pos_z']
                       if c in df.columns]
        velocity_cols = [c for c in ['release_speed'] if c in df.columns]
        movement_cols = [c for c in ['pfx_x', 'pfx_z'] if c in df.columns]

        release_variance = df[release_cols].var().mean() if release_cols else 0.0
        velocity_variance = df[velocity_cols].var().mean() if velocity_cols else 0.0
        movement_variance = df[movement_cols].var().mean() if movement_cols else 0.0

        return {
            'release_variance': float(release_variance),
            'velocity_variance': float(velocity_variance),
            'movement_variance': float(movement_variance)
        }

    def _compute_cluster_quality(
        self,
        features: np.ndarray,
        pitch_types: Optional[pd.Series]
    ) -> Tuple[float, float]:
        """Compute inter-cluster separation and intra-cluster tightness."""
        if pitch_types is None or len(pitch_types) == 0:
            return 0.0, 0.0

        # Get valid entries
        valid_mask = pitch_types.notna()
        valid_features = features[valid_mask]
        valid_types = pitch_types[valid_mask]

        unique_types = valid_types.unique()
        if len(unique_types) < 2:
            return 0.0, 0.0

        # Compute centroids
        centroids = {}
        variances = []

        for pt in unique_types:
            pt_mask = valid_types.values == pt
            pt_features = valid_features[pt_mask]

            if len(pt_features) >= 5:
                centroids[pt] = pt_features.mean(axis=0)
                variances.append(pt_features.var(axis=0).mean())

        if len(centroids) < 2:
            return 0.0, 0.0

        # Inter-cluster distance (average pairwise centroid distance)
        centroid_list = list(centroids.values())
        separations = []
        for i in range(len(centroid_list)):
            for j in range(i + 1, len(centroid_list)):
                dist = np.linalg.norm(centroid_list[i] - centroid_list[j])
                separations.append(dist)

        separation = np.mean(separations) if separations else 0.0
        tightness = np.mean(variances) if variances else 0.0

        return float(separation), float(tightness)

    def visualize_arsenal_2d(
        self,
        pitches_df: pd.DataFrame,
        method: str = 'pca'
    ) -> Dict:
        """
        Project arsenal to 2D for visualization.

        Args:
            pitches_df: Pitcher's pitch data
            method: 'pca' or 'tsne'

        Returns:
            Dict with 2D coordinates and labels
        """
        available_cols = [c for c in self.feature_cols if c in pitches_df.columns]
        df = pitches_df[available_cols].dropna().copy()

        if len(df) < 20:
            return {'x': [], 'y': [], 'labels': []}

        features = self.scaler.fit_transform(df.values)

        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1))
        else:
            reducer = PCA(n_components=2)

        reduced = reducer.fit_transform(features)

        # Get pitch type labels
        if 'pitch_type' in pitches_df.columns:
            labels = pitches_df.loc[df.index, 'pitch_type'].tolist()
        else:
            labels = ['unknown'] * len(reduced)

        return {
            'x': reduced[:, 0].tolist(),
            'y': reduced[:, 1].tolist(),
            'labels': labels
        }

    def detect_arsenal_instability(
        self,
        pitches_df: pd.DataFrame,
        rolling_window: int = 50
    ) -> List[Dict]:
        """
        Detect signs of arsenal instability over time.

        Instability (inconsistent clusters, drifting release point)
        often precedes poor performance.

        Args:
            pitches_df: Pitcher's pitch data sorted by time
            rolling_window: Number of pitches per window

        Returns:
            List of instability events
        """
        events = []

        if len(pitches_df) < rolling_window * 2:
            return events

        available_cols = [c for c in self.feature_cols if c in pitches_df.columns]

        for i in range(rolling_window, len(pitches_df) - rolling_window, rolling_window // 2):
            window_current = pitches_df.iloc[i:i + rolling_window]
            window_prev = pitches_df.iloc[i - rolling_window:i]

            # Compare release point stability
            if 'release_pos_x' in window_current.columns:
                current_release = window_current[['release_pos_x', 'release_pos_z']].mean()
                prev_release = window_prev[['release_pos_x', 'release_pos_z']].mean()

                drift = np.sqrt(
                    (current_release['release_pos_x'] - prev_release['release_pos_x'])**2 +
                    (current_release['release_pos_z'] - prev_release['release_pos_z'])**2
                )

                if drift > 0.2:  # Significant drift threshold
                    events.append({
                        'type': 'release_drift',
                        'position': i,
                        'magnitude': float(drift),
                        'timestamp': pitches_df.iloc[i].get('game_date')
                    })

            # Compare velocity stability
            if 'release_speed' in window_current.columns:
                current_velo = window_current['release_speed'].mean()
                prev_velo = window_prev['release_speed'].mean()

                velo_drop = prev_velo - current_velo

                if velo_drop > 1.5:  # 1.5 mph drop
                    events.append({
                        'type': 'velocity_drop',
                        'position': i,
                        'magnitude': float(velo_drop),
                        'timestamp': pitches_df.iloc[i].get('game_date')
                    })

        return events


def compute_arsenal_similarity(
    polytope_a: ArsenalPolytope,
    polytope_b: ArsenalPolytope
) -> float:
    """
    Compute similarity between two pitcher arsenals.

    Args:
        polytope_a: First pitcher's polytope
        polytope_b: Second pitcher's polytope

    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Compare pitch type distributions
    types_a = set(polytope_a.pitch_type_distribution.keys())
    types_b = set(polytope_b.pitch_type_distribution.keys())

    type_overlap = len(types_a & types_b) / len(types_a | types_b) if types_a | types_b else 0

    # Compare effective dimensionality
    dim_diff = abs(polytope_a.effective_dimensionality - polytope_b.effective_dimensionality)
    dim_score = max(0, 1 - dim_diff / 5)

    # Compare stability metrics
    stability_diff = abs(polytope_a.release_point_variance - polytope_b.release_point_variance)
    stability_score = max(0, 1 - stability_diff / 0.5)

    # Weighted combination
    return 0.4 * type_overlap + 0.3 * dim_score + 0.3 * stability_score
