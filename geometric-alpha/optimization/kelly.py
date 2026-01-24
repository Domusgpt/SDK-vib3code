"""
Simultaneous Kelly Criterion Optimizer

Convex optimization for optimal bet sizing across multiple
concurrent betting opportunities.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import logging

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from config.settings import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class BettingOpportunity:
    """Representation of a betting opportunity."""

    opportunity_id: str
    game_id: str

    # Teams
    home_team: str
    away_team: str

    # Bet details
    bet_type: str  # 'moneyline', 'total', 'spread'
    selection: str  # 'home', 'away', 'over', 'under'

    # Odds (decimal format)
    decimal_odds: float

    # Model's probability estimate
    model_prob: float

    # Market implied probability
    market_prob: float

    # Correlation group (for correlated bets)
    correlation_group: Optional[str] = None

    @property
    def american_odds(self) -> float:
        """Convert decimal odds to American format."""
        if self.decimal_odds >= 2.0:
            return (self.decimal_odds - 1) * 100
        else:
            return -100 / (self.decimal_odds - 1)

    @property
    def edge(self) -> float:
        """Calculate edge vs market."""
        return self.model_prob - self.market_prob

    @property
    def expected_value(self) -> float:
        """Calculate expected value per unit wagered."""
        return self.model_prob * (self.decimal_odds - 1) - (1 - self.model_prob)

    def is_value(self, min_edge: float = 0.02) -> bool:
        """Check if opportunity has sufficient edge."""
        return self.edge >= min_edge


@dataclass
class OptimalPortfolio:
    """Result of Kelly optimization."""

    # Bet allocations (as fraction of bankroll)
    allocations: Dict[str, float] = field(default_factory=dict)

    # Portfolio metrics
    expected_growth_rate: float = 0.0
    total_exposure: float = 0.0
    max_single_bet: float = 0.0

    # Solver info
    solver_status: str = ""
    solver_time: float = 0.0

    def get_bet_amounts(self, bankroll: float) -> Dict[str, float]:
        """Convert fractions to dollar amounts."""
        return {k: v * bankroll for k, v in self.allocations.items()}

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Optimal Portfolio Summary",
            f"========================",
            f"Total Exposure: {self.total_exposure:.1%}",
            f"Max Single Bet: {self.max_single_bet:.1%}",
            f"Expected Growth: {self.expected_growth_rate:.4%}",
            f"Solver Status: {self.solver_status}",
            f"",
            f"Allocations:"
        ]

        for opp_id, frac in sorted(self.allocations.items(), key=lambda x: -x[1]):
            if frac > 0.001:
                lines.append(f"  {opp_id}: {frac:.2%}")

        return "\n".join(lines)


class SimultaneousKellySolver:
    """
    Solver for simultaneous Kelly Criterion optimization.

    Uses convex optimization (cvxpy) to find the optimal bet sizing
    for multiple concurrent betting opportunities.

    Mathematical formulation:
    max E[log(1 + Σ f_i * b_i * I_i - Σ f_i)]

    Subject to:
    - Σ f_i ≤ max_exposure (total bankroll at risk)
    - f_i ≤ max_single_bet (no single bet too large)
    - f_i ≥ 0 (no short positions)

    Where:
    - f_i = fraction of bankroll to bet on opportunity i
    - b_i = net odds (decimal - 1)
    - I_i = indicator that bet i wins
    """

    def __init__(
        self,
        bankroll: float = None,
        max_exposure: float = None,
        max_single_bet: float = None,
        min_edge: float = None
    ):
        """
        Initialize Kelly solver.

        Args:
            bankroll: Total bankroll
            max_exposure: Maximum total exposure (fraction)
            max_single_bet: Maximum single bet (fraction)
            min_edge: Minimum edge to consider betting
        """
        config = CONFIG.optimization

        self.bankroll = bankroll or config.initial_bankroll
        self.max_exposure = max_exposure or config.max_exposure
        self.max_single_bet = max_single_bet or config.max_single_bet
        self.min_edge = min_edge or config.min_edge_threshold
        self.solver = config.solver

        if not CVXPY_AVAILABLE:
            logger.warning("cvxpy not available. Using simplified Kelly.")

    def optimize(
        self,
        opportunities: List[BettingOpportunity],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> OptimalPortfolio:
        """
        Find optimal bet sizing for all opportunities.

        Args:
            opportunities: List of betting opportunities
            correlation_matrix: Optional correlation between outcomes

        Returns:
            OptimalPortfolio with optimal allocations
        """
        # Filter to value opportunities only
        value_opps = [o for o in opportunities if o.is_value(self.min_edge)]

        if not value_opps:
            logger.info("No value opportunities found")
            return OptimalPortfolio(solver_status="no_value")

        logger.info(f"Optimizing {len(value_opps)} value opportunities")

        if CVXPY_AVAILABLE:
            return self._solve_cvxpy(value_opps, correlation_matrix)
        else:
            return self._solve_simplified(value_opps)

    def _solve_cvxpy(
        self,
        opportunities: List[BettingOpportunity],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> OptimalPortfolio:
        """Solve using cvxpy convex optimization."""
        n = len(opportunities)

        # Extract parameters
        probs = np.array([o.model_prob for o in opportunities])
        net_odds = np.array([o.decimal_odds - 1 for o in opportunities])

        # Decision variable: fraction of bankroll for each bet
        f = cp.Variable(n)

        # Objective: Maximize expected log growth
        # Approximation: E[log(W)] ≈ log(W_0) + Σ p_i * log(1 + f_i * b_i) + (1-p_i) * log(1 - f_i)
        # This is a concave function (sum of logs)

        # For numerical stability, we use the linear approximation for small bets
        # which is equivalent to maximizing expected value with variance penalty

        # Full formulation (more accurate but slower):
        growth_win = cp.sum(cp.multiply(probs, cp.log(1 + cp.multiply(f, net_odds))))
        growth_lose = cp.sum(cp.multiply(1 - probs, cp.log(1 - f)))
        objective = growth_win + growth_lose

        # Constraints
        constraints = [
            f >= 0,                          # No short positions
            f <= self.max_single_bet,        # Max single bet
            cp.sum(f) <= self.max_exposure,  # Max total exposure
            f <= 0.99                         # Cannot bet more than bankroll
        ]

        # Handle correlations if provided
        if correlation_matrix is not None:
            # Add correlation-aware constraints
            # For highly correlated bets, reduce combined exposure
            for i in range(n):
                for j in range(i + 1, n):
                    if correlation_matrix[i, j] > 0.5:
                        # If correlated, limit combined bet
                        constraints.append(
                            f[i] + f[j] <= self.max_single_bet * 1.5
                        )

        # Solve
        problem = cp.Problem(cp.Maximize(objective), constraints)

        try:
            problem.solve(solver=getattr(cp, self.solver, cp.ECOS))
            status = problem.status
        except Exception as e:
            logger.warning(f"Primary solver failed: {e}, trying SCS")
            try:
                problem.solve(solver=cp.SCS)
                status = problem.status
            except Exception as e2:
                logger.error(f"All solvers failed: {e2}")
                return OptimalPortfolio(solver_status="failed")

        if f.value is None:
            return OptimalPortfolio(solver_status="infeasible")

        # Extract results
        allocations = {}
        for i, opp in enumerate(opportunities):
            allocation = max(0.0, float(f.value[i]))
            if allocation > 0.001:  # Only include meaningful bets
                allocations[opp.opportunity_id] = allocation

        # Compute portfolio metrics
        total_exposure = sum(allocations.values())
        max_single = max(allocations.values()) if allocations else 0.0
        growth_rate = float(problem.value) if problem.value is not None else 0.0

        return OptimalPortfolio(
            allocations=allocations,
            expected_growth_rate=growth_rate,
            total_exposure=total_exposure,
            max_single_bet=max_single,
            solver_status=status
        )

    def _solve_simplified(
        self,
        opportunities: List[BettingOpportunity]
    ) -> OptimalPortfolio:
        """
        Simplified Kelly without cvxpy.

        Uses independent Kelly for each bet, then scales down
        if total exposure exceeds limit.
        """
        allocations = {}
        total = 0.0

        for opp in opportunities:
            # Independent Kelly formula
            # f* = (bp - q) / b
            # where b = net odds, p = prob of winning, q = 1 - p
            b = opp.decimal_odds - 1
            p = opp.model_prob
            q = 1 - p

            kelly_fraction = (b * p - q) / b

            # Apply half-Kelly for safety
            kelly_fraction = kelly_fraction / 2

            # Apply individual cap
            kelly_fraction = max(0, min(kelly_fraction, self.max_single_bet))

            if kelly_fraction > 0.001:
                allocations[opp.opportunity_id] = kelly_fraction
                total += kelly_fraction

        # Scale down if over exposure limit
        if total > self.max_exposure:
            scale = self.max_exposure / total
            allocations = {k: v * scale for k, v in allocations.items()}
            total = self.max_exposure

        max_single = max(allocations.values()) if allocations else 0.0

        # Approximate growth rate
        growth_rate = 0.0
        for opp in opportunities:
            if opp.opportunity_id in allocations:
                f = allocations[opp.opportunity_id]
                b = opp.decimal_odds - 1
                p = opp.model_prob
                growth_rate += p * np.log(1 + f * b) + (1 - p) * np.log(1 - f)

        return OptimalPortfolio(
            allocations=allocations,
            expected_growth_rate=growth_rate,
            total_exposure=total,
            max_single_bet=max_single,
            solver_status="simplified"
        )

    def compute_correlation_matrix(
        self,
        opportunities: List[BettingOpportunity]
    ) -> np.ndarray:
        """
        Compute correlation matrix between betting outcomes.

        Same-game parlays have high correlation.
        Same-team bets have moderate correlation.
        Independent games have zero correlation.
        """
        n = len(opportunities)
        corr = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                opp_i = opportunities[i]
                opp_j = opportunities[j]

                # Same game
                if opp_i.game_id == opp_j.game_id:
                    if opp_i.bet_type == opp_j.bet_type:
                        # Same market (e.g., both moneyline)
                        corr[i, j] = corr[j, i] = -0.9  # Opposite outcomes
                    else:
                        # Different markets (e.g., ML and total)
                        # ML favorite winning often means higher total
                        corr[i, j] = corr[j, i] = 0.3
                else:
                    # Different games - assume independent
                    corr[i, j] = corr[j, i] = 0.0

        return corr


def single_kelly_fraction(
    win_prob: float,
    decimal_odds: float,
    fractional_kelly: float = 0.5
) -> float:
    """
    Calculate single-bet Kelly fraction.

    Args:
        win_prob: Probability of winning
        decimal_odds: Decimal odds
        fractional_kelly: Fraction of full Kelly (default 0.5 = half Kelly)

    Returns:
        Optimal bet fraction
    """
    b = decimal_odds - 1  # Net odds
    p = win_prob
    q = 1 - p

    if b <= 0:
        return 0.0

    full_kelly = (b * p - q) / b

    return max(0, full_kelly * fractional_kelly)


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal."""
    if american_odds > 0:
        return 1 + american_odds / 100
    else:
        return 1 + 100 / abs(american_odds)


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1 / decimal_odds
