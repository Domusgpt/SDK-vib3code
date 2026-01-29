# Geometric Alpha: Complete System Guide

*A plain-English explanation of everything this system does, how it works, and how you'd use it.*

---

## Table of Contents

1. [What Is This?](#what-is-this)
2. [The Big Idea (In Plain English)](#the-big-idea)
3. [How Baseball Betting Normally Works](#how-baseball-betting-normally-works)
4. [What Makes This System Different](#what-makes-this-system-different)
5. [The Data: Where Information Comes From](#the-data)
6. [Feature Engineering: Turning Data Into Insights](#feature-engineering)
7. [The Prediction Model: Making Picks](#the-prediction-model)
8. [The Betting Engine: Deciding How Much To Bet](#the-betting-engine)
9. [The Safety Net: Validation and Trust](#the-safety-net)
10. [Paper Trading: Testing Without Real Money](#paper-trading)
11. [The Full Workflow: Start To Finish](#the-full-workflow)
12. [Every Module Explained](#every-module-explained)
13. [Glossary](#glossary)

---

## 1. What Is This? <a name="what-is-this"></a>

Geometric Alpha is a **baseball betting analysis system**. It uses publicly available pitch tracking data from MLB (called "Statcast") to try to predict game outcomes more accurately than the betting market.

**Think of it like this**: Every pitch thrown in MLB is tracked by cameras and radar. We know exactly how fast it went, how much it curved, where it crossed the plate, how fast it spun. This system takes all that data and asks: *"Can we use the physics and geometry of pitching to predict who wins better than the sportsbooks can?"*

**What it is NOT**:
- It is not a guaranteed way to make money
- It does not connect to any sportsbook or place bets automatically
- It has not been proven to be profitable yet
- It is a research tool that needs validation before real use

---

## 2. The Big Idea (In Plain English) <a name="the-big-idea"></a>

Most sports bettors and models look at stats like ERA (earned run average), batting average, and win percentage. These are useful but basic.

Geometric Alpha takes a different approach. Instead of asking "what did the pitcher do?", it asks **"what shape does the pitcher's skill take?"**

Here's what that means:

### The Tunnel

Imagine you're a batter. A pitcher throws a fastball and a changeup. If both pitches look identical for the first 20 feet of flight, you can't tell them apart until it's too late. That's called **tunneling**.

Most systems measure this crudely. We measure it using the **angle the pitches make from the batter's eye** — which is how the batter's brain actually processes it. If two pitches are less than 0.2 degrees apart (the limit of human vision), the batter literally cannot tell them apart.

### The Arsenal Shape

Instead of saying "this pitcher throws 93mph with a slider," we model the pitcher's entire pitch repertoire as a **3D shape**. Think of plotting every pitch in a space where the axes are velocity, spin, and movement. All those points form a shape (a "polytope").

- A **big shape** means diverse pitches (hard to predict what's coming)
- A **stable shape** means consistent mechanics (reliable pitcher)
- A **changing shape** means something is off (injury? fatigue? tipping?)

### The Umpire's Zone

The rulebook says the strike zone is a specific rectangle. In reality, every umpire has their own zone. Some call high strikes, some don't. Some have huge zones, some are tight.

We build a **geometric model of each umpire's actual zone** by looking at their recent called strikes. A bigger zone means fewer walks, fewer runs, and affects over/under bets.

### The Defensive Gaps

We divide the field into regions based on which fielder can reach each spot fastest. This reveals **seams** — gaps in the defense where hits are most likely to fall through. If a batter's spray chart (where they tend to hit the ball) lines up with defensive seams, they're more likely to get hits.

---

## 3. How Baseball Betting Normally Works <a name="how-baseball-betting-normally-works"></a>

Before understanding what our system does, you need to understand how betting works.

### Types of Bets

| Bet Type | What You're Betting On | Example |
|----------|----------------------|---------|
| **Moneyline** | Which team wins | Yankees -150 (bet $150 to win $100) |
| **Total (Over/Under)** | Combined runs scored | Over 8.5 (-110) |
| **Spread (Run Line)** | Win by how much | Yankees -1.5 (+120) |
| **Player Props** | Individual player stats | Judge Over 0.5 HR (+350) |

### How Odds Work

American odds tell you the payout:
- **-150** means bet $150 to win $100 (you're the favorite)
- **+150** means bet $100 to win $150 (you're the underdog)
- **-110** on both sides is standard (the sportsbook takes ~4.5% "juice")

### What Is "Edge"?

Edge is the difference between what you think will happen and what the odds say.

**Example**: You think the Yankees have a 60% chance of winning. The odds imply they have a 55% chance. Your edge is 5%. That means the bet is "good" — you're getting better odds than the true probability.

**The catch**: Finding real edge is extremely hard. Sportsbooks employ teams of analysts and use sophisticated models. Beating them consistently is rare.

---

## 4. What Makes This System Different <a name="what-makes-this-system-different"></a>

| Traditional Approach | Geometric Alpha |
|---------------------|-----------------|
| Pitcher ERA | Pitcher's arsenal "shape" in kinematic space |
| Batting average | Batter spray chart vs defensive Voronoi seams |
| Simple strike zone | Umpire-specific zone hull with expansion factor |
| Pitch speed | Angular divergence from batter's eye perspective |
| Independent bet sizing | Covariance-aware Kelly Criterion |
| Trust gut feel | Statistical validation before betting |

**The hypothesis**: By using geometric and physics-based features that traditional models ignore, we can find small edges the market doesn't price in.

**Honest caveat**: This hypothesis is unproven. The system exists to test it rigorously.

---

## 5. The Data: Where Information Comes From <a name="the-data"></a>

### Statcast (Pitch Tracking)

**Source**: MLB's pitch tracking system, accessed via `pybaseball` (free, public)

For every pitch thrown in MLB, we get:
- **Velocity**: How fast (e.g., 95.3 mph)
- **Spin Rate**: How fast it's spinning (e.g., 2,400 RPM)
- **Movement**: How much it curves horizontally and vertically
- **Release Point**: Where the pitcher releases the ball
- **Plate Location**: Where it crosses home plate
- **Acceleration**: How the ball decelerates through air

**Volume**: ~750,000 pitches per season, going back to 2015.

**Data file**: `data/statcast.py`

### Betting Odds

**Source**: The-Odds-API (requires API key, free tier available)

For every game, we get:
- Moneyline odds from multiple sportsbooks
- Total (over/under) lines and odds
- Spread (run line) odds
- Closing odds (final odds before game starts — critical for measuring skill)

**Data file**: `data/odds.py`

### What Happens When Data Is Missing

If pitch data has gaps (e.g., missing spin rate), the system uses **physics inversion** to estimate it. Here's the idea: if we know the ball's acceleration and velocity, we can work backwards through the Magnus force equation (the physics of spinning objects in air) to estimate the spin rate.

This is clever but introduces 5-15% error. The system documents this limitation.

---

## 6. Feature Engineering: Turning Data Into Insights <a name="feature-engineering"></a>

Raw data (pitch velocities, locations, etc.) isn't directly useful for predictions. We need to transform it into meaningful features. This is where the "geometric" part comes in.

### Feature 1: Pitch Tunneling Score

**File**: `features/tunneling.py`

**What it measures**: How well a pitcher disguises different pitch types.

**How it works**:
1. Take two consecutive pitches from the same pitcher
2. Calculate where each pitch is when it's 20 feet from the pitcher's hand
3. Measure the **angle** between them from the batter's perspective
4. Compare that to how far apart they end up at home plate
5. The ratio = tunnel score (higher = more deceptive)

**Why it matters**: A pitcher whose fastball and changeup look identical for the first third of flight is much harder to hit. The batter's decision to swing happens around 0.175 seconds after release — if they can't tell the difference by then, they're guessing.

**Key insight**: We measure angular divergence (how different pitches look from the batter's eye) rather than simple distance. This accounts for the fact that a 100mph fastball and 75mph curve are at very different positions at the same TIME, but can be at the same DISTANCE from the pitcher.

**Example**: A tunnel score of 3.0+ means the pitches diverge 3x more at the plate than at the tunnel point. That's elite deception.

### Feature 2: Arsenal Polytope

**File**: `features/arsenal.py`

**What it measures**: The overall "shape" and diversity of a pitcher's repertoire.

**How it works**:
1. Take a pitcher's last 500 pitches
2. Plot them in 8-dimensional space (velocity, spin, horizontal movement, vertical movement, release X, release Z, velocity-X, velocity-Z)
3. Wrap the tightest possible shape around all the points (convex hull)
4. Measure that shape's properties

**What we learn**:
- **Volume**: How much space the pitcher covers. Bigger = more diverse arsenal
- **Vertices**: How many "extremes" the pitcher has. More = harder to predict
- **Stability**: Is the shape consistent or changing? Changing might mean injury or fatigue
- **Dimensionality**: How many "real" dimensions the arsenal spans. A pitcher who throws everything at the same speed effectively has fewer dimensions

**Why it matters**: A pitcher with a large, stable polytope is throwing many different types of pitches with consistent mechanics. That's hard to hit against.

**Key safety feature**: We use a "rolling window" that only looks at past data when building these shapes. This prevents the system from accidentally using future information during backtesting (a common mistake called "look-ahead bias").

### Feature 3: Umpire Zone Hull

**File**: `features/umpire_hull.py`

**What it measures**: Each umpire's personal strike zone.

**How it works**:
1. Look at every "called strike" from an umpire over their last 5 games
2. Plot all those pitch locations
3. Draw a shape (convex hull) around them
4. Compare to the rulebook zone

**What we learn**:
- **Area**: How big their zone is (in square feet)
- **Expansion Factor**: How much bigger/smaller than the rulebook zone. 1.1 = 10% larger than the rules say
- **Centroid Drift**: Is their zone shifted left, right, up, or down?
- **Symmetry**: Do they call the same zone for righties and lefties?

**Why it matters**: A big zone means more called strikes, which means fewer walks, which means fewer runs. This directly affects over/under bets. An umpire with a 10% expanded zone could suppress scoring by 0.5-1 runs per game.

### Feature 4: Defensive Voronoi Coverage

**File**: `features/voronoi.py`

**What it measures**: How well the defense covers the field.

**How it works**:
1. Place each fielder at their position on the field
2. Divide the field into regions based on which fielder can reach each spot fastest
3. Account for sprint speed (faster players cover more ground)
4. Identify "seams" — borders between fielders' territories where nobody clearly has it

**What we learn**:
- **Total coverage**: What percentage of the field is well-covered
- **Seam density**: How much of the field is in no-man's-land
- **Largest gap**: The biggest hole in the defense
- **Infield vs outfield**: Where coverage is weakest

**Why it matters**: If a batter tends to hit the ball where defensive seams are, they'll get more hits. This helps predict batting outcomes beyond simple batting average.

---

## 7. The Prediction Model: Making Picks <a name="the-prediction-model"></a>

### The Basic Model (XGBoost)

**File**: `models/predictor.py`

All those features get fed into a machine learning model (XGBoost or LightGBM) that learns patterns from historical data.

**What it predicts**:
- **Home win probability**: 0-100% chance the home team wins
- **Expected total runs**: How many total runs to expect
- **Expected margin**: How many runs the home team wins/loses by

**How training works**:
1. Take 3-5 years of historical games
2. For each game, compute all the features above
3. Record the actual outcome (who won, how many runs)
4. Let XGBoost find the patterns
5. Test on a held-out set of games it's never seen

**Critical requirement**: The model must be **calibrated**. That means when it says "60% chance," the home team should actually win about 60% of the time. If it says 60% but they win 70%, the model is overconfident. We measure this with ECE (Expected Calibration Error) and require it to be under 5%.

### The Run Expectancy Model

**File**: `models/run_expectancy.py`

This is a secondary model that predicts runs from individual pitch interactions.

**Key concept — RE24 Matrix**: Baseball has 24 possible "base-out states" (e.g., runner on first with 1 out, bases loaded with 0 outs, etc.). For each state, we know the historically expected remaining runs in the inning. This model predicts how pitch outcomes (strike, ball, hit, etc.) change the expected runs.

**How tunnel scores affect it**: A pitch with a high tunnel score is more likely to result in a strikeout (batter is deceived). The model increases strikeout probability by ~20% for tunnel scores above 2.0.

**How umpire zones affect it**: An expanded umpire zone suppresses runs. The model applies a reduction factor: `runs * (1 - (expansion - 1) * 0.3)`.

### Voyage AI Embeddings (Optional Enhancement)

**File**: `integrations/voyage_embeddings.py`

This is an optional AI-powered addition. Voyage AI creates "embeddings" — numerical representations that capture the meaning of text.

**What it does**:
1. Describe each game in words: "Yankees vs Red Sox, Cole pitching, night game at Fenway, 72F"
2. Convert that description to a vector of numbers
3. Find historically similar games by comparing vectors
4. Use outcomes of similar games as a sanity check

**Why it helps**: Sometimes context matters in ways numbers don't capture. "Day game after a night game" or "rivalry game at a historic park" might affect outcomes in ways features miss. Embeddings can capture these patterns.

### Validated Predictor (Trust and Confidence)

**File**: `models/validated_predictor.py`

This wrapper adds two critical pieces that were missing:

**Confidence** (how certain the model is about THIS prediction):
- How similar is this game to games the model was trained on?
- Are all the features available, or are some missing?
- Is the model being asked to predict something outside its experience?

**Trust** (how reliable the model is in general):
- Is the model calibrated? (ECE check)
- How much training data was used?
- How recent is the training data?

If confidence is low, the system warns you. If trust is low, the system tells you not to bet.

---

## 8. The Betting Engine: Deciding How Much To Bet <a name="the-betting-engine"></a>

Having a prediction isn't enough. You also need to know how much to bet. Bet too much on a single game and one loss wipes you out. Bet too little and you don't make enough to matter.

### The Kelly Criterion

**File**: `optimization/kelly.py`

The Kelly Criterion is a mathematical formula developed by a Bell Labs scientist in 1956. It tells you the **mathematically optimal bet size** to maximize long-term growth.

**Simple version**:
```
Bet fraction = (edge) / (odds)
```

If you have a 5% edge and the odds are even money (2.0 decimal), Kelly says bet 5% of your bankroll.

**Why we use "Half Kelly"**: Full Kelly is aggressive. Small errors in your probability estimates lead to huge bets. Half Kelly (betting half the recommended amount) gives up some growth but dramatically reduces the chance of going broke.

### Covariance-Aware Kelly (The Upgrade)

**File**: `optimization/kelly.py`

The standard Kelly formula assumes each bet is independent. But in baseball betting, they're not.

**Example**: You bet "Yankees Moneyline" AND "Yankees game Over 8.5 runs." These are correlated — if the Yankees are blowing out their opponent (winning), there are probably more runs scored (over). Treating them as independent leads to over-betting on the Yankees game.

Our system builds a **covariance matrix** — a mathematical structure that captures how bets relate to each other:
- Same game, same side: Highly correlated (+0.85)
- Same game, different markets: Moderately correlated (+0.15 to +0.35)
- Different games, same team: Weakly correlated (+0.10)
- Different games, different teams: Independent (0.0)

The optimizer then solves for the best allocation across ALL bets simultaneously, accounting for these correlations.

### Empirical Kelly (The Fix)

**File**: `optimization/empirical_kelly.py`

The original system used **guessed correlations** (0.35 between moneyline and totals, for example). The fix: compute correlations from **actual historical bet outcomes**. When you feed in thousands of past games, the system measures the real correlation between "home team winning" and "game going over."

Every correlation now comes with:
- Where it came from (measured data vs fallback estimate)
- How many games it's based on
- How confident we are in it
- A warning if we don't have enough data

### Conditional Bets

**File**: `optimization/conditional.py`

Some bets depend on each other in complex ways:
- "Player X hits a home run" makes "Team Y wins" more likely
- "Player X Over 1.5 total bases" requires "Player X actually plays"
- "First 5 innings leader" strongly predicts "full game winner"

The system models these as a **dependency graph** — a network where each bet connects to the bets it affects. When one bet's outcome is known, it updates the probabilities of connected bets.

### Portfolio and Bankroll Management

**File**: `optimization/portfolio.py`

This tracks everything:
- **Current bankroll**: How much money you have
- **Pending bets**: Bets that haven't settled yet
- **Available funds**: Bankroll minus pending exposure
- **Performance history**: Every bet, its outcome, profit/loss
- **Risk of ruin**: Monte Carlo simulation of the probability of going broke

**Key metrics tracked**:
- **ROI (Return on Investment)**: Total profit divided by total wagered
- **CLV (Closing Line Value)**: Did you get better odds than the closing line? This is the #1 indicator of betting skill. If you consistently beat the closing line, you have a real edge. If you don't, you're probably losing long-term.
- **Max Drawdown**: The biggest peak-to-trough decline in your bankroll
- **Sharpe Ratio**: Risk-adjusted return (higher = better returns per unit of risk)

### Professional Refinements

**File**: `optimization/professional_refinements.py`

Concepts borrowed from professional betting operations:

- **CLV Tracker**: Monitors whether you're consistently beating the closing line
- **Steam Move Detector**: Detects when sharp (professional) books like Pinnacle move their lines, signaling new information
- **Calibration Optimizer**: Continuously checks if your model's probabilities match reality
- **Risk of Ruin Analysis**: Calculates the probability of going broke given your current strategy

---

## 9. The Safety Net: Validation and Trust <a name="the-safety-net"></a>

### The Trustworthy Betting System

**File**: `evaluation/trustworthy_system.py`

This is the guard rail that prevents you from betting with a bad model. It checks four things:

**Check 1: Probability Calibration**
- When the model says 60%, does it win ~60% of the time?
- Measured by ECE (Expected Calibration Error)
- Must be under 5% for betting
- If it fails: "DO NOT use for betting until fixed"

**Check 2: Empirical Correlations**
- Are bet correlations measured from real data or guessed?
- How many games is each estimate based on?
- If insufficient data: "Need more data for reliable correlations"

**Check 3: Data-Driven Trust**
- Trust is computed from: sample size (40%), statistical significance (25%), recency (20%), stability (15%)
- No more arbitrary "trust = 0.8" everywhere
- Trust below 0.4: "Treat as independent, ignore relationship"

**Check 4: Edge Validation**
- Is the profit statistically significant (not just luck)?
- Is CLV positive (beating the closing line)?
- Does the model beat a simple baseline (just using market odds)?
- If no edge: "Model does not beat market odds baseline"

### The Edge Validation Framework

**File**: `evaluation/trustworthy_system.py`

This answers the most important question: **"Do we actually have an edge, or are we fooling ourselves?"**

Tests performed:
1. **T-test on profits**: Is profit significantly different from zero? (p < 0.05)
2. **Bootstrap confidence intervals**: What's the range of likely ROI?
3. **CLV analysis**: Are we consistently getting better prices than closing lines?
4. **Model vs Market**: Does our model predict better than just using the odds as probabilities?

**Minimum requirements**:
- 500+ bets
- Profit p-value < 0.05
- Positive CLV
- Model outperforms market baseline

---

## 10. Paper Trading: Testing Without Real Money <a name="paper-trading"></a>

**File**: `paper_trading/simulator.py`

Before risking any real money, the system runs in "paper trading" mode — simulating bets without actually placing them.

### How It Works

1. **Start a session** with a virtual bankroll ($10,000)
2. **Each day**: Feed in today's predictions and market odds
3. **The system**: Runs Kelly optimization and "places" virtual bets
4. **When games finish**: Feed in outcomes to settle bets
5. **Track everything**: ROI, win rate, CLV, drawdown, calibration

### Validation Requirements

You must pass ALL of these before considering real money:

| Requirement | Threshold | Why |
|------------|-----------|-----|
| Total bets | 500+ | Statistical significance |
| Time period | 90+ days | Capture seasonal variance |
| Profit | p < 0.05 | Not just luck |
| CLV | > 0.5% | Consistently beating closing line |
| Calibration | ECE < 5% | Probabilities are accurate |
| Max drawdown | < 30% | Risk management works |

### What the Validation Report Tells You

```
VALIDATION STATUS
-----------------
Status: PASSED / IN PROGRESS / FAILED
Ready for Live: YES / NO

If FAILED:
  Blocking Issues:
  - Insufficient bets: 312 < 500 required
  - Profit not statistically significant (p=0.234)
  - Model not calibrated: ECE=0.073 (need < 0.050)

Recommendation: Continue paper trading. Need 188 more bets.
```

---

## 11. The Full Workflow: Start To Finish <a name="the-full-workflow"></a>

Here's exactly how you'd use this system:

### Phase 1: Setup (One Time)

```
1. Install Python packages (numpy, xgboost, etc.)
2. Get API keys:
   - Voyage AI (free): https://dash.voyageai.com/
   - The-Odds-API (free tier): https://the-odds-api.com/
3. Install pybaseball (free, no key needed)
```

### Phase 2: Train the Model

```
1. Load historical Statcast data (2021-2024)
   → System fetches ~3 million pitches

2. Compute geometric features for every game:
   → Tunnel scores for each pitcher
   → Arsenal polytopes for each pitcher
   → Umpire zone hulls
   → Defensive Voronoi coverage

3. Train XGBoost model with time-series validation:
   → Uses 2021-2023 for training
   → Tests on 2024 to verify

4. CHECK CALIBRATION:
   → ECE must be < 5%
   → If it fails, fix before continuing

5. Save trained model to disk
```

### Phase 3: Build Correlation Data

```
1. Load historical bet outcomes
2. Feed into EmpiricalCorrelationEstimator
3. Compute actual correlations (not guesses)
4. Verify sample sizes are sufficient
5. Save correlation estimator to disk
```

### Phase 4: Paper Trade (Minimum 3 Months)

```
EVERY DAY during baseball season:

Morning:
1. System computes features for today's probable pitchers
2. Generates win probabilities for each game
3. Fetches current betting odds
4. Calculates edge on each opportunity
5. Kelly optimizer recommends bet sizes

Evening:
1. Feed in game outcomes
2. System settles bets
3. Updates bankroll, CLV, calibration metrics

After 90+ days and 500+ bets:
→ Run validation report
→ Check all requirements pass
```

### Phase 5: Go Live (If Validated)

```
1. Start with 25% of intended bankroll
2. Monitor CLV daily (this is your leading indicator)
3. If CLV stays positive for 4+ weeks: scale up
4. If CLV turns negative: STOP and investigate
5. Re-validate quarterly
```

---

## 12. Every Module Explained <a name="every-module-explained"></a>

### Core Files

| File | Purpose | Plain English |
|------|---------|---------------|
| `core/engine.py` | Main orchestrator | The conductor that coordinates all the other parts |
| `core/pipeline.py` | Data processing pipeline | Moves data through each processing step in order |
| `config/settings.py` | Configuration | All the knobs and dials that control system behavior |
| `cli.py` | Command line interface | How you actually run the system from your terminal |

### Data Layer

| File | Purpose | Plain English |
|------|---------|---------------|
| `data/statcast.py` | Pitch tracking data | Downloads and cleans MLB pitch data (velocity, spin, location) |
| `data/odds.py` | Betting odds | Fetches current and historical betting lines from sportsbooks |
| `data/environment.py` | Game context | Weather, stadium, time of day (placeholder) |

### Feature Engineering

| File | Purpose | Plain English |
|------|---------|---------------|
| `features/tunneling.py` | Pitch deception | Measures how well a pitcher disguises different pitches from the batter's eye |
| `features/arsenal.py` | Pitcher repertoire shape | Models the geometric "shape" of a pitcher's entire pitch mix |
| `features/umpire_hull.py` | Umpire zone model | Builds a geometric model of each umpire's actual strike zone |
| `features/voronoi.py` | Defensive coverage | Maps the field into regions by which fielder can reach each spot fastest |

### Models

| File | Purpose | Plain English |
|------|---------|---------------|
| `models/predictor.py` | Win probability | Predicts which team wins and by how much |
| `models/run_expectancy.py` | Run scoring | Predicts how many runs to expect based on pitch interactions |
| `models/validated_predictor.py` | Predictions with confidence | Adds proper uncertainty (how sure we are) to every prediction |

### Optimization

| File | Purpose | Plain English |
|------|---------|---------------|
| `optimization/kelly.py` | Bet sizing | Calculates how much to bet on each opportunity |
| `optimization/empirical_kelly.py` | Improved bet sizing | Same but uses real measured correlations instead of guesses |
| `optimization/conditional.py` | Dependent bets | Handles bets that affect each other (same game parlays) |
| `optimization/portfolio.py` | Money management | Tracks bankroll, profit/loss, and risk metrics |
| `optimization/professional_refinements.py` | Pro techniques | CLV tracking, steam moves, calibration monitoring |

### Validation and Safety

| File | Purpose | Plain English |
|------|---------|---------------|
| `evaluation/trustworthy_system.py` | Trust verification | Tests if the model is accurate and the edge is real |
| `paper_trading/simulator.py` | Practice mode | Simulates betting without real money |
| `backtest/simulator.py` | Historical testing | Tests the system on past data to see if it would have worked |
| `backtest/metrics.py` | Performance stats | Calculates ROI, Sharpe ratio, win rate, drawdown |

### Training

| File | Purpose | Plain English |
|------|---------|---------------|
| `training/pipeline.py` | Model training | Complete workflow to train and validate the prediction model |
| `training/enhanced_pipeline.py` | Training with AI | Adds Voyage AI embeddings to the training process |

### Integrations

| File | Purpose | Plain English |
|------|---------|---------------|
| `integrations/voyage_embeddings.py` | AI similarity | Uses Voyage AI to find historically similar games |

---

## 13. Glossary <a name="glossary"></a>

| Term | Definition |
|------|-----------|
| **Angular Divergence** | The angle between two pitch paths as seen from the batter's perspective. Measured in degrees. |
| **Arsenal** | All the different pitch types a pitcher throws (fastball, slider, changeup, etc.) |
| **Backtest** | Testing a strategy on historical data to see how it would have performed |
| **Bankroll** | Total money available for betting |
| **Calibration** | Whether probability estimates are accurate. A "calibrated" model's 60% predictions should win 60% of the time. |
| **CLV (Closing Line Value)** | The difference between the odds you got and the final odds before the game. Positive CLV = you got a better price = skill indicator. |
| **Convex Hull** | The smallest shape that encloses a set of points. Like stretching a rubber band around pins on a board. |
| **Covariance Matrix** | A mathematical table showing how pairs of bets correlate with each other |
| **Decimal Odds** | Betting odds where 2.00 means you double your money. Your payout = stake x odds. |
| **Drawdown** | The largest drop from a peak bankroll to a subsequent low point. Shows worst-case pain. |
| **ECE (Expected Calibration Error)** | How far off a model's probabilities are from reality. Lower is better. Under 5% is the goal. |
| **Edge** | The difference between your estimated probability and the market's implied probability |
| **Embedding** | A numerical representation of text that captures its meaning. Similar texts have similar embeddings. |
| **Foveal Resolution** | The sharpest vision the human eye can achieve: about 0.2 degrees |
| **Half Kelly** | Betting half the amount Kelly Criterion recommends. Sacrifices some growth for much less risk. |
| **Juice (Vig)** | The sportsbook's built-in profit margin. -110 on both sides means ~4.5% juice. |
| **Kelly Criterion** | A mathematical formula for optimal bet sizing that maximizes long-term growth |
| **Look-Ahead Bias** | Accidentally using future information when testing on past data. This makes backtests look better than reality. |
| **Mahalanobis Distance** | A measure of how far a data point is from a distribution. Used to detect when a game is "unusual" compared to training data. |
| **Moneyline** | A bet on which team wins, regardless of score |
| **Over/Under (Total)** | A bet on whether the combined score will be above or below a number set by the sportsbook |
| **Paper Trading** | Simulating bets without real money to test a strategy |
| **Polytope** | A geometric shape in multiple dimensions. We use it to represent a pitcher's full range of pitches. |
| **RE24** | Run Expectancy based on 24 base-out states. Shows expected runs remaining in an inning from each state. |
| **Risk of Ruin** | The probability of losing your entire bankroll |
| **ROI (Return on Investment)** | Total profit divided by total amount wagered. 5% ROI means $5 profit per $100 bet. |
| **Seam** | In Voronoi analysis: the boundary between two fielders' coverage zones. Hits to seams are more likely to fall in. |
| **SGP (Same-Game Parlay)** | Multiple bets from the same game combined. All must win for the parlay to pay. |
| **Sharpe Ratio** | Risk-adjusted return. Higher = better returns per unit of risk taken. Above 1.0 is considered good. |
| **Spread (Run Line)** | A bet that includes a point handicap. Yankees -1.5 means they must win by 2+ runs. |
| **Statcast** | MLB's tracking system that records pitch velocity, spin, location, and trajectory for every pitch |
| **Steam Move** | A sudden, significant line movement at sharp sportsbooks, indicating new information or large professional wagers |
| **Tunnel / Tunneling** | When two different pitches travel along nearly identical paths early in flight, making them hard to distinguish |
| **Voronoi Diagram** | A division of space into regions based on which reference point is nearest. Used to model defensive coverage. |
| **XGBoost** | A machine learning algorithm that builds an ensemble of decision trees. Popular for tabular data prediction. |

---

*This document covers the complete Geometric Alpha system as of January 2026. For technical implementation details, see the source code and inline documentation.*
