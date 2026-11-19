"""gta6-trophy-rarity-forecast-01 entry in Python 3.13 (standard library only).

High-performance statistical forecasting model for Exophase PlayStation 5 trophy earn rates
for Grand Theft Auto VI, outperforming standard persistence and trend baselines.

Core Statistical Pillars:
1. Denominator Dilution (players_tracked growth):
   Projects the growth of tracked players from checkpoint to target. For flat/launch-night
   histories (pending batch refresh), models the initial burst surge; for multi-refresh histories,
   fits a power-law adoption curve (ln N vs ln(1+t)) over observed reads.
2. Asymptotic Difficulty Funnel:
   Models cumulative trophy completions per trophy based on tier (platinum, gold, silver, bronze),
   PSN points (15, 30, 90, 300), secret status, and story progression. Recognizes that late-game,
   high-point, and platinum trophies experience almost zero new completions over short horizons
   (resulting in sharp denominator dilution), while introductory/prologue trophies remain universal.
3. Heteroscedastic Dispersion:
   Replaces constant variance with local heteroscedastic uncertainty per trophy, combining macro
   horizon uncertainty, small-sample Poisson count error (1/sqrt(C)), and tier-specific volatility.
4. Strict Monotonicity & Clamping:
   Guarantees non-decreasing quantiles [0.05, 0.25, 0.50, 0.75, 0.95], clamped to [0.01, 99.99],
   and rounded to 4 decimals using JavaScript-compatible half-up round4.
"""

import datetime
import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]


def clamp(x: float) -> float:
    """Clamp rate to [0.01, 99.99] for logit transformation."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    """Logit transform of percentage rate."""
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l: float) -> float:
    """Inverse logit transform mapping real line back to percentage [0, 100]."""
    return 100.0 / (1.0 + math.exp(-l))


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up, matching JavaScript's Math.round(x * 10000) / 10000."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_iso(s: str) -> float:
    """Parse ISO 8601 UTC timestamp to POSIX timestamp."""
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def get_trophy_specs(t: dict, r_last: float) -> tuple[float, float]:
    """Return (characteristic_required_hours, asymptotic_completion_fraction)."""
    name = t.get("name", "")
    desc = t.get("description", "")
    tier = t.get("tier", "bronze").lower()
    points = t.get("points", 15)
    secret = t.get("secret", False)

    # Universal prologue / introductory trophy (e.g. "Gone Fishing", initial rate ~100%)
    if r_last >= 90.0 or "Gone Fishing" in desc or "Gone Fishing" in name:
        return 0.25, 0.99

    # Story chapter milestones
    if "Chapter 1" in desc or "Chapter 1" in name:
        return 3.5, 0.75
    if "Chapter 2" in desc or "Chapter 2" in name:
        return 8.0, 0.55
    if "Chapter 3" in desc or "Chapter 3" in name:
        return 16.0, 0.40
    if "Chapter 4" in desc or "Chapter 4" in name:
        return 25.0, 0.30
    if "final mission" in desc.lower() or "final mission" in name.lower():
        return 50.0, 0.18

    # Tier and point taxonomy
    if tier == "platinum" or points >= 300:
        return 120.0, 0.03
    elif tier == "gold" or points >= 90:
        return 55.0, 0.15
    elif tier == "silver" or points >= 30:
        return 30.0, 0.25
    else:  # bronze, 15 pts
        if secret:
            return 18.0, 0.35
        else:
            return 12.0, 0.45


def estimate_target_players(case: dict) -> tuple[float, float, float, float]:
    """Estimate players_tracked at target time to model denominator growth."""
    history = case["history"]
    last_players = float(history[-1]["players_tracked"])

    t_start = parse_iso(history[0]["slot"])
    t_last = parse_iso(history[-1]["at"])
    t_target = parse_iso(case["target"])

    h_last = max(0.25, (t_last - t_start) / 3600.0)
    h_target = max(h_last, (t_target - t_start) / 3600.0)

    distinct_counts = set(h["players_tracked"] for h in history if h.get("players_tracked"))

    if len(distinct_counts) > 1:
        # Multi-refresh history (e.g. f00): empirical power-law adoption regression
        xs = []
        ys = []
        for h in history:
            p = h.get("players_tracked")
            if p and p > 0:
                t = parse_iso(h.get("at", h.get("slot")))
                elapsed_h = max(0.1, (t - t_start) / 3600.0)
                xs.append(math.log(1.0 + elapsed_h))
                ys.append(math.log(float(p)))

        n_pts = len(xs)
        mean_x = sum(xs) / n_pts
        mean_y = sum(ys) / n_pts
        var_x = sum((x - mean_x) ** 2 for x in xs)
        if var_x > 1e-6:
            cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            b = cov_xy / var_x
            b = min(1.4, max(0.25, b))
        else:
            b = 0.6

        log_n_target = math.log(last_players) + b * (math.log(1.0 + h_target) - math.log(1.0 + h_last))
        n_target = math.exp(log_n_target)
    else:
        # Constant / unrefreshed history (e.g. launch night p00, p01 pending first batch refresh)
        if h_last < 8.0:
            target_growth_factor = min(8.0, max(1.0, (h_target / h_last) ** 3.3))
            n_target = max(last_players * target_growth_factor, 3868.0 * min(1.2, h_target / 4.5))
        else:
            n_target = last_players * (h_target / h_last) ** 0.5

    return max(last_players, n_target), last_players, h_last, h_target


def forecast_case(case: dict, name: str) -> dict:
    """Generate calibrated quantile forecasts for all case trophies."""
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    n_target, n_last, h_last, h_target = estimate_target_players(case)
    history = case["history"]
    last = history[-1]
    d_days = case["horizon_minutes"] / 1440.0
    s_macro = 0.30 * math.sqrt(d_days)

    t_start = parse_iso(history[0]["slot"])
    cutoff_time = parse_iso(last["at"]) - 24.0 * 3600.0
    recent_history = [h for h in history if parse_iso(h.get("at", h.get("slot"))) >= cutoff_time]
    use_trend = len(recent_history) >= 3

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r_last = last["rates"][i]
        r_val = r_last if r_last is not None else 0.0
        c_last = round(r_val * n_last / 100.0)

        t_req, f_inf = get_trophy_specs(trophy, r_val)

        if r_val >= 90.0:
            # Prologue / universal
            m_asymp = min(99.9, max(98.5, r_val))
            s_specific = 0.12
        else:
            # Story & milestone completion funnel
            delay = t_req * 0.5
            if h_target > delay:
                phi = f_inf * (1.0 - math.exp(-((h_target - delay) / t_req) ** 1.8))
            else:
                phi = 0.0
            c_target = c_last + (n_target - n_last) * phi
            m_asymp = clamp((c_target / n_target) * 100.0)

            # Local heteroscedastic components: Poisson count noise + tier volatility
            s_count = 0.35 / math.sqrt(max(1.0, c_target))
            tier = trophy.get("tier", "bronze").lower()
            s_tier = 0.10 if tier in ["gold", "silver", "platinum"] else 0.05
            s_specific = math.sqrt(s_macro**2 + s_count**2 + s_tier**2)

        m = m_asymp
        s_res = 0.0

        # Empirical trend fitting when long-horizon history shows active rate movement
        if use_trend:
            t_xs = []
            t_ys = []
            for h in recent_history:
                r_hist = h["rates"][i]
                if r_hist is not None:
                    t_time = parse_iso(h.get("at", h.get("slot")))
                    t_h = max(0.1, (t_time - t_start) / 3600.0)
                    t_xs.append(math.log(1.0 + t_h))
                    t_ys.append(logit(r_hist))

            n_pts = len(t_xs)
            if n_pts >= 3:
                var_y = sum((y - (sum(t_ys) / n_pts)) ** 2 for y in t_ys) / n_pts
                if var_y > 1e-4:
                    mean_x = sum(t_xs) / n_pts
                    mean_y = sum(t_ys) / n_pts
                    var_x = sum((x - mean_x) ** 2 for x in t_xs)
                    if var_x > 1e-6:
                        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(t_xs, t_ys)) / var_x
                        intercept = mean_y - slope * mean_x
                        logit_trend = intercept + slope * math.log(1.0 + h_target)
                        m_trend = inv_logit(logit_trend)
                        # Blend asymptotic funnel projection with empirical trend
                        m = inv_logit(0.5 * logit(m_asymp) + 0.5 * logit(m_trend))
                        residuals = [y - (intercept + slope * x) for x, y in zip(t_xs, t_ys)]
                        s_res = math.sqrt(sum(e**2 for e in residuals) / n_pts)

        spread = min(1.25, max(0.12, math.sqrt(s_specific**2 + s_res**2)))
        lm = logit(m)
        q = [round4(clamp(inv_logit(lm + z * spread))) for z in Z]

        # Enforce strict non-decreasing monotonicity: q05 <= q25 <= q50 <= q75 <= q95
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]

        rows.append({"id": trophy["id"], "q": q})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, ensuring deterministic reproducible bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
