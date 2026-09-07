
"""
Task 1 - Team Discipline Analysis
FIFA World Cup 2026 Group Project

Analytic question:
Do teams that advanced to the knockout stage commit fewer fouls per match
on average than teams eliminated in the group stage?

This file is designed as a proper demo/placeholder.
If a real CSV file is not provided, it generates synthetic demo data.

IMPORTANT:
Synthetic demo results must NOT be used as final assignment evidence.
Replace the demo data with your actual FIFA World Cup 2026 dataset.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_DATA_FILE = Path("data/task1_match_level.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/task1")
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95
ALPHA = 1 - CONFIDENCE_LEVEL


def generate_demo_data(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic match-level data for development/testing only.

    Required columns:
        match_id
        team
        fouls
        advanced_to_knockout

    The values below are artificial and are not FIFA World Cup 2026 results.
    """
    rng = np.random.default_rng(seed)

    advanced_teams = [
        "Demo Advanced 01",
        "Demo Advanced 02",
        "Demo Advanced 03",
        "Demo Advanced 04",
        "Demo Advanced 05",
        "Demo Advanced 06",
        "Demo Advanced 07",
        "Demo Advanced 08",
    ]

    eliminated_teams = [
        "Demo Eliminated 01",
        "Demo Eliminated 02",
        "Demo Eliminated 03",
        "Demo Eliminated 04",
        "Demo Eliminated 05",
        "Demo Eliminated 06",
        "Demo Eliminated 07",
        "Demo Eliminated 08",
    ]

    rows = []
    match_id = 1

    # Synthetic group-stage style records.
    # Each demo team gets three matches so aggregation can be tested.
    for team in advanced_teams:
        for _ in range(3):
            fouls = int(np.clip(np.rint(rng.normal(loc=10.0, scale=2.0)), 4, 18))
            rows.append(
                {
                    "match_id": f"M{match_id:03d}",
                    "team": team,
                    "fouls": fouls,
                    "advanced_to_knockout": True,
                }
            )
            match_id += 1

    for team in eliminated_teams:
        for _ in range(3):
            fouls = int(np.clip(np.rint(rng.normal(loc=12.0, scale=2.3)), 4, 20))
            rows.append(
                {
                    "match_id": f"M{match_id:03d}",
                    "team": team,
                    "fouls": fouls,
                    "advanced_to_knockout": False,
                }
            )
            match_id += 1

    return pd.DataFrame(rows)


def load_match_data(csv_path: Path | None) -> Tuple[pd.DataFrame, bool]:
    """
    Load real match-level data if available.

    Returns:
        (dataframe, is_demo_data)
    """
    if csv_path is not None and csv_path.exists():
        df = pd.read_csv(csv_path)
        return df, False

    print(
        "\n[WARNING] Real Task 1 CSV was not found.\n"
        "Synthetic demo data will be used so the script can run.\n"
        "Do not use synthetic results in the final assignment.\n"
    )
    return generate_demo_data(), True


def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the expected match-level input data.
    """
    required_columns = {
        "match_id",
        "team",
        "fouls",
        "advanced_to_knockout",
    }

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(sorted(missing))
            + "\nExpected columns: "
            + ", ".join(sorted(required_columns))
        )

    cleaned = df.copy()

    cleaned["team"] = cleaned["team"].astype(str).str.strip()
    cleaned["match_id"] = cleaned["match_id"].astype(str).str.strip()
    cleaned["fouls"] = pd.to_numeric(cleaned["fouls"], errors="coerce")

    # Accept common representations of booleans.
    if cleaned["advanced_to_knockout"].dtype != bool:
        mapping = {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "1": True,
            "0": False,
            "advanced": True,
            "eliminated": False,
            "knockout": True,
            "group": False,
        }
        cleaned["advanced_to_knockout"] = (
            cleaned["advanced_to_knockout"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )

    before = len(cleaned)
    cleaned = cleaned.dropna(
        subset=["match_id", "team", "fouls", "advanced_to_knockout"]
    ).copy()

    cleaned = cleaned[cleaned["team"] != ""]
    cleaned = cleaned[cleaned["match_id"] != ""]
    cleaned = cleaned[cleaned["fouls"] >= 0]

    after = len(cleaned)
    removed = before - after

    if removed > 0:
        print(f"[INFO] Removed {removed} invalid/incomplete row(s).")

    if cleaned.empty:
        raise ValueError("No valid rows remain after cleaning.")

    cleaned["advanced_to_knockout"] = cleaned["advanced_to_knockout"].astype(bool)

    return cleaned


def build_team_level_dataset(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level records to one row per team.

    This prevents teams with more matches from being treated as multiple
    independent observations in the team-level comparison.
    """
    team_level = (
        match_df.groupby(["team", "advanced_to_knockout"], as_index=False)
        .agg(
            matches_played=("match_id", "nunique"),
            total_fouls=("fouls", "sum"),
        )
    )

    team_level["fouls_per_match"] = (
        team_level["total_fouls"] / team_level["matches_played"]
    )

    team_level["group"] = np.where(
        team_level["advanced_to_knockout"],
        "Advanced to knockout",
        "Eliminated in group stage",
    )

    return team_level.sort_values(["advanced_to_knockout", "team"]).reset_index(
        drop=True
    )


def descriptive_statistics(values: pd.Series) -> Dict[str, float]:
    """
    Return useful descriptive statistics for one group.
    """
    x = pd.Series(values, dtype="float64").dropna()

    if x.empty:
        raise ValueError("Cannot calculate descriptive statistics on empty data.")

    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std_dev": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
        "min": float(x.min()),
        "q1": float(x.quantile(0.25)),
        "q3": float(x.quantile(0.75)),
        "max": float(x.max()),
        "range": float(x.max() - x.min()),
    }


def mean_confidence_interval(
    values: pd.Series,
    confidence: float = CONFIDENCE_LEVEL,
) -> Tuple[float, float, float]:
    """
    Calculate a two-sided t confidence interval for a population mean.

    Returns:
        mean, lower_bound, upper_bound
    """
    x = pd.Series(values, dtype="float64").dropna().to_numpy()
    n = len(x)

    if n < 2:
        raise ValueError("At least 2 observations are required for a mean CI.")

    mean = float(np.mean(x))
    sem = float(stats.sem(x))
    t_critical = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    margin = t_critical * sem

    return mean, mean - margin, mean + margin


def welch_difference_ci(
    group_a: pd.Series,
    group_b: pd.Series,
    confidence: float = CONFIDENCE_LEVEL,
) -> Dict[str, float]:
    """
    Calculate a Welch confidence interval for mean(group_a) - mean(group_b).
    """
    a = pd.Series(group_a, dtype="float64").dropna().to_numpy()
    b = pd.Series(group_b, dtype="float64").dropna().to_numpy()

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Both groups require at least 2 observations.")

    mean_diff = float(np.mean(a) - np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    n_a = len(a)
    n_b = len(b)

    se_sq = var_a / n_a + var_b / n_b
    se = math.sqrt(se_sq)

    numerator = se_sq**2
    denominator = (
        ((var_a / n_a) ** 2) / (n_a - 1)
        + ((var_b / n_b) ** 2) / (n_b - 1)
    )
    df = numerator / denominator

    t_critical = float(stats.t.ppf((1 + confidence) / 2, df=df))
    margin = t_critical * se

    return {
        "mean_difference": mean_diff,
        "lower_ci": mean_diff - margin,
        "upper_ci": mean_diff + margin,
        "degrees_of_freedom": float(df),
        "standard_error": float(se),
    }


def shapiro_check(values: pd.Series, alpha: float = ALPHA) -> Dict[str, object]:
    """
    Run Shapiro-Wilk normality check when sample size permits.
    """
    x = pd.Series(values, dtype="float64").dropna().to_numpy()

    if len(x) < 3:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "passes_at_alpha": None,
            "note": "Shapiro-Wilk requires at least 3 observations.",
        }

    result = stats.shapiro(x)

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "passes_at_alpha": bool(result.pvalue >= alpha),
        "note": (
            "No strong evidence against normality."
            if result.pvalue >= alpha
            else "Normality assumption may be questionable."
        ),
    }


def iqr_outliers(values: pd.Series) -> pd.Series:
    """
    Identify possible outliers using the 1.5 x IQR rule.
    """
    x = pd.Series(values, dtype="float64").dropna()
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return x[(x < lower) | (x > upper)]


def run_inferential_tests(
    advanced: pd.Series,
    eliminated: pd.Series,
    alpha: float = ALPHA,
) -> Dict[str, object]:
    """
    Run assumption diagnostics and a one-sided Welch two-sample t-test.

    Hypotheses:
        H0: mu_advanced >= mu_eliminated
        H1: mu_advanced <  mu_eliminated

    Welch's test is used because it does not require equal population variances.
    """
    advanced = pd.Series(advanced, dtype="float64").dropna()
    eliminated = pd.Series(eliminated, dtype="float64").dropna()

    if len(advanced) < 2 or len(eliminated) < 2:
        raise ValueError("Both groups require at least 2 teams for the t-test.")

    shapiro_advanced = shapiro_check(advanced, alpha)
    shapiro_eliminated = shapiro_check(eliminated, alpha)

    levene = stats.levene(advanced, eliminated, center="median")

    t_test = stats.ttest_ind(
        advanced,
        eliminated,
        equal_var=False,
        alternative="less",
    )

    difference_ci = welch_difference_ci(
        advanced,
        eliminated,
        confidence=1 - alpha,
    )

    return {
        "shapiro_advanced": shapiro_advanced,
        "shapiro_eliminated": shapiro_eliminated,
        "levene_statistic": float(levene.statistic),
        "levene_p_value": float(levene.pvalue),
        "t_statistic": float(t_test.statistic),
        "p_value": float(t_test.pvalue),
        "reject_null": bool(t_test.pvalue < alpha),
        "difference_ci": difference_ci,
    }


def save_visualisations(
    team_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save a box plot and histogram comparison.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    advanced = team_df.loc[
        team_df["advanced_to_knockout"], "fouls_per_match"
    ]
    eliminated = team_df.loc[
        ~team_df["advanced_to_knockout"], "fouls_per_match"
    ]

    # Box plot
    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [advanced, eliminated],
        tick_labels=["Advanced", "Eliminated"],
        showmeans=True,
    )
    plt.ylabel("Fouls per match")
    plt.title("Task 1: Fouls per Match by Tournament Outcome")
    plt.tight_layout()
    plt.savefig(output_dir / "task1_boxplot.png", dpi=200)
    plt.close()

    # Histogram
    plt.figure(figsize=(8, 5))
    bins = max(5, min(10, int(np.sqrt(len(team_df))) + 1))
    plt.hist(advanced, bins=bins, alpha=0.6, label="Advanced")
    plt.hist(eliminated, bins=bins, alpha=0.6, label="Eliminated")
    plt.xlabel("Fouls per match")
    plt.ylabel("Number of teams")
    plt.title("Task 1: Distribution of Fouls per Match")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "task1_histogram.png", dpi=200)
    plt.close()


def save_results(
    team_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save processed data and summary tables.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    team_df.to_csv(output_dir / "task1_team_level_data.csv", index=False)
    summary_df.to_csv(output_dir / "task1_descriptive_statistics.csv", index=False)


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run_analysis(
    csv_path: Path | None = DEFAULT_DATA_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, object]:
    """
    Run the complete Task 1 workflow.
    """
    raw_df, is_demo = load_match_data(csv_path)
    match_df = validate_input_data(raw_df)
    team_df = build_team_level_dataset(match_df)

    advanced = team_df.loc[
        team_df["advanced_to_knockout"], "fouls_per_match"
    ]
    eliminated = team_df.loc[
        ~team_df["advanced_to_knockout"], "fouls_per_match"
    ]

    if advanced.empty or eliminated.empty:
        raise ValueError(
            "The dataset must contain both advanced and eliminated teams."
        )

    advanced_stats = descriptive_statistics(advanced)
    eliminated_stats = descriptive_statistics(eliminated)

    advanced_ci = mean_confidence_interval(advanced)
    eliminated_ci = mean_confidence_interval(eliminated)

    tests = run_inferential_tests(advanced, eliminated)

    summary_df = pd.DataFrame(
        [
            {
                "group": "Advanced to knockout",
                **advanced_stats,
                "ci_95_lower": advanced_ci[1],
                "ci_95_upper": advanced_ci[2],
            },
            {
                "group": "Eliminated in group stage",
                **eliminated_stats,
                "ci_95_lower": eliminated_ci[1],
                "ci_95_upper": eliminated_ci[2],
            },
        ]
    )

    save_results(team_df, summary_df, output_dir)
    save_visualisations(team_df, output_dir)

    print_section("TASK 1 - ANALYTIC QUESTION")
    print(
        "Do teams that advanced to the knockout stage commit fewer fouls "
        "per match on average than teams eliminated in the group stage?"
    )

    print_section("DATA STATUS")
    print(f"Data source mode: {'SYNTHETIC DEMO' if is_demo else 'REAL CSV'}")
    print(f"Valid match-level rows: {len(match_df)}")
    print(f"Team-level observations: {len(team_df)}")
    print(f"Advanced teams: {len(advanced)}")
    print(f"Eliminated teams: {len(eliminated)}")

    print_section("DESCRIPTIVE STATISTICS")
    print(summary_df.round(3).to_string(index=False))

    print_section("95% CONFIDENCE INTERVALS")
    print(
        f"Advanced mean: {advanced_ci[0]:.3f}, "
        f"95% CI [{advanced_ci[1]:.3f}, {advanced_ci[2]:.3f}]"
    )
    print(
        f"Eliminated mean: {eliminated_ci[0]:.3f}, "
        f"95% CI [{eliminated_ci[1]:.3f}, {eliminated_ci[2]:.3f}]"
    )

    diff_ci = tests["difference_ci"]
    print(
        "Mean difference (Advanced - Eliminated): "
        f"{diff_ci['mean_difference']:.3f}"
    )
    print(
        "95% CI for mean difference: "
        f"[{diff_ci['lower_ci']:.3f}, {diff_ci['upper_ci']:.3f}]"
    )

    print_section("ASSUMPTION DIAGNOSTICS")
    print(
        "Shapiro-Wilk, Advanced: "
        f"W={tests['shapiro_advanced']['statistic']:.4f}, "
        f"p={tests['shapiro_advanced']['p_value']:.4f}. "
        f"{tests['shapiro_advanced']['note']}"
    )
    print(
        "Shapiro-Wilk, Eliminated: "
        f"W={tests['shapiro_eliminated']['statistic']:.4f}, "
        f"p={tests['shapiro_eliminated']['p_value']:.4f}. "
        f"{tests['shapiro_eliminated']['note']}"
    )
    print(
        "Levene variance check: "
        f"stat={tests['levene_statistic']:.4f}, "
        f"p={tests['levene_p_value']:.4f}"
    )

    advanced_outliers = iqr_outliers(advanced)
    eliminated_outliers = iqr_outliers(eliminated)

    print(f"Possible Advanced-group IQR outliers: {len(advanced_outliers)}")
    print(f"Possible Eliminated-group IQR outliers: {len(eliminated_outliers)}")

    print_section("WELCH TWO-SAMPLE T-TEST")
    print("H0: mean_advanced >= mean_eliminated")
    print("H1: mean_advanced < mean_eliminated")
    print(f"Alpha: {ALPHA:.2f}")
    print(f"t statistic: {tests['t_statistic']:.4f}")
    print(f"one-sided p-value: {tests['p_value']:.4f}")

    if tests["reject_null"]:
        print(
            "Decision: Reject H0. The sample provides evidence that teams "
            "advancing to the knockout stage committed fewer fouls per match "
            "on average."
        )
    else:
        print(
            "Decision: Fail to reject H0. The sample does not provide "
            "sufficient evidence that teams advancing to the knockout stage "
            "committed fewer fouls per match on average."
        )

    if is_demo:
        print(
            "\n[DEMO NOTICE] The conclusion above is based on synthetic data "
            "and must not be reported as a FIFA World Cup 2026 finding."
        )

    print_section("OUTPUT FILES")
    print(output_dir / "task1_team_level_data.csv")
    print(output_dir / "task1_descriptive_statistics.csv")
    print(output_dir / "task1_boxplot.png")
    print(output_dir / "task1_histogram.png")

    return {
        "is_demo": is_demo,
        "match_df": match_df,
        "team_df": team_df,
        "summary_df": summary_df,
        "tests": tests,
    }


def run_self_tests() -> None:
    """
    Lightweight tests that can be run without pytest:

        python task1.py --self-test
    """
    print("Running Task 1 self-tests...")

    demo = generate_demo_data(seed=123)

    assert not demo.empty
    assert {
        "match_id",
        "team",
        "fouls",
        "advanced_to_knockout",
    }.issubset(demo.columns)

    cleaned = validate_input_data(demo)
    assert len(cleaned) == len(demo)
    assert cleaned["fouls"].ge(0).all()

    team_df = build_team_level_dataset(cleaned)
    assert team_df["team"].nunique() == len(team_df)
    assert (team_df["matches_played"] == 3).all()

    first_team = cleaned["team"].iloc[0]
    expected_mean = cleaned.loc[
        cleaned["team"] == first_team, "fouls"
    ].mean()
    actual_mean = team_df.loc[
        team_df["team"] == first_team, "fouls_per_match"
    ].iloc[0]
    assert np.isclose(expected_mean, actual_mean)

    advanced = team_df.loc[
        team_df["advanced_to_knockout"], "fouls_per_match"
    ]
    eliminated = team_df.loc[
        ~team_df["advanced_to_knockout"], "fouls_per_match"
    ]

    desc = descriptive_statistics(advanced)
    assert desc["n"] == len(advanced)
    assert np.isclose(desc["mean"], advanced.mean())

    mean, lower, upper = mean_confidence_interval(advanced)
    assert lower < mean < upper

    tests = run_inferential_tests(advanced, eliminated)
    assert 0.0 <= tests["p_value"] <= 1.0
    assert math.isfinite(tests["t_statistic"])

    diff_ci = tests["difference_ci"]
    assert diff_ci["lower_ci"] < diff_ci["upper_ci"]

    print("All self-tests passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 1: FIFA World Cup 2026 team discipline analysis."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help=(
            "Path to match-level CSV. Default: data/task1_match_level.csv. "
            "If missing, synthetic demo data is used."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated tables and plots.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in tests and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        run_self_tests()
        return

    run_analysis(csv_path=args.data, output_dir=args.output)


if __name__ == "__main__":
    main()
