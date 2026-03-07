# analyze.py
# Aggregates evaluation results and generates charts for the presentation.
# Outputs 4 PNGs and summary_stats.csv to data/results/.
#
# Run after evaluate.py has populated the evaluations table.
import sqlite3
import sys
from pathlib import Path

from config import DB_PATH, RESULTS_DIR


def _require_libs():
    try:
        import matplotlib
        import pandas
        import seaborn
        return pandas, matplotlib, seaborn
    except ImportError:
        print("ERROR: pip install matplotlib seaborn pandas")
        sys.exit(1)


class ResultsAnalyzer:
    def __init__(self, db_path: Path = DB_PATH):
        pd, _, _ = _require_libs()
        self.pd = pd
        self.conn = sqlite3.connect(db_path)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def load_evaluations(self) -> "pd.DataFrame":
        """Join evaluations, ai_responses, and questions into a flat DataFrame."""
        query = """
            SELECT
                e.eval_id,
                e.response_id,
                e.question_id,
                e.platform,
                e.visibility_score,
                e.top_match_score,
                e.coverage_gap_count,
                e.has_hallucination,
                q.target_population,
                q.target_sector,
                q.target_region,
                q.difficulty
            FROM evaluations e
            JOIN ai_responses r ON e.response_id = r.response_id
            JOIN questions q ON e.question_id = q.question_id
        """
        df = self.pd.read_sql_query(query, self.conn)
        return df

    def platform_summary(self, df: "pd.DataFrame") -> "pd.DataFrame":
        return (
            df.groupby("platform")
            .agg(
                avg_visibility=("visibility_score", "mean"),
                avg_top_match=("top_match_score", "mean"),
                hallucination_rate=("has_hallucination", "mean"),
                total_responses=("eval_id", "count"),
                avg_coverage_gaps=("coverage_gap_count", "mean"),
            )
            .round(3)
            .reset_index()
            .sort_values("avg_visibility", ascending=False)
        )

    def population_pivot(self, df: "pd.DataFrame") -> "pd.DataFrame":
        return df.pivot_table(
            index="target_population",
            columns="platform",
            values="visibility_score",
            aggfunc="mean",
        ).round(3)

    def region_summary(self, df: "pd.DataFrame") -> "pd.DataFrame":
        return (
            df.groupby("target_region")["visibility_score"]
            .mean()
            .round(3)
            .reset_index()
            .sort_values("visibility_score")
        )

    def sector_summary(self, df: "pd.DataFrame") -> "pd.DataFrame":
        return (
            df.groupby("target_sector")["visibility_score"]
            .mean()
            .round(3)
            .reset_index()
            .sort_values("visibility_score")
        )

    # ------------------------------------------------------------------ charts

    def _platform_comparison_chart(self, df: "pd.DataFrame") -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        summary = self.platform_summary(df)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("AI Platform Comparison — AiEO Visibility Challenge", fontsize=14, fontweight="bold")

        # Visibility score
        colors = sns.color_palette("Set2", len(summary))
        axes[0].barh(summary["platform"], summary["avg_visibility"], color=colors)
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel("Average Visibility Score (0–1)")
        axes[0].set_title("Visibility of Canadian Programs")
        for i, (val, platform) in enumerate(zip(summary["avg_visibility"], summary["platform"])):
            axes[0].text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=10)

        # Hallucination rate
        axes[1].barh(summary["platform"], summary["hallucination_rate"] * 100, color=colors)
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel("Hallucination Rate (%)")
        axes[1].set_title("Misinformation / Hallucination Rate")
        for i, (val, platform) in enumerate(zip(summary["hallucination_rate"], summary["platform"])):
            axes[1].text(val * 100 + 0.5, i, f"{val*100:.1f}%", va="center", fontsize=10)

        plt.tight_layout()
        path = RESULTS_DIR / "platform_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")

    def _population_heatmap_chart(self, df: "pd.DataFrame") -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        pivot = self.population_pivot(df)
        if pivot.empty:
            print("  Not enough data for population heatmap.")
            return

        fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.6)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(
            "Visibility Score by Population Group & Platform\n(1.0 = full coverage, 0.0 = invisible)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("AI Platform")
        ax.set_ylabel("Target Population")
        plt.tight_layout()
        path = RESULTS_DIR / "population_gaps.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")

    def _region_gap_chart(self, df: "pd.DataFrame") -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        region_df = self.region_summary(df)
        if region_df.empty:
            print("  Not enough data for region chart.")
            return

        fig, ax = plt.subplots(figsize=(10, max(5, len(region_df) * 0.5)))
        colors = ["#d73027" if v < 0.3 else "#fee08b" if v < 0.6 else "#1a9850"
                  for v in region_df["visibility_score"]]
        ax.barh(region_df["target_region"], region_df["visibility_score"], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Average Visibility Score (0–1)")
        ax.set_title("AI Visibility Gaps by Region\n(Red = Low coverage ≤ 0.3)", fontsize=12, fontweight="bold")
        ax.axvline(0.3, color="red", linestyle="--", alpha=0.5, label="Low threshold (0.3)")
        ax.axvline(0.6, color="orange", linestyle="--", alpha=0.5, label="Medium threshold (0.6)")
        ax.legend(fontsize=8)
        for i, (val, region) in enumerate(zip(region_df["visibility_score"], region_df["target_region"])):
            ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=9)
        plt.tight_layout()
        path = RESULTS_DIR / "region_gaps.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")

    def _sector_gap_chart(self, df: "pd.DataFrame") -> None:
        import matplotlib.pyplot as plt

        sector_df = self.sector_summary(df)
        if sector_df.empty:
            print("  Not enough data for sector chart.")
            return

        fig, ax = plt.subplots(figsize=(10, max(5, len(sector_df) * 0.5)))
        colors = ["#d73027" if v < 0.3 else "#fee08b" if v < 0.6 else "#1a9850"
                  for v in sector_df["visibility_score"]]
        ax.barh(sector_df["target_sector"], sector_df["visibility_score"], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Average Visibility Score (0–1)")
        ax.set_title("AI Visibility Gaps by Sector\n(Red = Low coverage ≤ 0.3)", fontsize=12, fontweight="bold")
        ax.axvline(0.3, color="red", linestyle="--", alpha=0.5)
        ax.axvline(0.6, color="orange", linestyle="--", alpha=0.5)
        for i, (val, sector) in enumerate(zip(sector_df["visibility_score"], sector_df["target_sector"])):
            ax.text(val + 0.01, i, f"{val:.2f}", va="center", fontsize=9)
        plt.tight_layout()
        path = RESULTS_DIR / "sector_gaps.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")

    def _save_summary_stats(self, df: "pd.DataFrame") -> None:
        platform_df = self.platform_summary(df)
        platform_df.to_csv(RESULTS_DIR / "summary_platform.csv", index=False)

        region_df = self.region_summary(df)
        region_df.to_csv(RESULTS_DIR / "summary_region.csv", index=False)

        sector_df = self.sector_summary(df)
        sector_df.to_csv(RESULTS_DIR / "summary_sector.csv", index=False)

        pop_pivot = self.population_pivot(df)
        pop_pivot.to_csv(RESULTS_DIR / "summary_population.csv")

        # Overall summary
        overall = {
            "total_responses": len(df),
            "total_questions_tested": df["question_id"].nunique(),
            "platforms_tested": df["platform"].nunique(),
            "overall_avg_visibility": round(df["visibility_score"].mean(), 3),
            "overall_hallucination_rate_pct": round(df["has_hallucination"].mean() * 100, 1),
            "lowest_visibility_region": region_df.iloc[0]["target_region"] if not region_df.empty else "N/A",
            "lowest_visibility_region_score": region_df.iloc[0]["visibility_score"] if not region_df.empty else 0,
            "lowest_visibility_sector": sector_df.iloc[0]["target_sector"] if not sector_df.empty else "N/A",
            "lowest_visibility_sector_score": sector_df.iloc[0]["visibility_score"] if not sector_df.empty else 0,
        }
        self.pd.DataFrame([overall]).to_csv(RESULTS_DIR / "summary_stats.csv", index=False)
        print(f"  Saved summary CSVs to {RESULTS_DIR}/")

    def run(self) -> None:
        df = self.load_evaluations()

        if df.empty:
            print("No evaluation data found. Run evaluate.py first.")
            return

        print(f"Loaded {len(df)} evaluated responses across {df['platform'].nunique()} platforms.")
        print("Generating charts...")
        self._platform_comparison_chart(df)
        self._population_heatmap_chart(df)
        self._region_gap_chart(df)
        self._sector_gap_chart(df)
        self._save_summary_stats(df)

        # Print quick summary to terminal
        print("\n=== QUICK SUMMARY ===")
        print(self.platform_summary(df).to_string(index=False))
        print(f"\nResults saved to {RESULTS_DIR}/")
        self.conn.close()


if __name__ == "__main__":
    ResultsAnalyzer().run()
