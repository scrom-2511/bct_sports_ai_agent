import uuid
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain.tools import tool

matplotlib.use("Agg")

SAFE_BUILTINS = {
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "sorted": sorted,
    "range": range,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "round": round,
    "abs": abs,
    "enumerate": enumerate,
    "zip": zip,
    "print": print,
}


def _find_column(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """
    Finds a column name in df matching any keyword (case-insensitive or substring).
    """

    # First try exact matches
    for kw in keywords:
        for col in df.columns:
            if str(col).lower() == kw.lower():
                return col

    # Then try partial matches
    for kw in keywords:
        for col in df.columns:
            if kw.lower() in str(col).lower():
                return col

    return None


def make_tools(df: pd.DataFrame, chart_dir: Path):
    """
    Creates streamlined sports statistics tools for the AI Agent:
    - Load dataset info
    - Player stats lookup
    - Team stats summary
    - Match analysis / Head-to-Head
    - Rankings & Best player finder
    - Charts & Performance trends
    - Pandas code execution
    """

    @tool
    def dataset_info() -> str:
        """
        Returns dataset shape, column types, missing values count, and preview rows.
        """
        return (
            f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
            f"Columns & Data Types:\n{df.dtypes.to_string()}\n\n"
            f"Missing Values:\n{df.isnull().sum().to_string()}\n\n"
            f"Preview (First 3 Rows):\n{df.head(3).to_string()}"
        )

    @tool
    def missing_values() -> str:
        """
        Returns count of missing/null values per column.
        """
        return df.isnull().sum().to_string()

    @tool
    def summary_statistics() -> str:
        """
        Returns summary statistics (mean, min, max, std) for numerical/categorical fields.
        """
        return df.describe(include="all").to_string()

    @tool
    def player_stats_lookup(player_name: str) -> str:
        """
        Looks up detailed performance statistics for a specific player by name.
        """
        player_col = _find_column(df, ["player", "player_name", "athlete", "name", "batsman", "bowler", "driver", "skater"])
        if not player_col:
            str_cols = df.select_dtypes(include=["object", "string"]).columns
            player_col = str_cols[0] if len(str_cols) > 0 else None

        if not player_col:
            return f"Error: No player or name column found in dataset. Available columns: {list(df.columns)}"

        matches = df[df[player_col].astype(str).str.contains(player_name, case=False, na=False)]
        return matches.to_string() if not matches.empty else f"No player matching '{player_name}' found in column '{player_col}'."

    @tool
    def team_stats_summary(team_name: str) -> str:
        """
        Summarizes stats and metrics for a specific team.
        """
        team_col = _find_column(df, ["team", "team_name", "squad", "club", "franchise", "country", "nation"])
        if not team_col:
            return f"Error: No team column found in dataset. Available columns: {list(df.columns)}"

        team_df = df[df[team_col].astype(str).str.contains(team_name, case=False, na=False)]
        if team_df.empty:
            return f"No team matching '{team_name}' found in column '{team_col}'."

        return (
            f"--- Team Summary for {team_name} ({len(team_df)} records, Column: '{team_col}') ---\n\n"
            f"Aggregated Team Stats:\n{team_df.select_dtypes(include=[np.number]).describe().to_string()}"
        )

    @tool
    def match_analysis_tool(entity1: str, entity2: str) -> str:
        """
        Compares two players or two teams side-by-side for head-to-head match analysis.
        """
        results = []
        player_col = _find_column(df, ["player", "player_name", "athlete", "name", "batsman", "bowler", "driver"])
        team_col = _find_column(df, ["team", "team_name", "squad", "club", "franchise", "country"])

        if player_col:
            p_df = df[df[player_col].astype(str).str.contains(f"{entity1}|{entity2}", case=False, na=False)]
            if not p_df.empty:
                results.append("Player Head-to-Head Comparison:\n" + p_df.to_string())

        if team_col:
            t1 = df[df[team_col].astype(str).str.contains(entity1, case=False, na=False)]
            t2 = df[df[team_col].astype(str).str.contains(entity2, case=False, na=False)]
            if not t1.empty and not t2.empty:
                comp = pd.concat(
                    [t1.mean(numeric_only=True).rename(entity1), t2.mean(numeric_only=True).rename(entity2)],
                    axis=1,
                )
                results.append("Team Mean Metrics Comparison:\n" + comp.to_string())

        return "\n\n".join(results) if results else f"Could not find entities matching '{entity1}' and '{entity2}' in dataset."

    @tool
    def find_best_players(metric: str, top_n: int = 5, ascending: bool = False) -> str:
        """
        Finds rankings / top N players sorted by a specified column metric.
        """
        matched_col = None

        for c in df.columns:
            if metric.lower() in str(c).lower():
                matched_col = c
                break
        
        if not matched_col:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            return f"Metric '{metric}' not found. Available numeric columns: {numeric_cols}"

        sorted_df = df.sort_values(by=matched_col, ascending=ascending).head(top_n)

        player_col = _find_column(df, ["player", "player_name", "athlete", "name", "batsman", "bowler"])
        team_col = _find_column(df, ["team", "team_name", "squad", "club", "franchise"])
        pos_col = _find_column(df, ["position", "pos", "role"])

        show_cols = []

        for col in [player_col, team_col, pos_col, matched_col]:
            if col and col in df.columns:
                show_cols.append(col)
                
        if not show_cols:
            show_cols = list(df.columns[:3]) + [matched_col]

        return sorted_df[show_cols].to_string()

    @tool
    def run_pandas_code(code: str) -> str:
        """
        Executes pandas code on `df` and returns the result variable (for custom metrics & trend analysis).
        """
        local_vars = {"df": df, "pd": pd, "np": np}
        try:
            exec(code, {"__builtins__": SAFE_BUILTINS}, local_vars)
        except Exception as e:
            return f"Error executing code: {e}"

        if "result" not in local_vars:
            return "Error: Assign the final answer to a variable named `result`."

        res = local_vars["result"]
        return res.to_string() if isinstance(res, (pd.DataFrame, pd.Series)) else str(res)

    @tool
    def plot_chart(code: str) -> str:
        """
        Executes matplotlib/seaborn plotting code to display charts and performance trends.
        """
        plt.figure(figsize=(9, 5))
        sns.set_theme(style="darkgrid")
        local_vars = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}

        try:
            exec(code, {"__builtins__": SAFE_BUILTINS}, local_vars)
        except Exception as e:
            plt.close()
            return f"Error executing plotting code: {e}"

        chart_path = chart_dir / f"chart_{uuid.uuid4().hex[:8]}.png"
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()
        return f"CHART_SAVED:{chart_path}"

    return [
        dataset_info,
        missing_values,
        summary_statistics,
        player_stats_lookup,
        team_stats_summary,
        match_analysis_tool,
        find_best_players,
        run_pandas_code,
        plot_chart,
    ]


