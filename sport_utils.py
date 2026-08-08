import pandas as pd

def detect_sport(df: pd.DataFrame) -> str:
    """
    Auto-detects the sport type (Cricket, Football, Basketball, or General Sports)
    from column names and dataset values.
    """
    if df is None or df.empty:
        return "General Sports"

    cols = [str(c).lower() for c in df.columns]

    cricket_keywords = ["runs", "wickets", "overs", "strike_rate", "strike rate", "sr", "economy", "econ", "bowler", "batter", "batsman", "batting", "bowling", "ipl", "innings", "50s", "100s", "fours", "sixes"]
    football_keywords = ["goals", "assists", "clean_sheets", "clean sheets", "xg", "xa", "yellow_cards", "red_cards", "tackles", "interceptions", "passes", "position", "epl", "premier league", "shots", "saves"]
    basketball_keywords = ["pts", "points", "rebounds", "reb", "ast", "steals", "stl", "blocks", "blk", "3pm", "fg%", "ft%", "nba"]

    cricket_score = sum(1 for kw in cricket_keywords if any(kw in c for c in cols))
    football_score = sum(1 for kw in football_keywords if any(kw in c for c in cols))
    basketball_score = sum(1 for kw in basketball_keywords if any(kw in c for c in cols))

    # Also check string columns for role values if present
    for col in df.select_dtypes(include=["object", "string"]).columns:
        vals = df[col].dropna().astype(str).str.lower().head(50).tolist()
        if any(v in ["batter", "batsman", "bowler", "all-rounder", "wicket-keeper", "wicketkeeper"] for v in vals):
            cricket_score += 3
        if any(v in ["forward", "midfielder", "defender", "goalkeeper", "fw", "mf", "df", "gk"] for v in vals):
            football_score += 3
        if any(v in ["guard", "center", "power forward", "small forward", "pg", "sg", "sf", "pf", "c"] for v in vals):
            basketball_score += 3

    scores = {
        "Cricket": cricket_score,
        "Football": football_score,
        "Basketball": basketball_score,
    }

    best_sport = max(scores, key=scores.get)
    if scores[best_sport] > 0:
        return best_sport
    return "General Sports"


def get_sport_roles(sport_type: str) -> list[str]:
    """
    Returns relevant player roles/positions for the given sport.
    """
    if sport_type == "Cricket":
        return ["All Roles", "Batter", "Bowler", "All-rounder", "Wicketkeeper"]
    elif sport_type == "Football":
        return ["All Positions", "Forward / Attacker", "Midfielder", "Defender", "Goalkeeper"]
    elif sport_type == "Basketball":
        return ["All Positions", "Guard", "Forward", "Center"]
    else:
        return ["All Roles"]


def get_sport_questions(sport_type: str, selected_role: str = "All Roles") -> list[str]:
    """
    Returns dynamic preset sample questions based on sport type and selected role.
    """
    role = (selected_role or "All Roles").lower()

    if sport_type == "Cricket":
        if "batter" in role:
            return [
                "Who are the top 5 batters by total runs?",
                "Who has the highest strike rate among batters with over 200 runs?",
                "Who has scored the most 50s and 100s?",
                "Show top 5 batters by batting average and total fours & sixes.",
            ]
        elif "bowler" in role:
            return [
                "Who are the top 5 bowlers by total wickets?",
                "Who has the best bowling economy rate?",
                "Who are the top 5 bowlers with the lowest bowling average?",
                "Which bowler has taken the most 4-wicket or 5-wicket hauls?",
            ]
        elif "all-rounder" in role:
            return [
                "Who are the top 5 all-rounders based on runs and wickets?",
                "Show players with more than 150 runs and 10 wickets.",
                "Compare overall ratings of top all-rounders.",
            ]
        elif "wicketkeeper" in role:
            return [
                "Who are the top wicketkeepers by dismissals and catches?",
                "Compare batting performance and runs of top wicketkeepers.",
            ]
        else:
            return [
                "Who are the top 5 batters by total runs?",
                "Who are the top 5 bowlers by total wickets?",
                "Which team has the highest total runs scored?",
                "Show top 5 players with highest strike rate.",
                "Compare the top 2 batters head-to-head.",
            ]

    elif sport_type == "Football":
        if "forward" in role or "attacker" in role:
            return [
                "Who are the top 5 goal scorers?",
                "Who has the highest expected goals (xG)?",
                "Who has the highest goal contributions (Goals + Assists)?",
                "Show top 5 attackers with highest shot conversion accuracy.",
            ]
        elif "midfielder" in role:
            return [
                "Who are the top 5 players by total assists?",
                "Who created the most key passes and chances?",
                "Who has the highest pass completion percentage?",
                "Show top midfielders by goal contributions.",
            ]
        elif "defender" in role:
            return [
                "Who are the top defenders with the most tackles and interceptions?",
                "Who has won the most aerial and ground duels?",
                "Which defenders have contributed to the most clean sheets?",
            ]
        elif "goalkeeper" in role:
            return [
                "Which goalkeeper has the most clean sheets?",
                "Who has the highest save percentage among goalkeepers?",
                "Who made the most total saves this season?",
            ]
        else:
            return [
                "Who are the top 5 goal scorers?",
                "Who are the top 5 assist providers?",
                "Which player has the highest xG (Expected Goals)?",
                "Which goalkeeper has the most clean sheets?",
                "Compare the top 2 goal scorers head-to-head.",
            ]

    elif sport_type == "Basketball":
        if "guard" in role:
            return [
                "Who are the top 5 guards by total points?",
                "Who leads in 3-pointers made (3PM)?",
                "Who are the top assist leaders among guards?",
            ]
        elif "forward" in role:
            return [
                "Who are the top scoring forwards?",
                "Who has the best field goal percentage (FG%)?",
                "Who leads in total rebounds and steals?",
            ]
        elif "center" in role:
            return [
                "Who are the top shot blockers (BLK)?",
                "Who has the most offensive and defensive rebounds?",
            ]
        else:
            return [
                "Who are the top 5 scoring players (PTS)?",
                "Who has the highest field goal percentage?",
                "Who leads in assists and steals?",
                "Who are the top 5 rebounders?",
            ]

    else:
        return [
            "Who are the top 5 performing players?",
            "Summarize key stats across all teams.",
            "Show statistical summary and distribution of top metrics.",
            "Find the highest and lowest performing records in the dataset.",
        ]
