import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

FEATURE_COLUMNS = [
    "sleep",
    "mood",
    "exercise",
    "steps",
    "screentime",
    "study",
    "expense",
    "water",
    "stress",
]

DISPLAY_NAMES = {
    "sleep": "Sleep",
    "mood": "Mood",
    "exercise": "Exercise",
    "steps": "Steps",
    "screentime": "Screen Time",
    "study": "Study Hours",
    "expense": "Expense",
    "water": "Water Intake",
    "stress": "Stress",
}


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _safe_corr(series_a, series_b):
    series_a = _safe_numeric(series_a).dropna()
    series_b = _safe_numeric(series_b).dropna()
    if len(series_a) < 5 or len(series_b) < 5:
        return None
    if series_a.std() == 0 or series_b.std() == 0:
        return None
    try:
        corr, _ = pearsonr(series_a, series_b)
        return float(corr)
    except Exception:
        return None


def _safe_slope(series_a, series_b):
    series_a = _safe_numeric(series_a).dropna()
    series_b = _safe_numeric(series_b).dropna()
    if len(series_a) < 5 or len(series_b) < 5:
        return None
    try:
        slope, _, _, _, _ = linregress(series_a, series_b)
        return float(slope)
    except Exception:
        return None


def _strength_label(correlation):
    abs_corr = abs(correlation)
    if abs_corr >= 0.60:
        return "strong"
    if abs_corr >= 0.35:
        return "moderate"
    if abs_corr >= 0.20:
        return "mild"
    return None


def _relationship_phrase(feature, correlation):
    strength = _strength_label(correlation)
    if strength is None:
        return None
    direction = "positive" if correlation >= 0 else "negative"
    return f"{DISPLAY_NAMES[feature]} has a {strength} {direction} relationship with Productivity."


def _format_exercise(series):
    series = series.astype(str).str.lower()
    return series.replace({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}).astype(float)


def _prepare_df(history_df, raw_df=None):
    if history_df is not None and isinstance(history_df, pd.DataFrame) and len(history_df) >= 7 and "productivity" in history_df.columns:
        df = history_df.copy()
    elif raw_df is not None and isinstance(raw_df, pd.DataFrame) and "Productivity" in raw_df.columns:
        df = raw_df.copy()
        df = df.rename(columns={"Productivity": "productivity", "ScreenTime": "screentime", "StudyHours": "study"})
    else:
        return None

    if "exercise" in df.columns:
        df["exercise"] = _format_exercise(df["exercise"])

    for feature in FEATURE_COLUMNS:
        if feature in df.columns:
            df[feature] = _safe_numeric(df[feature])

    if "productivity" in df.columns:
        df["productivity"] = _safe_numeric(df["productivity"])

    return df


def _extract_top_features(model, feature_names):
    if model is None or not hasattr(model, "feature_importances_"):
        return None
    importances = np.array(model.feature_importances_)
    if len(importances) != len(feature_names):
        return None
    indices = np.argsort(importances)[::-1]
    return [(feature_names[i], float(importances[i])) for i in indices]


def generate_habit_insights(history_df, user_data, prediction, raw_df=None, model=None):
    df = _prepare_df(history_df, raw_df=raw_df)
    insights = []

    if df is None or "productivity" not in df.columns or df["productivity"].dropna().empty:
        return insights

    correlations = {}
    for feature in FEATURE_COLUMNS:
        if feature in df.columns:
            corr = _safe_corr(df[feature], df["productivity"])
            if corr is not None:
                correlations[feature] = corr

    # Build insights from feature-to-productivity relationships.
    sorted_features = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)
    for feature, corr in sorted_features:
        phrase = _relationship_phrase(feature, corr)
        if phrase:
            insights.append(phrase)
        if len(insights) >= 3:
            break

    # Add targeted habit insights if calculations support them.
    if "water" in df.columns and "mood" in df.columns:
        mood_corr = _safe_corr(df["water"], df["mood"])
        if mood_corr is not None and abs(mood_corr) >= 0.25:
            relation = "positive" if mood_corr > 0 else "negative"
            insights.append(f"Water Intake has a { _strength_label(mood_corr) } {relation} relationship with Mood.")

    if "exercise" in df.columns and "stress" in df.columns:
        stress_corr = _safe_corr(df["exercise"], df["stress"])
        if stress_corr is not None and stress_corr < -0.20:
            insights.append("Exercise appears to reduce Stress.")

    if "screentime" in df.columns and "productivity" in df.columns:
        screen_corr = correlations.get("screentime")
        if screen_corr is not None and screen_corr < -0.20:
            insights.append("Screen Time negatively affects Productivity.")

    # Add model-based feature importance insight when available.
    top_features = _extract_top_features(model, ["Sleep", "Mood", "Exercise", "Steps", "ScreenTime", "StudyHours", "Expense", "Water", "Stress"])
    if top_features is not None:
        top_name, top_value = top_features[0]
        if top_value >= 0.12:
            insights.append(f"{top_name} is a primary driver of predicted Productivity according to the model.")

    # only keep unique insights and 5 max
    final_insights = []
    for insight in insights:
        if insight not in final_insights:
            final_insights.append(insight)
        if len(final_insights) >= 5:
            break

    return final_insights
