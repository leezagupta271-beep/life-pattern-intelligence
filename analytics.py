import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime


def calculate_burnout_risk(user_data):
    """
    Calculate Burnout Risk Score (0-100%) and Category (Low, Moderate, High)
    based on Sleep, Stress, ScreenTime, StudyHours, Water, and Exercise.
    """
    stress = float(user_data.get("stress", 5))
    sleep = float(user_data.get("sleep", 7.0))
    screentime = float(user_data.get("screentime", 4.0))
    study = float(user_data.get("study", 3.0))
    water = float(user_data.get("water", 2.0))
    exercise_val = user_data.get("exercise", "No")
    is_exercise = 1 if (exercise_val == "Yes" or exercise_val == 1) else 0

    # Risk factor weights
    stress_factor = (stress / 10.0) * 35.0
    sleep_factor = max(0, (8.0 - sleep) / 8.0) * 25.0
    screen_factor = min(1.0, screentime / 10.0) * 15.0
    study_factor = min(1.0, study / 10.0) * 15.0 * (1.0 - (0.5 * is_exercise))
    water_factor = max(0, (3.0 - water) / 3.0) * 10.0

    risk_score = round(min(100.0, max(0.0, stress_factor + sleep_factor + screen_factor + study_factor + water_factor)), 1)

    if risk_score < 35.0:
        category = "Low Risk"
        color = "#22C55E"
    elif risk_score < 65.0:
        category = "Moderate Risk"
        color = "#F59E0B"
    else:
        category = "High Risk"
        color = "#FF5F6D"

    # Drivers list
    drivers = []
    if stress >= 7:
        drivers.append(f"High Stress Level ({stress}/10)")
    if sleep < 6.5:
        drivers.append(f"Sleep Deficit ({sleep}h logged vs 8h target)")
    if screentime > 5.0:
        drivers.append(f"Elevated Screen Time ({screentime}h)")
    if water < 2.0:
        drivers.append(f"Low Hydration ({water}L)")
    if not is_exercise:
        drivers.append("Lack of physical exercise")

    if not drivers:
        drivers.append("Balanced routine & healthy habit indicators")

    return {
        "score": risk_score,
        "category": category,
        "color": color,
        "drivers": drivers
    }


def calculate_health_score(user_data):
    """
    Compute a simple health score using the same habit factors already used elsewhere.
    """
    sleep = float(user_data.get("sleep", 7.0))
    stress = float(user_data.get("stress", 4.0))
    water = float(user_data.get("water", 2.0))
    study = float(user_data.get("study", 3.0))
    steps = float(user_data.get("steps", 8000))
    mood = float(user_data.get("mood", 7))
    exercise_val = user_data.get("exercise", "No")
    is_exercise = 1 if (exercise_val == "Yes" or str(exercise_val).lower() in ["yes", "1", "true"]) else 0

    score = (
        (sleep / 8.0) * 20.0
        + min(1.0, steps / 10000.0) * 20.0
        + min(1.0, water / 3.0) * 20.0
        + ((10.0 - stress) / 10.0) * 20.0
        + (mood / 10.0) * 10.0
        + (20.0 if is_exercise else 0.0)
    )
    return round(min(100.0, max(0.0, score)), 1)


def _encode_exercise_value(value):
    val = str(value).strip().lower()
    return 1.0 if val in ["yes", "true", "1", "y", "t"] else 0.0


def _predict_stress_for_sleep_change(history_df, current_sleep, current_stress, new_sleep):
    if history_df is None:
        return current_stress

    if not isinstance(history_df, pd.DataFrame):
        history_df = pd.DataFrame(history_df)

    if "sleep" not in history_df.columns or "stress" not in history_df.columns:
        return current_stress

    df = history_df.dropna(subset=["sleep", "stress"])
    if len(df) < 5 or df["sleep"].std() == 0:
        return current_stress

    try:
        slope, intercept, _, _, _ = stats.linregress(df["sleep"], df["stress"])
        estimated = slope * new_sleep + intercept
        return round(max(1.0, min(10.0, estimated)), 1)
    except Exception:
        return current_stress


def calculate_sleep_whatif_chain(user_data, prediction, history=None, raw_df=None, model=None, scaler=None):
    current_sleep = float(user_data.get("sleep", 7.0))
    sleep_plus = min(current_sleep + 2.0, 12.0)
    sleep_delta = round(sleep_plus - current_sleep, 1)

    current_pred = float(prediction) if prediction is not None else 0.0
    feature_cols = ["Sleep", "Mood", "Exercise", "Steps", "ScreenTime", "StudyHours", "Expense", "Water", "Stress"]
    exercise_val = _encode_exercise_value(user_data.get("exercise", "No"))
    model_input = pd.DataFrame([[
        sleep_plus,
        float(user_data.get("mood", 7)),
        exercise_val,
        float(user_data.get("steps", 0)),
        float(user_data.get("screentime", 0)),
        float(user_data.get("study", 0)),
        float(user_data.get("expense", 0)),
        float(user_data.get("water", 0)),
        float(user_data.get("stress", 0)),
    ]], columns=feature_cols)

    new_pred = current_pred
    if model is not None and scaler is not None:
        try:
            new_pred = float(model.predict(scaler.transform(model_input))[0])
        except Exception:
            new_pred = current_pred

    pred_pct_change = round(((new_pred - current_pred) / max(current_pred, 1.0)) * 100.0, 1)

    current_stress = float(user_data.get("stress", 4.0))
    new_stress = _predict_stress_for_sleep_change(history, current_sleep, current_stress, sleep_plus)
    stress_pct_change = round(((new_stress - current_stress) / max(current_stress, 1.0)) * 100.0, 1)

    current_health = calculate_health_score(user_data)
    projected_user = dict(user_data)
    projected_user["sleep"] = sleep_plus
    projected_user["stress"] = new_stress
    projected_health = calculate_health_score(projected_user)
    health_pct_change = round(((projected_health - current_health) / max(current_health, 1.0)) * 100.0, 1)

    current_burnout = calculate_burnout_risk(user_data)["score"]
    projected_burnout = calculate_burnout_risk(projected_user)["score"]
    burnout_delta = round(current_burnout - projected_burnout, 1)

    return {
        "sleep_delta": sleep_delta,
        "predicted_productivity_pct": pred_pct_change,
        "stress_pct_change": stress_pct_change,
        "health_pct_change": health_pct_change,
        "burnout_delta": burnout_delta,
        "new_prediction": round(new_pred, 1),
        "new_stress": new_stress,
        "current_health": current_health,
        "projected_health": projected_health,
        "current_burnout": current_burnout,
        "projected_burnout": projected_burnout,
    }


def calculate_habit_correlations(df):
    """
    Calculate Pearson correlation between each numerical habit and Productivity.
    """
    feature_cols = ["Sleep", "Mood", "Exercise", "Steps", "ScreenTime", "StudyHours", "Expense", "Water", "Stress"]
    valid_cols = [col for col in feature_cols if col in df.columns]

    correlations = {}
    if "Productivity" in df.columns:
        for col in valid_cols:
            corr_val, _ = stats.pearsonr(df[col].fillna(0), df["Productivity"].fillna(0))
            correlations[col] = round(float(corr_val), 3)

    return correlations


def calculate_feature_importances(model, feature_columns):
    """
    Extract feature importances from trained RandomForestRegressor model.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return {col: round(float(imp), 4) for col, imp in zip(feature_columns, importances)}
    else:
        # Fallback uniform weights
        return {col: round(1.0 / len(feature_columns), 4) for col in feature_columns}


def generate_dynamic_recommendations(user_data, prediction, hist_df=None, raw_df=None, model=None, scaler=None):
    """
    Generate fully dynamic recommendations derived from user metrics, historical data,
    health scores, burnout risk, and ML model simulations.
    """
    sleep = float(user_data.get("sleep", 7.5))
    stress = float(user_data.get("stress", 4.0))
    screentime = float(user_data.get("screentime", 4.0))
    water = float(user_data.get("water", 2.0))
    study = float(user_data.get("study", 3.0))
    steps = float(user_data.get("steps", 8000))
    mood = float(user_data.get("mood", 7))
    expense = float(user_data.get("expense", 50.0))
    exercise_val = user_data.get("exercise", "No")
    is_exercise = 1 if (exercise_val == "Yes" or exercise_val == 1 or str(exercise_val).lower() in ["yes", "1", "true"]) else 0

    if hist_df is not None and isinstance(hist_df, pd.DataFrame) and not hist_df.empty:
        sleep_avg = float(hist_df["sleep"].mean()) if "sleep" in hist_df.columns else sleep
        stress_avg = float(hist_df["stress"].mean()) if "stress" in hist_df.columns else stress
        screentime_avg = float(hist_df["screentime"].mean()) if "screentime" in hist_df.columns else screentime
        water_avg = float(hist_df["water"].mean()) if "water" in hist_df.columns else water
        study_avg = float(hist_df["study"].mean()) if "study" in hist_df.columns else study
        steps_avg = float(hist_df["steps"].mean()) if "steps" in hist_df.columns else steps
    else:
        sleep_avg = sleep
        stress_avg = stress
        screentime_avg = screentime
        water_avg = water
        study_avg = study
        steps_avg = steps

    pred_val = float(prediction) if prediction is not None else 85.0
    current_burnout = calculate_burnout_risk(user_data)["score"]
    current_health = calculate_health_score(user_data)

    feature_cols = ["Sleep", "Mood", "Exercise", "Steps", "ScreenTime", "StudyHours", "Expense", "Water", "Stress"]

    def _predict_sim(sl, md, ex, stp, scr, std, exp, wt, strss):
        if model is None or scaler is None:
            return pred_val
        try:
            ex_enc = 1.0 if (ex == "Yes" or ex == 1 or str(ex).lower() in ["yes", "1", "true"]) else 0.0
            input_df = pd.DataFrame([[sl, md, ex_enc, stp, scr, std, exp, wt, strss]], columns=feature_cols)
            return float(model.predict(scaler.transform(input_df))[0])
        except Exception:
            return pred_val

    recs = []

    # 1. Sleep recommendation
    if sleep < 7.5:
        target_sleep = 8.0
        deficit = round(target_sleep - sleep, 1)
        sim_pred = _predict_sim(target_sleep, mood, exercise_val, steps, screentime, study, expense, water, stress)
        delta_prod = round(sim_pred - pred_val, 1)

        proj_user = dict(user_data)
        proj_user["sleep"] = target_sleep
        proj_burnout = calculate_burnout_risk(proj_user)["score"]
        delta_burnout = round(current_burnout - proj_burnout, 1)

        if delta_prod > 0.5:
            expected = f"+{delta_prod}% Productivity"
        elif delta_burnout > 1.0:
            expected = f"-{delta_burnout}% Burnout Risk"
        else:
            expected = "+15% Focus"

        priority = "High Priority" if (sleep < 6.0 or current_burnout >= 50.0) else "Medium Priority"
        recs.append({
            "priority": priority,
            "title": f"Increase Sleep by {deficit}h",
            "text": f"Your logged sleep is {sleep:.1f}h (historical avg: {sleep_avg:.1f}h). Increasing sleep to {target_sleep}h improves cognitive recovery and focus.",
            "expected": expected,
            "score": 3 if priority == "High Priority" else 2
        })

    # 2. Stress reduction recommendation
    if stress >= 5:
        target_stress = max(2.0, stress - 2.0)
        sim_pred = _predict_sim(sleep, mood, exercise_val, steps, screentime, study, expense, water, target_stress)
        delta_prod = round(sim_pred - pred_val, 1)

        proj_user = dict(user_data)
        proj_user["stress"] = target_stress
        proj_burnout = calculate_burnout_risk(proj_user)["score"]
        delta_burnout = round(current_burnout - proj_burnout, 1)

        if delta_burnout > 1.0:
            expected = f"-{delta_burnout}% Burnout Risk"
        elif delta_prod > 0.5:
            expected = f"+{delta_prod}% Life Score"
        else:
            expected = "+10% Stress Relief"

        priority = "High Priority" if (stress >= 7 or current_burnout >= 55.0) else "Medium Priority"
        recs.append({
            "priority": priority,
            "title": "Reduce Stress & Mental Fatigue",
            "text": f"Current stress level is {int(stress)}/10 (historical avg: {stress_avg:.1f}/10). Incorporating 15 minutes of mindfulness or short breaks lowers physiological strain.",
            "expected": expected,
            "score": 3 if priority == "High Priority" else 2
        })

    # 3. Screen Time recommendation
    if screentime > 4.5:
        target_screen = max(2.0, screentime - 2.0)
        sim_pred = _predict_sim(sleep, mood, exercise_val, steps, target_screen, study, expense, water, stress)
        delta_prod = round(sim_pred - pred_val, 1)

        if delta_prod > 0.5:
            expected = f"+{delta_prod}% Productivity"
        else:
            expected = f"-{round(min(30.0, (screentime - 3.5) * 5.0), 1)}% Screen Fatigue"

        priority = "High Priority" if screentime >= 6.5 else "Medium Priority"
        recs.append({
            "priority": priority,
            "title": "Reduce Screen Time",
            "text": f"Logged screen time is {screentime:.1f}h (historical avg: {screentime_avg:.1f}h). Keeping screen time below 4.0h reduces cognitive eye strain and mental fatigue.",
            "expected": expected,
            "score": 3 if priority == "High Priority" else 2
        })

    # 4. Water intake recommendation
    if water < 2.5:
        target_water = 3.0
        deficit = round(target_water - water, 1)
        proj_user = dict(user_data)
        proj_user["water"] = target_water
        proj_health = calculate_health_score(proj_user)
        delta_health = round(proj_health - current_health, 1)

        expected = f"+{max(1.0, delta_health)} pts Health Score"
        priority = "High Priority" if water < 1.5 else "Medium Priority"
        recs.append({
            "priority": priority,
            "title": f"Increase Water Intake by {deficit}L",
            "text": f"Logged water is {water:.1f}L (target: 3.0L). Drinking {deficit}L more water supports daily metabolic rate and energy levels.",
            "expected": expected,
            "score": 3 if priority == "High Priority" else 2
        })

    # 5. Study hours / Focus recommendation
    if study < 3.5 or pred_val < 75.0:
        target_study = min(6.0, max(4.0, study + 1.5))
        deficit = round(target_study - study, 1)
        sim_pred = _predict_sim(sleep, mood, exercise_val, steps, screentime, target_study, expense, water, stress)
        delta_prod = round(sim_pred - pred_val, 1)

        expected = f"+{max(1.0, delta_prod)}% Focus Output"
        priority = "High Priority" if (study < 2.0 or pred_val < 70.0) else "Medium Priority"
        recs.append({
            "priority": priority,
            "title": "Enhance Deep Study Hours",
            "text": f"Logged study duration is {study:.1f}h (historical avg: {study_avg:.1f}h). Adding {deficit}h of focused study directly boosts your productivity score.",
            "expected": expected,
            "score": 3 if priority == "High Priority" else 2
        })

    # 6. Physical exercise & steps recommendation
    if steps < 8000 or not is_exercise:
        target_steps = max(10000, steps + 2000)
        proj_user = dict(user_data)
        proj_user["steps"] = target_steps
        proj_user["exercise"] = "Yes"
        proj_health = calculate_health_score(proj_user)
        delta_health = round(proj_health - current_health, 1)

        expected = f"+{max(1.0, delta_health)} pts Vitality"
        priority = "High Priority" if (steps < 4000 and not is_exercise) else "Low Priority"
        recs.append({
            "priority": priority,
            "title": "Increase Physical Activity",
            "text": f"Logged steps today: {int(steps):,} (target: 10,000). Adding a light 20-minute exercise or walk enhances cardiovascular stamina and mood.",
            "expected": expected,
            "score": 1
        })

    if not recs:
        recs.append({
            "priority": "Low Priority",
            "title": "Maintain Balanced Routine",
            "text": f"Your current habits are well balanced across sleep ({sleep:.1f}h), stress ({stress}/10), and screen time ({screentime:.1f}h). Continue this consistent schedule.",
            "expected": "Stable Life Score",
            "score": 1
        })

    recs.sort(key=lambda x: x["score"], reverse=True)
    return recs


def calculate_ai_insights(user_data, prediction, df, model):
    """
    Generate dynamic AI prediction insights:
    - Most Influential Habit
    - Weakest Habit
    - Strongest Habit
    - Improvement Suggestions
    - Projected 7-day productivity trend
    """
    feature_columns = ["Sleep", "Mood", "Exercise", "Steps", "ScreenTime", "StudyHours", "Expense", "Water", "Stress"]
    importances = calculate_feature_importances(model, feature_columns)

    most_influential = max(importances, key=importances.get)

    sleep = float(user_data.get("sleep", 7.5))
    stress = float(user_data.get("stress", 4))
    water = float(user_data.get("water", 2.0))
    screentime = float(user_data.get("screentime", 4.2))
    exercise_val = user_data.get("exercise", "No")

    strengths = []
    weaknesses = []

    if sleep >= 7.5:
        strengths.append("Optimal Sleep Duration")
    else:
        weaknesses.append("Insufficient Sleep")

    if stress <= 4:
        strengths.append("Well-managed Stress Level")
    else:
        weaknesses.append("Elevated Stress")

    if water >= 2.5:
        strengths.append("Great Hydration")
    else:
        weaknesses.append("Low Water Intake")

    if exercise_val == "Yes" or exercise_val == 1:
        strengths.append("Regular Physical Activity")
    else:
        weaknesses.append("No Logged Exercise")

    if screentime > 5.0:
        weaknesses.append("High Screen Time")

    strongest_habit = strengths[0] if strengths else "Consistent Tracking Routine"
    weakest_habit = weaknesses[0] if weaknesses else "Late Night Screen Time"

    dynamic_recs = generate_dynamic_recommendations(user_data, prediction, raw_df=df, model=model)
    suggestions = [r["text"] for r in dynamic_recs[:3]]

    base_pred = float(prediction) if prediction else 85.0
    np.random.seed(42)
    daily_variation = np.array([0, 1.2, -0.8, 2.1, 1.5, 3.0, 2.4])
    future_trend = np.round(np.clip(base_pred + daily_variation, 0, 100), 1).tolist()

    return {
        "most_influential": most_influential,
        "strongest_habit": strongest_habit,
        "weakest_habit": weakest_habit,
        "suggestions": suggestions,
        "future_trend": future_trend
    }


def generate_health_summary_cards(user_data, prediction):
    """
    Generate dynamic summary cards for UI tags based on real user data.
    """
    sleep = float(user_data.get("sleep", 7.5))
    stress = float(user_data.get("stress", 4))
    water = float(user_data.get("water", 2.0))
    exercise_val = user_data.get("exercise", "No")
    is_exercise = (exercise_val == "Yes" or exercise_val == 1)
    pred_val = float(prediction) if prediction else 88.0

    cards = []

    # Sleep card
    if sleep >= 7.5:
        cards.append({"title": "Excellent Sleep", "text": f"{sleep}h logged", "type": "success", "icon": "fa-moon"})
    elif sleep >= 6.0:
        cards.append({"title": "Moderate Sleep", "text": f"{sleep}h logged", "type": "warning", "icon": "fa-moon"})
    else:
        cards.append({"title": "Sleep Deficit", "text": f"{sleep}h logged", "type": "danger", "icon": "fa-bed"})

    # Hydration card
    if water >= 2.5:
        cards.append({"title": "Optimal Hydration", "text": f"{water}L daily", "type": "success", "icon": "fa-glass-water"})
    else:
        cards.append({"title": "Low Water Intake", "text": f"{water}L logged", "type": "warning", "icon": "fa-glass-water"})

    # Exercise card
    if is_exercise:
        cards.append({"title": "Healthy Routine", "text": "Active workout logged", "type": "success", "icon": "fa-person-running"})
    else:
        cards.append({"title": "Sedentary Routine", "text": "No exercise logged", "type": "info", "icon": "fa-couch"})

    # Productivity card
    if pred_val >= 85:
        cards.append({"title": "Peak Productivity", "text": f"Life Score {pred_val}", "type": "success", "icon": "fa-bolt"})
    elif pred_val >= 70:
        cards.append({"title": "Good Productivity", "text": f"Life Score {pred_val}", "type": "info", "icon": "fa-chart-line"})
    else:
        cards.append({"title": "Needs Focus", "text": f"Life Score {pred_val}", "type": "warning", "icon": "fa-bullseye"})

    return cards


def generate_habit_impact_scores(user_data):
    """
    Generate simple habit impact percentages for display.
    """
    sleep = float(user_data.get("sleep", 7.0))
    water = float(user_data.get("water", 2.0))
    stress = float(user_data.get("stress", 4.0))
    steps = float(user_data.get("steps", 0))
    mood = float(user_data.get("mood", 7))
    exercise_val = user_data.get("exercise", "No")
    is_exercise = str(exercise_val).strip().lower() in ["yes", "true", "1"]

    sleep_pct = round(min(100.0, (sleep / 8.0) * 100.0), 1)
    water_pct = round(min(100.0, (water / 3.0) * 100.0), 1)
    stress_pct = round(max(0.0, 100.0 - stress * 10.0), 1)
    exercise_pct = round(
        min(
            100.0,
            (25.0 if is_exercise else 5.0)
            + min(1.0, steps / 10000.0) * 55.0
            + min(1.0, mood / 10.0) * 20.0,
        ),
        1,
    )

    return {
        "sleep_pct": sleep_pct,
        "exercise_pct": exercise_pct,
        "water_pct": water_pct,
        "stress_pct": stress_pct,
    }


def calculate_radar_scores(user_data, hist_df=None, prediction=None, raw_df=None):
    """
    Compute 7 Life Balance Radar axis scores (0-100) from real user data.
    Returns user scores, benchmark from top-quartile of raw_df, and insights.
    """
    sleep = float(user_data.get('sleep', 7.5))
    water = float(user_data.get('water', 2.0))
    stress = float(user_data.get('stress', 4.0))
    steps = float(user_data.get('steps', 8000))
    study = float(user_data.get('study', 3.5))
    screentime = float(user_data.get('screentime', 4.2))
    mood = float(user_data.get('mood', 7))
    exercise_val = user_data.get('exercise', 'No')
    is_exercise = 1 if str(exercise_val).strip().lower() in ['yes', '1', 'true'] else 0
    pred_val = float(prediction) if prediction is not None else float(user_data.get('prediction', 85.0))

    axes = ['Sleep Quality', 'Hydration', 'Stress Control', 'Physical Activity', 'Productivity', 'Study & Focus', 'Digital Wellness']

    # --- User scores ---
    sleep_score    = round(min(100.0, (sleep / 9.0) * 100.0), 1)
    hydration_score = round(min(100.0, (water / 3.5) * 100.0), 1)
    stress_score   = round(max(0.0, 100.0 - stress * 10.0), 1)
    activity_score = round(min(100.0, (25.0 if is_exercise else 5.0) + min(steps / 10000.0, 1.0) * 55.0 + (mood / 10.0) * 20.0), 1)
    prod_score     = round(min(100.0, max(0.0, float(pred_val))), 1)
    study_score    = round(min(100.0, (study / 8.0) * 100.0), 1)
    digital_score  = round(max(0.0, 100.0 - (screentime / 10.0) * 100.0), 1)

    user_scores = [sleep_score, hydration_score, stress_score, activity_score, prod_score, study_score, digital_score]

    # --- Benchmark from top-quartile of raw_df ---
    benchmark = [88, 82, 75, 80, 88, 72, 78]  # fallback
    if raw_df is not None and isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        try:
            df_b = raw_df.copy()
            q75 = df_b['Productivity'].quantile(0.75)
            top = df_b[df_b['Productivity'] >= q75]
            if len(top) >= 10:
                ex_enc = top['Exercise'].apply(lambda v: 1.0 if str(v).strip().lower() in ['yes', '1', 'true', '1.0'] else (float(v) if str(v).replace('.','',1).isdigit() else 0.0))
                benchmark = [
                    round(min(100.0, (float(top['Sleep'].mean())  / 9.0) * 100.0), 1),
                    round(min(100.0, (float(top['Water'].mean())  / 3.5) * 100.0), 1),
                    round(max(0.0, 100.0 - float(top['Stress'].mean()) * 10.0), 1),
                    round(min(100.0, (25.0 * float(ex_enc.mean()) + min(float(top['Steps'].mean()) / 10000.0, 1.0) * 55.0 + 20.0)), 1),
                    round(min(100.0, float(top['Productivity'].mean())), 1),
                    round(min(100.0, (float(top['StudyHours'].mean()) / 8.0) * 100.0), 1),
                    round(max(0.0, 100.0 - (float(top['ScreenTime'].mean()) / 10.0) * 100.0), 1),
                ]
        except Exception:
            pass

    # --- Insights ---
    diffs = [u - b for u, b in zip(user_scores, benchmark)]
    strongest_idx = int(np.argmax(user_scores))
    weakest_idx   = int(np.argmin(user_scores))
    balance_score = round(float(np.mean(user_scores)), 1)

    return {
        'axes': axes,
        'user_scores': user_scores,
        'benchmark': benchmark,
        'strongest': axes[strongest_idx],
        'strongest_score': user_scores[strongest_idx],
        'weakest': axes[weakest_idx],
        'weakest_score': user_scores[weakest_idx],
        'balance_score': balance_score,
        'gaps': [round(b - u, 1) for u, b in zip(user_scores, benchmark)],
    }


def classify_lifestyle_cluster(user_data, raw_df=None):
    """
    K-Means (k=4) lifestyle clustering using Sleep, Stress, Productivity,
    Exercise, Water, StudyHours, Steps, ScreenTime.
    Returns cluster name, explanation, and characteristics.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler as _MMS

    sleep      = float(user_data.get('sleep', 7.5))
    stress     = float(user_data.get('stress', 4.0))
    water      = float(user_data.get('water', 2.0))
    study      = float(user_data.get('study', 3.5))
    steps      = float(user_data.get('steps', 8000))
    screentime = float(user_data.get('screentime', 4.2))
    mood       = float(user_data.get('mood', 7))
    exercise_val = user_data.get('exercise', 'No')
    exercise_enc = 1.0 if str(exercise_val).strip().lower() in ['yes', '1', 'true'] else 0.0
    prediction = float(user_data.get('prediction', 85.0))

    # Features used for clustering
    FEATURES = ['Sleep', 'Stress', 'Productivity', 'Exercise', 'Water', 'StudyHours', 'Steps', 'ScreenTime']
    user_row = [sleep, stress, prediction, exercise_enc, water, study, steps, screentime]

    # Default fallback labels if raw_df is missing
    fallback_clusters = {
        0: ('Balanced Routine',     'Consistent sleep, moderate stress and solid daily habits.',     ['Consistent Sleep', 'Moderate Stress', 'Steady Steps']),
        1: ('High Performer',       'High productivity, sufficient sleep and low stress.',            ['High Productivity', 'Good Sleep', 'Low Stress', 'Active']),
        2: ('Recovery Mode',        'Low sleep and high stress are reducing overall performance.',    ['Low Sleep', 'High Stress', 'Low Productivity']),
        3: ('Stressed Overachiever','High study hours but elevated stress and low sleep.',            ['High Study Hours', 'High Stress', 'Sleep Deficit']),
    }

    if raw_df is None or not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        # Rule-based fallback
        if prediction >= 85 and sleep >= 7.0 and stress <= 5:
            cid = 1
        elif stress >= 7 or sleep < 6:
            cid = 2
        elif study >= 5 and stress >= 6:
            cid = 3
        else:
            cid = 0
        name, expl, chars = fallback_clusters[cid]
        return {'cluster_id': cid, 'cluster_name': name, 'explanation': expl, 'characteristics': chars}

    try:
        df_c = raw_df.copy()
        # Encode exercise
        df_c['Exercise'] = df_c['Exercise'].apply(
            lambda v: 1.0 if str(v).strip().lower() in ['yes', '1', 'true', '1.0'] else
                      (float(v) if str(v).replace('.','',1).isdigit() else 0.0)
        )
        missing = [f for f in FEATURES if f not in df_c.columns]
        if missing:
            raise ValueError(f'Missing columns: {missing}')

        sample = df_c[FEATURES].dropna().sample(min(len(df_c), 5000), random_state=42)

        scaler_c = _MMS()
        sample_scaled = scaler_c.fit_transform(sample)

        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        km.fit(sample_scaled)
        labels = km.labels_

        # Characterise each cluster by centroid
        centroids = scaler_c.inverse_transform(km.cluster_centers_)
        centroid_df = pd.DataFrame(centroids, columns=FEATURES)

        # Sort clusters by Productivity centroid (highest = 0)
        prod_order = centroid_df['Productivity'].rank(ascending=False).astype(int) - 1
        # Map original cluster index → semantic label index
        label_map = {}
        for orig_idx in range(4):
            prod_rank = int(prod_order.iloc[orig_idx])
            sleep_c = float(centroid_df.loc[orig_idx, 'Sleep'])
            stress_c = float(centroid_df.loc[orig_idx, 'Stress'])
            study_c  = float(centroid_df.loc[orig_idx, 'StudyHours'])

            if prod_rank == 0:
                semantic = 1  # High Performer
            elif stress_c >= 6.5 and study_c >= 4.0:
                semantic = 3  # Stressed Overachiever
            elif sleep_c < 6.5 or stress_c >= 7.0:
                semantic = 2  # Recovery Mode
            else:
                semantic = 0  # Balanced Routine
            label_map[orig_idx] = semantic

        # Assign user to nearest cluster
        user_scaled = scaler_c.transform([user_row])
        user_cluster_orig = int(km.predict(user_scaled)[0])
        user_semantic = label_map[user_cluster_orig]

        name, expl_tmpl, _ = fallback_clusters[user_semantic]

        # Build dynamic explanation from centroid values
        c = centroid_df.loc[user_cluster_orig]
        sleep_label  = 'sufficient sleep'  if c['Sleep'] >= 7.0 else 'insufficient sleep'
        stress_label = 'low stress'        if c['Stress'] <= 4   else ('moderate stress' if c['Stress'] <= 6.5 else 'high stress')
        prod_label   = 'high productivity' if c['Productivity'] >= 80 else ('moderate productivity' if c['Productivity'] >= 65 else 'low productivity')
        active_label = 'regular exercise'  if c['Exercise'] >= 0.4 else 'minimal exercise'
        water_label  = 'good hydration'    if c['Water'] >= 2.5 else 'low water intake'
        study_label  = 'high study hours'  if c['StudyHours'] >= 4.0 else 'moderate study'
        screen_label = 'high screen time'  if c['ScreenTime'] >= 5.5 else 'controlled screen time'

        traits = [sleep_label, stress_label, prod_label, active_label, water_label]
        explanation = f"Your current data matches the {name} pattern: characterised by {', '.join(traits)}."

        # Characteristic pills
        chars = []
        if c['Sleep'] >= 7.0:    chars.append(f"Sleep: {c['Sleep']:.1f}h ✓")
        else:                     chars.append(f"Sleep: {c['Sleep']:.1f}h ⚠")
        if c['Stress'] <= 4:     chars.append(f"Stress: {c['Stress']:.1f}/10 ✓")
        else:                     chars.append(f"Stress: {c['Stress']:.1f}/10 ⚠")
        chars.append(f"Productivity: {c['Productivity']:.0f}")
        if c['Exercise'] >= 0.4: chars.append('Exercise: Active ✓')
        else:                     chars.append('Exercise: Minimal ⚠')
        chars.append(f"Screen: {c['ScreenTime']:.1f}h")

        return {
            'cluster_id': user_semantic,
            'cluster_name': name,
            'explanation': explanation,
            'characteristics': chars[:5],
        }

    except Exception as e:
        # Rule-based fallback
        if prediction >= 85 and sleep >= 7.0 and stress <= 5:
            cid = 1
        elif stress >= 7 or sleep < 6:
            cid = 2
        elif study >= 5 and stress >= 6:
            cid = 3
        else:
            cid = 0
        name, expl, chars = fallback_clusters[cid]
        return {'cluster_id': cid, 'cluster_name': name, 'explanation': expl, 'characteristics': chars}


def _ensure_date_column(df):
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _calculate_productivity_patterns(hist_df, raw_df=None):
    df_to_use = None
    if hist_df is not None:
        if isinstance(hist_df, list):
            hist_df = pd.DataFrame(hist_df)
        if isinstance(hist_df, pd.DataFrame) and not hist_df.empty:
            if "date" in hist_df.columns and "productivity" in hist_df.columns:
                df_to_use = hist_df.rename(columns={"date": "Date", "productivity": "Productivity"}).copy()
            elif "Date" in hist_df.columns and "Productivity" in hist_df.columns:
                df_to_use = hist_df.copy()

    if (df_to_use is None or df_to_use.empty or df_to_use["Productivity"].dropna().empty) and raw_df is not None and isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        df_to_use = raw_df.copy()

    if df_to_use is None or df_to_use.empty:
        return {
            "best_focus_hours": None,
            "best_day": None,
            "most_productive_week": None,
        }

    # Check for hourly/time-of-day data in Date column
    dates_raw = df_to_use["Date"].dropna()
    dates_parsed = pd.to_datetime(dates_raw, errors="coerce")

    has_time_info = False
    if not dates_parsed.dropna().empty:
        if (dates_parsed.dt.hour != 0).any() or (dates_parsed.dt.minute != 0).any():
            has_time_info = True
        elif any(":" in str(val) for val in dates_raw):
            has_time_info = True

    best_focus_hours = None
    if has_time_info:
        best_focus_hours = None
    else:
        best_focus_hours = None

    df_calc = df_to_use.copy()
    df_calc["dt"] = pd.to_datetime(df_calc["Date"], errors="coerce")
    df_calc["Productivity"] = pd.to_numeric(df_calc["Productivity"], errors="coerce")
    df_calc = df_calc.dropna(subset=["dt", "Productivity"])

    best_day = None
    most_productive_week = None

    if not df_calc.empty:
        df_calc["weekday"] = df_calc["dt"].dt.day_name()
        weekday_perf = df_calc.groupby("weekday")["Productivity"].mean()
        if not weekday_perf.empty:
            best_day = str(weekday_perf.idxmax())

        iso = df_calc["dt"].dt.isocalendar()
        df_calc["year_week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
        weekly_perf = df_calc.groupby("year_week")["Productivity"].mean()
        if not weekly_perf.empty:
            best_week_key = str(weekly_perf.idxmax())
            week_num = int(best_week_key.split("-W")[-1])
            most_productive_week = f"Week {week_num}"

    return {
        "best_focus_hours": best_focus_hours,
        "best_day": best_day,
        "most_productive_week": most_productive_week,
    }


def _encode_exercise_series(series):
    return series.astype(str).str.strip().str.lower().isin(["yes", "true", "1"]).astype(float)


def calculate_monthly_summary(history, user_data, prediction, model=None, scaler=None, year_month=None):
    if history:
        hist_df = pd.DataFrame(history)
    else:
        hist_df = pd.DataFrame([user_data])
        hist_df["productivity"] = [prediction]

    hist_df = _ensure_date_column(hist_df)
    if year_month is None:
        year_month = datetime.now().strftime("%Y-%m")

    if "date" in hist_df.columns and hist_df["date"].notna().any():
        hist_df = hist_df.copy()
        hist_df["year_month"] = hist_df["date"].dt.strftime("%Y-%m")
        monthly_df = hist_df[hist_df["year_month"] == year_month].copy()
    else:
        monthly_df = hist_df.copy()
        monthly_df["year_month"] = year_month

    if monthly_df.empty:
        monthly_df = hist_df.tail(30).copy()
        monthly_df["year_month"] = year_month

    numeric_cols = ["sleep", "mood", "steps", "screentime", "study", "expense", "water", "stress", "productivity"]
    for col in numeric_cols:
        if col in monthly_df.columns:
            monthly_df[col] = pd.to_numeric(monthly_df[col], errors="coerce").fillna(0)

    avg_sleep = float(monthly_df["sleep"].mean()) if "sleep" in monthly_df.columns else 0.0
    avg_mood = float(monthly_df["mood"].mean()) if "mood" in monthly_df.columns else 0.0
    avg_steps = float(monthly_df["steps"].mean()) if "steps" in monthly_df.columns else 0.0
    avg_screentime = float(monthly_df["screentime"].mean()) if "screentime" in monthly_df.columns else 0.0
    avg_study = float(monthly_df["study"].mean()) if "study" in monthly_df.columns else 0.0
    total_expense = float(monthly_df["expense"].sum()) if "expense" in monthly_df.columns else 0.0
    avg_water = float(monthly_df["water"].mean()) if "water" in monthly_df.columns else 0.0
    avg_stress = float(monthly_df["stress"].mean()) if "stress" in monthly_df.columns else 0.0
    avg_productivity = float(monthly_df["productivity"].mean()) if "productivity" in monthly_df.columns else float(prediction)

    exercise_pct = 0.0
    if "exercise" in monthly_df.columns and not monthly_df.empty:
        exercise_pct = float(_encode_exercise_series(monthly_df["exercise"]).mean() * 100.0)

    predicted_productivity = float(prediction)
    if model is not None and scaler is not None and not monthly_df.empty:
        model_input = pd.DataFrame([[
            avg_sleep,
            avg_mood,
            1.0 if exercise_pct >= 50.0 else 0.0,
            avg_steps,
            avg_screentime,
            avg_study,
            total_expense / max(1, len(monthly_df)),
            avg_water,
            avg_stress,
        ]], columns=["Sleep", "Mood", "Exercise", "Steps", "ScreenTime", "StudyHours", "Expense", "Water", "Stress"])
        try:
            predicted_productivity = float(model.predict(scaler.transform(model_input))[0])
        except Exception:
            predicted_productivity = float(prediction)

    return {
        "year_month": year_month,
        "avg_sleep": round(avg_sleep, 2),
        "avg_mood": round(avg_mood, 2),
        "exercise_pct": round(exercise_pct, 1),
        "avg_steps": round(avg_steps, 1),
        "avg_screentime": round(avg_screentime, 2),
        "avg_study": round(avg_study, 2),
        "total_expense": round(total_expense, 2),
        "avg_water": round(avg_water, 2),
        "avg_stress": round(avg_stress, 2),
        "avg_productivity": round(avg_productivity, 2),
        "predicted_productivity": round(predicted_productivity, 2),
    }


def calculate_monthly_forecast(history, user_data):
    if history:
        hist_df = pd.DataFrame(history)
    else:
        hist_df = pd.DataFrame([user_data])

    hist_df = _ensure_date_column(hist_df)
    fields = ["sleep", "mood", "steps", "screentime", "study", "expense", "water", "stress"]

    if "date" in hist_df.columns and hist_df["date"].notna().any():
        hist_df = hist_df.copy()
        hist_df["year_month"] = hist_df["date"].dt.strftime("%Y-%m")
        monthly = hist_df.groupby("year_month")[fields].mean().reset_index()
    else:
        monthly = pd.DataFrame([{field: float(user_data.get(field, 0)) for field in fields}])

    forecast = {}
    if len(monthly) >= 2:
        for field in fields:
            if field in monthly.columns:
                trend = monthly[field].iloc[-1] - monthly[field].iloc[-2]
                forecast[field] = float(max(0.0, monthly[field].iloc[-1] + trend))
            else:
                forecast[field] = float(user_data.get(field, 0))
    else:
        for field in fields:
            if field in monthly.columns:
                forecast[field] = float(monthly[field].iloc[-1])
            else:
                forecast[field] = float(user_data.get(field, 0))

    return {f"next_month_{field}": round(value, 2) for field, value in forecast.items()}


def calculate_dashboard_analytics(history, user_data, prediction, raw_df, model, scaler):
    """
    Perform complete data science calculations across user historical predictions.
    """
    if history:
        hist_df = pd.DataFrame(history)
    else:
        hist_df = pd.DataFrame([user_data])
        hist_df['productivity'] = [prediction]

    # Convert numeric fields
    numeric_cols = ['sleep', 'mood', 'steps', 'screentime', 'study', 'expense', 'water', 'stress', 'productivity']
    for col in numeric_cols:
        if col in hist_df.columns:
            hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce').fillna(0)

    # Today's inputs
    sleep = float(user_data.get('sleep', 7.5))
    mood = int(user_data.get('mood', 7))
    exercise_val = user_data.get('exercise', 'No')
    is_exercise = 1 if (exercise_val == 'Yes' or exercise_val == 1) else 0
    steps = int(user_data.get('steps', 8000))
    screentime = float(user_data.get('screentime', 4.0))
    study = float(user_data.get('study', 3.0))
    expense = float(user_data.get('expense', 50.0))
    water = float(user_data.get('water', 2.0))
    stress = int(user_data.get('stress', 4))
    pred_val = float(prediction) if prediction else 85.0

    # Historical averages
    sleep_avg = float(hist_df['sleep'].mean())
    mood_avg = float(hist_df['mood'].mean())
    steps_avg = float(hist_df['steps'].mean())
    screentime_avg = float(hist_df['screentime'].mean())
    study_avg = float(hist_df['study'].mean())
    expense_avg = float(hist_df['expense'].mean())
    water_avg = float(hist_df['water'].mean())
    stress_avg = float(hist_df['stress'].mean())
    prod_avg = float(hist_df['productivity'].mean())

    def pct_trend_str(today, avg, invert=False):
        if avg == 0:
            return "Stable"
        diff = ((today - avg) / avg) * 100
        direction = "up" if diff >= 0 else "down"
        arrow = "▲" if diff >= 0 else "▼"
        if invert:
            is_good = diff <= 0
        else:
            is_good = diff >= 0
        status = "Good" if is_good else "Focus"
        return f"{arrow} {abs(diff):.1f}% vs avg"

    sleep_trend = pct_trend_str(sleep, sleep_avg)
    mood_trend = pct_trend_str(mood, mood_avg)
    screentime_trend = pct_trend_str(screentime, screentime_avg, invert=True)
    steps_trend = pct_trend_str(steps, steps_avg)
    expense_trend = pct_trend_str(expense, expense_avg, invert=True)

    # Mood emoji
    if mood >= 9:
        mood_emoji = "😀"
    elif mood >= 7:
        mood_emoji = "🙂"
    elif mood >= 5:
        mood_emoji = "😐"
    elif mood >= 3:
        mood_emoji = "😕"
    else:
        mood_emoji = "😢"

    steps_goal_pct = min(100.0, (steps / 10000.0) * 100)
    water_goal_pct = min(100.0, (water / 3.0) * 100)

    # Finance
    expense_weekly = float(hist_df['expense'].tail(7).sum())
    expense_monthly = float(hist_df['expense'].tail(30).sum())
    monthly_budget = 2000.0
    budget_remaining = max(0.0, monthly_budget - expense_monthly)
    savings_val = budget_remaining
    savings_goal = 10000.0
    monthly_savings = budget_remaining
    yearly_savings = monthly_savings * 12.0
    savings_goal_pct = min(100, int((yearly_savings / savings_goal) * 100)) if savings_goal else 0
    months_to_goal = int(round(savings_goal / max(1.0, monthly_savings))) if monthly_savings > 0 else None

    # Health score
    health_score = (sleep/8.0 * 20) + (min(1.0, steps/10000.0) * 20) + (min(1.0, water/3.0) * 20) + ((10 - stress)/10.0 * 20) + (mood/10.0 * 10) + (20 if is_exercise else 0)
    health_score = round(min(100.0, max(0.0, health_score)), 1)

    hist_health_scores = []
    for _, row in hist_df.iterrows():
        h_ex = 1 if str(row.get('exercise')).lower() in ['yes', '1', 'true'] else 0
        h_score = (float(row['sleep'])/8.0 * 20) + (min(1.0, float(row['steps'])/10000.0) * 20) + (min(1.0, float(row['water'])/3.0) * 20) + ((10 - float(row['stress']))/10.0 * 20) + (float(row['mood'])/10.0 * 10) + (20 if h_ex else 0)
        hist_health_scores.append(min(100.0, max(0.0, h_score)))
    health_avg = np.mean(hist_health_scores)
    health_diff = health_score - health_avg
    health_score_trend = f"{'+' if health_diff >= 0 else ''}{health_diff:.1f} pts vs avg"

    pattern_summary = _calculate_productivity_patterns(hist_df, raw_df=raw_df)

    # Goal Success
    study_prob = min(99, max(10, int((study_avg / 4.0) * 100)))
    study_progress = min(100, int((study_avg / 4.0) * 100))
    save_prob = min(99, max(10, int((yearly_savings / savings_goal) * 100)))
    save_progress = min(100, int((yearly_savings / savings_goal) * 100))
    marathon_prob = min(99, max(10, int((steps_avg / 10000.0) * 100)))
    marathon_progress = min(100, int((steps_avg / 10000.0) * 100))

    goals = [
        {
            "name": "Read 20 Books",
            "probability": f"{study_prob}% Probability",
            "progress": study_progress,
            "completion_text": f"Est. Completion: Nov 15 ({study_avg:.1f}h/day study avg)"
        },
        {
            "name": "Save $10k",
            "probability": f"{save_prob}% Probability",
            "progress": save_progress,
            "completion_text": f"Est. Completion: {months_to_goal} months" if months_to_goal else "Needs more savings focus"
        },
        {
            "name": "Run a Marathon",
            "probability": f"{marathon_prob}% Probability",
            "progress": marathon_progress,
            "completion_text": f"Steps Avg: {int(steps_avg):,} / 10,000 target"
        }
    ]

    burnout = calculate_burnout_risk(user_data)

    # What-If predictions
    feature_cols = ["Sleep", "Mood", "Exercise", "Steps", "ScreenTime", "StudyHours", "Expense", "Water", "Stress"]
    def predict_sim(sl, md, ex, stp, scr, std, exp, wt, strss):
        ex_encoded = 1 if str(ex).lower() in ['yes', '1', 'true'] else 0
        input_df = pd.DataFrame([[sl, md, ex_encoded, stp, scr, std, exp, wt, strss]], columns=feature_cols)
        scaled_input = scaler.transform(input_df)
        return float(model.predict(scaled_input)[0])

    sim_opt_sleep = predict_sim(8.0, mood, exercise_val, steps, screentime, study, expense, water, stress)
    sim_opt_screen = predict_sim(sleep, mood, exercise_val, steps, 2.0, study, expense, water, stress)
    sim_opt_both = predict_sim(8.0, min(10, mood+1), exercise_val, max(steps, 10000), 2.0, study, expense, water, max(1, stress-2))

    whatif = {
        "opt_sleep_diff": round(sim_opt_sleep - pred_val, 1),
        "opt_screen_diff": round(sim_opt_screen - pred_val, 1),
        "opt_both_diff": round(sim_opt_both - pred_val, 1),
        "opt_both_val": round(sim_opt_both, 1)
    }

    # AI Insights and Correlations
    correlations = {}
    valid_cols = ["sleep", "mood", "steps", "screentime", "study", "expense", "water", "stress"]
    for col in valid_cols:
        if hist_df[col].std() > 0 and hist_df['productivity'].std() > 0:
            corr_val, _ = stats.pearsonr(hist_df[col], hist_df['productivity'])
            correlations[col] = round(float(corr_val), 3)
        else:
            correlations[col] = 0.0

    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    most_pos_habit = sorted_corr[0][0] if sorted_corr and sorted_corr[0][1] > 0.05 else "None"
    most_neg_habit = sorted_corr[-1][0] if sorted_corr and sorted_corr[-1][1] < -0.05 else "None"

    ai_observations = []
    if correlations.get('sleep', 0) > 0.05:
        ai_observations.append(f"Your productivity increases when sleep hours improve (Correlation: {correlations['sleep']:.2f}).")
    else:
        ai_observations.append("Adequate sleep shows strong baseline stabilization for recovery days.")

    if correlations.get('screentime', 0) < -0.05:
        ai_observations.append(f"High screen time reduces productivity (Correlation: {correlations['screentime']:.2f}).")
    else:
        ai_observations.append("Screen time is currently aligned within standard recovery parameters.")

    if correlations.get('steps', 0) > 0.05:
        ai_observations.append(f"Walking more steps is positively associated with productivity (Correlation: {correlations['steps']:.2f}).")
    else:
        ai_observations.append("Step counts improve cardiovascular health, stabilizing energy fluctuations.")

    if correlations.get('stress', 0) < -0.05:
        ai_observations.append(f"Stress levels are negatively correlated with overall performance (Correlation: {correlations['stress']:.2f}).")
    else:
        ai_observations.append("Stress index remains well-managed throughout the week.")

    recommendations_list = generate_dynamic_recommendations(user_data, pred_val, hist_df=hist_df, raw_df=raw_df, model=model, scaler=scaler)
    if recommendations_list:
        top_rec = recommendations_list[0]
        recommendation_highlight = f"{top_rec['title']} — {top_rec['text']}"
    else:
        recommendation_highlight = "Your daily habits are currently well aligned. Keep up the consistent routine!"

    # Anomalies
    anomalies = []
    for col in ['sleep', 'screentime', 'expense', 'stress']:
        col_mean = hist_df[col].mean()
        col_std = hist_df[col].std()
        if col_std > 0:
            z = (float(user_data[col]) - col_mean) / col_std
            if abs(z) > 1.96:
                direction = "higher" if z > 0 else "lower"
                anomalies.append(f"Unusual {col} today: logged {user_data[col]} (significantly {direction} than your average).")

    if not anomalies:
        anomalies.append("No lifestyle anomalies detected in the last 48 hours. Your metrics are steady.")

    # Weekly AI Report
    achievements = []
    problems = []
    improvements = []

    if sleep > sleep_avg:
        achievements.append(f"Sleep duration (+{sleep - sleep_avg:.1f}h above average)")
    if steps > steps_avg:
        achievements.append(f"Steps count (+{steps - int(steps_avg)} above average)")
    if is_exercise:
        achievements.append("Active workout logged")

    if stress > stress_avg:
        problems.append(f"Elevated stress ({stress} logged vs avg {stress_avg:.1f})")
    if screentime > screentime_avg:
        problems.append(f"Screen time surge ({screentime}h vs avg {screentime_avg:.1f}h)")
    if water < water_avg:
        problems.append("Water intake below baseline")

    if len(hist_df) >= 14:
        this_week_sleep = hist_df['sleep'].tail(7).mean()
        prev_week_sleep = hist_df['sleep'].iloc[-14:-7].mean()
        diff = this_week_sleep - prev_week_sleep
        if abs(diff) > 0.1:
            direction = "improved" if diff > 0 else "reduced"
            improvements.append(f"Weekly sleep duration has {direction} by {abs(diff)*60:.0f}m.")
    if not improvements:
        improvements.append("Maintaining consistent habit compliance this week.")

    weekly_score = int(round(hist_df['productivity'].tail(7).mean()))

    weekly_report = {
        "achievements": achievements[:2] if achievements else ["Consistent tracking maintained"],
        "problems": problems[:2] if problems else ["No significant problems detected"],
        "improvements": improvements[:2],
        "weekly_score": weekly_score,
        "prev_weekly_score": int(round(hist_df['productivity'].iloc[-14:-7].mean())) if len(hist_df) >= 14 else weekly_score - 2
    }

    # Productivity Level
    if pred_val >= 85:
        productivity_level = "Optimal"
        productivity_color = "#22C55E"
    elif pred_val >= 70:
        productivity_level = "Moderate"
        productivity_color = "#F59E0B"
    else:
        productivity_level = "Needs Focus"
        productivity_color = "#FF5F6D"

    return {
        "mood_emoji": mood_emoji,
        "mood_trend": mood_trend,
        "mood_avg": round(mood_avg, 1),
        "sleep_avg": round(sleep_avg, 1),
        "study_avg": round(study_avg, 1),
        "sleep_trend": sleep_trend,
        "steps_avg": int(round(steps_avg)),
        "steps_trend": steps_trend,
        "steps_goal_pct": round(steps_goal_pct, 1),
        "water_goal_pct": round(water_goal_pct, 1),
        "expense_weekly": round(expense_weekly, 2),
        "expense_monthly": round(expense_monthly, 2),
        "savings_val": round(savings_val, 2),
        "monthly_savings": round(monthly_savings, 2),
        "savings_goal": int(savings_goal),
        "savings_goal_pct": savings_goal_pct,
        "budget_remaining": round(budget_remaining, 2),
        "screentime_avg": round(screentime_avg, 1),
        "screentime_trend": screentime_trend,
        "health_score": health_score,
        "health_score_trend": health_score_trend,
        "goals": goals,
        "burnout": burnout,
        "insights": {
            "most_influential": sorted_corr[0][0].capitalize() if sorted_corr else "Sleep",
            "strongest_habit": most_pos_habit.capitalize() if most_pos_habit != "None" else "Sleep Duration",
            "weakest_habit": most_neg_habit.capitalize() if most_neg_habit != "None" else "Screen Time",
            "suggestions": recommendations_list[:2],
            "observations": ai_observations
        },
        "recommendation_highlight": recommendation_highlight,
        "whatif": whatif,
        "anomalies": anomalies,
        "weekly_report": weekly_report,
        "best_focus_hours": pattern_summary["best_focus_hours"],
        "best_day": pattern_summary["best_day"],
        "most_productive_week": pattern_summary["most_productive_week"],
        "productivity_level": productivity_level,
        "productivity_color": productivity_color,
        "productivity_val": int(round(pred_val)),
        "radar": calculate_radar_scores(user_data, hist_df=hist_df, prediction=pred_val, raw_df=raw_df),
        "cluster": classify_lifestyle_cluster(user_data, raw_df=raw_df),
    }
