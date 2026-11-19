from datetime import datetime, timedelta
from functools import wraps
import os
from pathlib import Path
import secrets
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import db
import analytics
import analysis
import graphs
import export_analytics

BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "public" / "static"),
    static_url_path="/static",
)
secret_key = os.environ.get("SECRET_KEY")
if os.environ.get("VERCEL") and not secret_key:
    raise RuntimeError("Set the SECRET_KEY environment variable for Vercel deployments.")
app.secret_key = secret_key or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_PERMANENT=False
)
db.init_db()


def clear_user_session():
    session.clear()
    session.modified = True


def get_authenticated_user_id():
    user_id = session.get("user_id")
    if user_id is None:
        return None

    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        clear_user_session()
        return None

    user = db.get_user_by_id(user_id)
    if not user:
        clear_user_session()
        return None

    return user_id


@app.before_request
def check_session_timeout():
    if request.endpoint in ['static', 'login', 'signup', 'logout']:
        return

    user_id = get_authenticated_user_id()
    if not user_id:
        session.clear()
        if request.endpoint == 'home':
            return redirect(url_for('login'))
        return redirect(url_for('login'))

    last_activity_str = session.get("last_activity")
    now = datetime.now()
    timeout = timedelta(minutes=30)

    if last_activity_str:
        try:
            last_activity = datetime.fromisoformat(last_activity_str)
            if now - last_activity > timeout:
                clear_user_session()
                flash("Your session has expired due to inactivity. Please sign in again.", "info")
                return redirect(url_for("login"))
        except Exception:
            clear_user_session()
            return redirect(url_for("login"))

    session["last_activity"] = now.isoformat()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not get_authenticated_user_id():
            clear_user_session()
            flash("Please sign in to access your dashboard.", "info")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# Load trained model
model = joblib.load(BASE_DIR / "RandomForestRegressor.joblib")

# Recreate the preprocessing objects used during training
raw_df = pd.read_csv(BASE_DIR / "life_data_500000 (1).csv")
le = LabelEncoder()
raw_df["Exercise"] = le.fit_transform(raw_df["Exercise"])

# The model was trained on 9 features in this exact order:
# Sleep, Mood, Exercise, Steps, ScreenTime, StudyHours, Expense, Water, Stress
feature_columns = [
    "Sleep",
    "Mood",
    "Exercise",
    "Steps",
    "ScreenTime",
    "StudyHours",
    "Expense",
    "Water",
    "Stress",
]

X = raw_df[feature_columns]
y = raw_df["Productivity"]
X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

# PAGE-TO-GRAPHS MAPPING: Define which graphs each page needs to reduce generation overhead
PAGE_GRAPHS_MAPPING = {
    "dashboard": ["mini_lifescore", "mini_mood", "mini_productivity", "mini_sleep", "mini_screentime", "mini_water", "mini_steps", "mini_expense", "mini_savings", "mini_goals"],
    "ai_insights": ["feature_importance", "scatter", "radar"],
    "burnout": ["burnout_gauge"],
    "finance": ["expense_donut", "monthly_spending"],
    "habit_analytics": ["correlation_matrix", "trend_analysis", "habit_impact"],
    "health": ["mini_water", "mini_steps"],
    "mood": ["mood_calendar", "mood_timeline"],
    "productivity": ["heatmap", "sankey"],
    "screen_time": ["screentime_bars"],
    "sleep": ["sleep_trend"],
    "goals": [],
    "recommendations": [],
    "reports": [],
    "settings": [],
}


def seed_user_history_if_empty(user_id):
    # New users should start from a clean state until they log their first day.
    preds = db.get_user_predictions(user_id)
    if preds:
        return preds
    return []


def get_analytics_payload(user_id, user_data, pred_val, page_name="dashboard"):
    user_data["prediction"] = pred_val
    history = db.get_user_predictions(user_id)
    user_df = pd.DataFrame(history) if history else pd.DataFrame([user_data])

    # Calculate Data Science Analytics (these are cheap and used by multiple pages)
    payload = analytics.calculate_dashboard_analytics(history, user_data, pred_val, raw_df, model, scaler)
    payload["monthly_summary"] = analytics.calculate_monthly_summary(history, user_data, pred_val, model=model, scaler=scaler)
    payload["monthly_forecast"] = analytics.calculate_monthly_forecast(history, user_data)
    db.create_or_update_monthly_summary(
        user_id=user_id,
        year_month=payload["monthly_summary"]["year_month"],
        avg_sleep=payload["monthly_summary"]["avg_sleep"],
        avg_mood=payload["monthly_summary"]["avg_mood"],
        exercise_pct=payload["monthly_summary"]["exercise_pct"],
        avg_steps=payload["monthly_summary"]["avg_steps"],
        avg_screentime=payload["monthly_summary"]["avg_screentime"],
        avg_study=payload["monthly_summary"]["avg_study"],
        total_expense=payload["monthly_summary"]["total_expense"],
        avg_water=payload["monthly_summary"]["avg_water"],
        avg_stress=payload["monthly_summary"]["avg_stress"],
        avg_productivity=payload["monthly_summary"]["avg_productivity"],
        predicted_productivity=payload["monthly_summary"]["predicted_productivity"],
    )
    payload["health_cards"] = analytics.generate_health_summary_cards(user_data, pred_val)
    payload["habit_impact_scores"] = analytics.generate_habit_impact_scores(user_data)

    # Get required graphs for this page
    required_graphs = PAGE_GRAPHS_MAPPING.get(page_name, [])

    # OPTIMIZE: Only generate graphs that are actually needed for this page
    # This reduces page load time by avoiding unnecessary matplotlib rendering
    graph_dict = {}
    
    # Pre-calculate common dependencies needed by multiple graphs
    correlations = None
    if any(g in required_graphs for g in ["habit_impact", "correlation_matrix"]):
        correlations = analytics.calculate_habit_correlations(user_df)
    
    # Generate only required graphs
    if "heatmap" in required_graphs:
        graph_dict["heatmap"] = graphs.generate_productivity_heatmap(raw_df, user_data)
    
    if "burnout_gauge" in required_graphs:
        graph_dict["burnout_gauge"] = graphs.generate_burnout_gauge(payload["burnout"]["score"], payload["burnout"]["category"])
    
    if "habit_impact" in required_graphs:
        if correlations is None:
            correlations = analytics.calculate_habit_correlations(user_df)
        graph_dict["habit_impact"] = graphs.generate_habit_impact_chart(correlations)
    
    if "sleep_trend" in required_graphs:
        graph_dict["sleep_trend"] = graphs.generate_sleep_trend_chart(user_df, float(user_data.get("sleep", 7.5)))
    
    if "radar" in required_graphs:
        graph_dict["radar"] = graphs.generate_radar_chart(user_data, radar_data=payload.get("radar"))
    
    if "scatter" in required_graphs:
        graph_dict["scatter"] = graphs.generate_scatter_plots(raw_df, user_data)
    
    if "cluster" in required_graphs:
        graph_dict["cluster"] = graphs.generate_cluster_chart(raw_df, user_data, cluster_data=payload.get("cluster"))
    
    if "sankey" in required_graphs:
        graph_dict["sankey"] = graphs.generate_sankey_diagram(user_data, pred_val)
    
    if "correlation_matrix" in required_graphs:
        graph_dict["correlation_matrix"] = graphs.generate_correlation_matrix(user_df)
    
    if "distributions" in required_graphs:
        graph_dict["distributions"] = graphs.generate_distribution_graphs(raw_df, user_data)
    
    if "trend_analysis" in required_graphs:
        graph_dict["trend_analysis"] = graphs.generate_trend_analysis(user_df, pred_val)
    
    if "feature_importance" in required_graphs:
        graph_dict["feature_importance"] = graphs.generate_feature_importance_chart(model, feature_columns)
    
    if "mood_calendar" in required_graphs:
        graph_dict["mood_calendar"] = graphs.generate_mood_calendar_chart(user_df)
    
    if "mood_timeline" in required_graphs:
        graph_dict["mood_timeline"] = graphs.generate_mood_timeline_chart(user_df)
    
    if "expense_donut" in required_graphs:
        graph_dict["expense_donut"] = graphs.generate_expense_donut(user_df, float(user_data.get("expense", 0)))
    
    if "monthly_spending" in required_graphs:
        graph_dict["monthly_spending"] = graphs.generate_monthly_spending(user_df)
    
    if "screentime_bars" in required_graphs:
        graph_dict["screentime_bars"] = graphs.generate_screentime_bars(user_df)
    
    if "whatif_chart" in required_graphs:
        graph_dict["whatif_chart"] = graphs.generate_whatif_chart(payload["whatif"])
    
    # Mini KPI card graphs
    if "mini_lifescore" in required_graphs:
        graph_dict["mini_lifescore"] = graphs.generate_mini_line(user_df['productivity'].tail(7).tolist() if 'productivity' in user_df.columns else [pred_val], '#00F5D4')
    
    if "mini_mood" in required_graphs:
        graph_dict["mini_mood"] = graphs.generate_mini_bar(user_df['mood'].tail(7).tolist() if 'mood' in user_df.columns else [float(user_data.get('mood', 7))], '#FF4D9D')
    
    if "mini_productivity" in required_graphs:
        graph_dict["mini_productivity"] = graphs.generate_mini_line(user_df['productivity'].tail(7).tolist() if 'productivity' in user_df.columns else [pred_val], '#00E5A8')
    
    if "mini_sleep" in required_graphs:
        graph_dict["mini_sleep"] = graphs.generate_mini_bar(user_df['sleep'].tail(7).tolist() if 'sleep' in user_df.columns else [float(user_data.get('sleep', 7.5))], '#6C63FF')
    
    if "mini_screentime" in required_graphs:
        graph_dict["mini_screentime"] = graphs.generate_mini_bar(user_df['screentime'].tail(7).tolist() if 'screentime' in user_df.columns else [float(user_data.get('screentime', 4.2))], '#A855F7')
    
    if "mini_water" in required_graphs:
        graph_dict["mini_water"] = graphs.generate_mini_progress(payload["water_goal_pct"], '#22C55E')
    
    if "mini_steps" in required_graphs:
        graph_dict["mini_steps"] = graphs.generate_mini_bar(user_df['steps'].tail(7).tolist() if 'steps' in user_df.columns else [float(user_data.get('steps', 8432))], '#22C55E')
    
    if "mini_expense" in required_graphs:
        graph_dict["mini_expense"] = graphs.generate_mini_line(user_df['expense'].tail(7).tolist() if 'expense' in user_df.columns else [float(user_data.get('expense', 1240))], '#F59E0B')
    
    if "mini_savings" in required_graphs:
        graph_dict["mini_savings"] = graphs.generate_mini_line(user_df['expense'].tail(15).apply(lambda x: max(0.0, 100.0 - x)).tolist() if 'expense' in user_df.columns else [450], '#F59E0B')
    
    if "mini_goals" in required_graphs:
        graph_dict["mini_goals"] = graphs.generate_mini_donut(payload["goals"][0]["progress"], '#38BDF8')

    payload["graphs"] = graph_dict
    
    # Calculate sleep whatif only if needed (it's expensive and used by multiple pages)
    if page_name in ["dashboard", "sleep", "recommendations", "burnout"]:
        payload["sleep_whatif_chain"] = analytics.calculate_sleep_whatif_chain(user_data, pred_val, history=history, raw_df=raw_df, model=model, scaler=scaler)
    
    payload["habit_insights"] = analysis.generate_habit_insights(history if history else None, user_data, pred_val, raw_df=raw_df, model=model)
    return payload


def get_user_state_and_payload(user_id, page_name="dashboard"):
    # If the user has no historical entries, treat them as a fresh user
    # and do NOT synthesize or surface default/fallback prediction values.
    history = seed_user_history_if_empty(user_id)
    if not history:
        return None, None, None

    # Use the most recent historical record as the user's current state
    last_rec = history[-1]
    user_data = {
        "sleep": float(last_rec["sleep"]),
        "mood": int(last_rec["mood"]),
        "exercise": last_rec["exercise"],
        "steps": int(last_rec["steps"]),
        "screentime": float(last_rec["screentime"]),
        "study": float(last_rec["study"]),
        "expense": float(last_rec["expense"]),
        "water": float(last_rec["water"]),
        "stress": int(last_rec["stress"]),
    }
    pred_val = float(last_rec["productivity"])
    payload = get_analytics_payload(user_id, user_data, pred_val, page_name=page_name)
    return user_data, pred_val, payload


def render_lifestyle_page(template_name, active_page_name):
    user_id = session["user_id"]
    history = db.get_user_predictions(user_id)
    # If the user has no saved history, always render an explicit empty state
    # for lifestyle pages so no previous-user or fallback values are shown.
    if not history:
        return render_template(
            template_name,
            active_page=active_page_name,
            current_date=datetime.now().strftime("%b %d, %Y"),
            sleep=None,
            mood=None,
            exercise=None,
            steps=None,
            screentime=None,
            study=None,
            expense=None,
            water=None,
            stress=None,
            prediction=None,
            analytics=None,
            empty_state=True,
            user_name=session.get("user_name", "")
        )

    user_data, pred_val, payload = get_user_state_and_payload(user_id, page_name=active_page_name)
    return render_template(
        template_name,
        active_page=active_page_name,
        current_date=datetime.now().strftime("%b %d, %Y"),
        sleep=user_data["sleep"],
        mood=user_data["mood"],
        exercise=user_data["exercise"],
        steps=user_data["steps"],
        screentime=user_data["screentime"],
        study=user_data["study"],
        expense=user_data["expense"],
        water=user_data["water"],
        stress=user_data["stress"],
        prediction=pred_val,
        analytics=payload,
        empty_state=False,
        user_name=session.get("user_name", "there")
    )


@app.route("/")
def home():
    if not get_authenticated_user_id():
        clear_user_session()
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    history = db.get_user_predictions(user_id)

    if not history:
        return render_template(
            "dashboard.html",
            active_page="dashboard",
            current_date=datetime.now().strftime("%b %d, %Y"),
            user_name=session.get("user_name", "there"),
            empty_state=True,
            prediction=None,
            analytics=None,
            sleep=None,
            mood=None,
            exercise=None,
            steps=None,
            screentime=None,
            study=None,
            expense=None,
            water=None,
            stress=None
        )

    user_data, pred_val, payload = get_user_state_and_payload(user_id, page_name="dashboard")
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        current_date=datetime.now().strftime("%b %d, %Y"),
        user_name=session.get("user_name", "there"),
        empty_state=False,
        sleep=user_data["sleep"],
        mood=user_data["mood"],
        exercise=user_data["exercise"],
        steps=user_data["steps"],
        screentime=user_data["screentime"],
        study=user_data["study"],
        expense=user_data["expense"],
        water=user_data["water"],
        stress=user_data["stress"],
        prediction=pred_val,
        analytics=payload,
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    user_id = session["user_id"]
    seed_user_history_if_empty(user_id)

    form = request.form
    sleep = float(form.get("Sleep", 0) or 0)
    mood = int(form.get("Mood", 0) or 0)
    exercise_value = form.get("Exercise", "No") or "No"
    steps = int(form.get("Steps", 0) or 0)
    screentime = float(form.get("ScreenTime", 0) or 0)
    study = float(form.get("StudyHours", 0) or 0)
    expense = float(form.get("Expense", 0) or 0)
    water = float(form.get("Water", 0) or 0)
    stress = int(form.get("Stress", 0) or 0)

    # Match the notebook encoding: Yes/No -> 0/1
    if exercise_value in le.classes_:
        exercise = le.transform([exercise_value])[0]
    else:
        exercise = le.transform(["No"])[0]

    data = pd.DataFrame([[
        sleep,
        mood,
        exercise,
        steps,
        screentime,
        study,
        expense,
        water,
        stress
    ]], columns=feature_columns)

    # Use the same scaling parameters from the training data.
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0]
    pred_round = round(prediction, 2)

    # Write prediction to database as historical record
    db.create_prediction(
        user_id=user_id,
        date=datetime.now().strftime("%Y-%m-%d"),
        sleep=sleep,
        mood=mood,
        exercise=exercise_value,
        steps=steps,
        screentime=screentime,
        study=study,
        expense=expense,
        water=water,
        stress=stress,
        productivity=pred_round
    )

    user_data = {
        "sleep": sleep,
        "mood": mood,
        "exercise": exercise_value,
        "steps": steps,
        "screentime": screentime,
        "study": study,
        "expense": expense,
        "water": water,
        "stress": stress
    }
    payload = get_analytics_payload(user_id, user_data, pred_round, page_name="dashboard")

    return render_template(
        "dashboard.html",
        prediction=pred_round,
        current_date=datetime.now().strftime("%b %d, %Y"),
        sleep=sleep,
        mood=mood,
        exercise=exercise_value,
        steps=steps,
        screentime=screentime,
        study=study,
        expense=expense,
        water=water,
        stress=stress,
        analytics=payload,
        active_page="dashboard",
        empty_state=False,
        user_name=session.get("user_name", "there")
    )


@app.route("/ai-insights")
@login_required
def ai_insights():
    return render_lifestyle_page("ai_insights.html", "ai_insights")


@app.route("/habit-analytics")
@login_required
def habit_analytics():
    return render_lifestyle_page("habit_analytics.html", "habit_analytics")


@app.route("/productivity")
@login_required
def productivity():
    return render_lifestyle_page("productivity.html", "productivity")


@app.route("/mood")
@login_required
def mood():
    return render_lifestyle_page("mood.html", "mood")


@app.route("/sleep")
@login_required
def sleep():
    return render_lifestyle_page("sleep.html", "sleep")


@app.route("/health")
@login_required
def health():
    return render_lifestyle_page("health.html", "health")


@app.route("/screen-time")
@login_required
def screen_time():
    return render_lifestyle_page("screen_time.html", "screen_time")


@app.route("/finance")
@login_required
def finance():
    return render_lifestyle_page("finance.html", "finance")


@app.route("/goals")
@login_required
def goals():
    return render_lifestyle_page("goals.html", "goals")


@app.route("/burnout")
@login_required
def burnout():
    return render_lifestyle_page("burnout.html", "burnout")


@app.route("/recommendations")
@login_required
def recommendations():
    return render_lifestyle_page("recommendations.html", "recommendations")


@app.route("/reports")
@login_required
def reports():
    return render_lifestyle_page("reports.html", "reports")


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        session["target_sleep"] = float(request.form.get("target_sleep", 8.0))
        session["target_steps"] = int(request.form.get("target_steps", 10000))
        session["target_water"] = float(request.form.get("target_water", 3.0))
        session["target_budget"] = float(request.form.get("target_budget", 2000.0))
        flash("Custom targets updated successfully!", "success")
        return redirect(url_for("settings"))
    return render_lifestyle_page("settings.html", "settings")


@app.route("/export/<fmt>")
@login_required
def export_report(fmt):
    user_data = {
        "sleep": float(request.args.get("sleep", 7.5)),
        "mood": int(request.args.get("mood", 7)),
        "exercise": request.args.get("exercise", "No"),
        "steps": int(request.args.get("steps", 8432)),
        "screentime": float(request.args.get("screentime", 4.2)),
        "study": float(request.args.get("study", 3.5)),
        "expense": float(request.args.get("expense", 1240)),
        "water": float(request.args.get("water", 2.0)),
        "stress": int(request.args.get("stress", 4))
    }
    pred_val = float(request.args.get("prediction", 92.0))
    burnout_info = analytics.calculate_burnout_risk(user_data)
    insights = analytics.calculate_ai_insights(user_data, pred_val, raw_df, model)

    if fmt == "csv":
        return export_analytics.export_as_csv(user_data, pred_val, burnout_info, insights)
    else:
        return export_analytics.export_as_pdf(user_data, pred_val, burnout_info, insights)


@app.route("/login", methods=["GET", "POST"])
def login():
    if get_authenticated_user_id():
        return redirect(url_for("dashboard"))

    session.clear()
    session.modified = True

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        user = db.get_user_by_email(email)
        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password. Please try again.", "danger")
            return render_template("login.html")

        session.clear()
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        session["user_email"] = user["email"]
        session["last_activity"] = datetime.now().isoformat()
        db.record_login(user["id"])
        flash(f"Welcome back, {user['name']}!", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if get_authenticated_user_id():
        return redirect(url_for("dashboard"))

    session.clear()
    session.modified = True

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("signup.html")

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("signup.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return render_template("signup.html")

        existing_user = db.get_user_by_email(email)
        if existing_user:
            flash("An account with that email already exists. Please sign in.", "info")
            return redirect(url_for("login"))

        hashed_pw = generate_password_hash(password, method="scrypt")
        user_id = db.create_user(name, email, hashed_pw)

        if user_id:
            db.record_login(user_id)
            session.clear()
            session.modified = True
            flash("Account created successfully! Please sign in.", "success")
            return redirect(url_for("login"))
        else:
            flash("An error occurred while creating your account. Please try again.", "danger")
            return render_template("signup.html")

    return render_template("signup.html")


@app.route("/logout")
def logout():
    clear_user_session()
    flash("You have been logged out safely.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)