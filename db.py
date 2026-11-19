import sqlite3
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = (
    Path(os.environ["DATABASE_PATH"])
    if os.environ.get("DATABASE_PATH")
    else Path("/tmp/users.db")
    if os.environ.get("VERCEL")
    else BASE_DIR / "users.db"
)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT NOT NULL,
            sleep REAL,
            mood INTEGER,
            exercise TEXT,
            steps INTEGER,
            screentime REAL,
            study REAL,
            expense REAL,
            water REAL,
            stress INTEGER,
            productivity REAL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monthly_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            year_month TEXT NOT NULL,
            avg_sleep REAL,
            avg_mood REAL,
            exercise_pct REAL,
            avg_steps REAL,
            avg_screentime REAL,
            avg_study REAL,
            total_expense REAL,
            avg_water REAL,
            avg_stress REAL,
            avg_productivity REAL,
            predicted_productivity REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, year_month),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            login_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


def create_user(name, email, password_hash):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email.lower().strip(), password_hash),
        )
        conn.commit()
        user_id = cursor.lastrowid
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_user_by_email(email):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email.lower().strip(),))
    user = cursor.fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user


def create_prediction(user_id, date, sleep, mood, exercise, steps, screentime, study, expense, water, stress, productivity):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (user_id, date, sleep, mood, exercise, steps, screentime, study, expense, water, stress, productivity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, date, sleep, mood, exercise, steps, screentime, study, expense, water, stress, productivity),
    )
    conn.commit()
    pred_id = cursor.lastrowid
    conn.close()
    return pred_id


def create_or_update_monthly_summary(
    user_id,
    year_month,
    avg_sleep,
    avg_mood,
    exercise_pct,
    avg_steps,
    avg_screentime,
    avg_study,
    total_expense,
    avg_water,
    avg_stress,
    avg_productivity,
    predicted_productivity,
):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO monthly_summaries (
            user_id, year_month, avg_sleep, avg_mood, exercise_pct, avg_steps,
            avg_screentime, avg_study, total_expense, avg_water, avg_stress,
            avg_productivity, predicted_productivity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, year_month) DO UPDATE SET
            avg_sleep = excluded.avg_sleep,
            avg_mood = excluded.avg_mood,
            exercise_pct = excluded.exercise_pct,
            avg_steps = excluded.avg_steps,
            avg_screentime = excluded.avg_screentime,
            avg_study = excluded.avg_study,
            total_expense = excluded.total_expense,
            avg_water = excluded.avg_water,
            avg_stress = excluded.avg_stress,
            avg_productivity = excluded.avg_productivity,
            predicted_productivity = excluded.predicted_productivity
        """,
        (
            user_id,
            year_month,
            avg_sleep,
            avg_mood,
            exercise_pct,
            avg_steps,
            avg_screentime,
            avg_study,
            total_expense,
            avg_water,
            avg_stress,
            avg_productivity,
            predicted_productivity,
        ),
    )
    conn.commit()
    summary_id = cursor.lastrowid
    conn.close()
    return summary_id


def record_login(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO login_events (user_id, login_at) VALUES (?, CURRENT_TIMESTAMP)",
        (user_id,)
    )
    conn.commit()
    login_id = cursor.lastrowid
    conn.close()
    return login_id


def count_unique_logins():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT user_id) FROM login_events")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_login_events(user_id=None):
    conn = get_db()
    cursor = conn.cursor()
    if user_id is None:
        cursor.execute("SELECT * FROM login_events ORDER BY login_at DESC")
    else:
        cursor.execute(
            "SELECT * FROM login_events WHERE user_id = ? ORDER BY login_at DESC",
            (user_id,)
        )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_user_predictions(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY date ASC, id ASC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_user_monthly_summaries(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM monthly_summaries WHERE user_id = ? ORDER BY year_month ASC",
        (user_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
