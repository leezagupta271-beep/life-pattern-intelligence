import sqlite3
import os

DB_PATH = "yourdb.db"  # replace with your actual SQLite file name

def get_logged_in_users(db_path=DB_PATH):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE last_login IS NOT NULL;")
        return cur.fetchall()

if __name__ == "__main__":
    rows = get_logged_in_users()
    if not rows:
        print("No users have logged in yet.")
    else:
        for row in rows:
            print(dict(row))