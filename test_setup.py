import db
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta

# Initialize database
db.init_db()

# Create test user
password_hash = generate_password_hash("password123", method="scrypt")
user_id = db.create_user("Test User", "test@example.com", password_hash)
print(f"Created test user with ID: {user_id}")

# Add some test predictions for the past 30 days
for i in range(30):
    date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
    sleep = 7.0 + (i % 3) * 0.5
    mood = 7 + (i % 3)
    exercise = "Yes" if i % 2 == 0 else "No"
    steps = 8000 + (i % 2000)
    screentime = 4.0 + (i % 2)
    study = 3.0 + (i % 1)
    expense = 50 + (i % 100)
    water = 2.0 + (i % 1)
    stress = 4 + (i % 2)
    productivity = 75 + (i % 20)
    
    db.create_prediction(
        user_id=user_id,
        date=date,
        sleep=sleep,
        mood=mood,
        exercise=exercise,
        steps=steps,
        screentime=screentime,
        study=study,
        expense=expense,
        water=water,
        stress=stress,
        productivity=productivity
    )
    
print("Added 30 test predictions")

# Verify data was added
preds = db.get_user_predictions(user_id)
print(f"Total predictions for user: {len(preds)}")
