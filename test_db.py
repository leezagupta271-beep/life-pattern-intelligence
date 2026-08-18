import db

db.init_db()

print("Unique users who logged in:", db.count_unique_logins())
print("All login events:")
for event in db.get_login_events():
    print(event)