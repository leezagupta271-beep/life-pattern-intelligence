import io
from flask import Response, make_response
import pandas as pd


def export_as_csv(user_data, prediction, burnout_info, insights):
    """
    Generate CSV file response containing user inputs, prediction, burnout analysis, and AI insights.
    """
    output = io.StringIO()

    data = {
        "Metric": [
            "Life Score Prediction",
            "Burnout Risk Score (%)",
            "Burnout Risk Category",
            "Sleep (Hours)",
            "Mood (1-10)",
            "Exercise",
            "Steps",
            "Screen Time (Hours)",
            "Study Hours",
            "Expense ($)",
            "Water Intake (Liters)",
            "Stress Level (1-10)",
            "Most Influential Habit",
            "Strongest Habit",
            "Weakest Habit"
        ],
        "Value": [
            prediction,
            burnout_info.get("score"),
            burnout_info.get("category"),
            user_data.get("sleep"),
            user_data.get("mood"),
            user_data.get("exercise"),
            user_data.get("steps"),
            user_data.get("screentime"),
            user_data.get("study"),
            user_data.get("expense"),
            user_data.get("water"),
            user_data.get("stress"),
            insights.get("most_influential"),
            insights.get("strongest_habit"),
            insights.get("weakest_habit")
        ]
    }

    df = pd.DataFrame(data)
    df.to_csv(output, index=False)

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=life_pattern_analytics_report.csv"
    response.headers["Content-type"] = "text/csv"
    return response


def export_as_pdf(user_data, prediction, burnout_info, insights):
    """
    Generate clean plain-text/HTML PDF summary response.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Life Pattern Intelligence Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
            h1 {{ color: #A855F7; }}
            .card {{ background: #f4f4f9; padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
            th {{ background: #A855F7; color: white; }}
        </style>
    </head>
    <body>
        <h1>Life Pattern Intelligence - Executive Report</h1>
        <div class="card">
            <h2>Overall Life Score: <strong>{prediction}/100</strong></h2>
            <p><strong>Burnout Risk:</strong> {burnout_info.get('score')}% ({burnout_info.get('category')})</p>
            <p><strong>Most Influential Habit:</strong> {insights.get('most_influential')}</p>
        </div>

        <h3>Input Habit Profile</h3>
        <table>
            <tr><th>Habit</th><th>Logged Value</th></tr>
            <tr><td>Sleep</td><td>{user_data.get('sleep')} hours</td></tr>
            <tr><td>Mood</td><td>{user_data.get('mood')}/10</td></tr>
            <tr><td>Exercise</td><td>{user_data.get('exercise')}</td></tr>
            <tr><td>Steps</td><td>{user_data.get('steps')}</td></tr>
            <tr><td>Screen Time</td><td>{user_data.get('screentime')} hours</td></tr>
            <tr><td>Water Intake</td><td>{user_data.get('water')} Liters</td></tr>
            <tr><td>Stress</td><td>{user_data.get('stress')}/10</td></tr>
        </table>
    </body>
    </html>
    """
    response = make_response(html_content)
    response.headers["Content-Disposition"] = "attachment; filename=life_pattern_report.html"
    response.headers["Content-type"] = "text/html"
    return response
