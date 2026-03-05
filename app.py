import os
import datetime
import mysql.connector
import face_recognition
from flask import Flask, render_template, request
from vision_engine import process_video

app = Flask(__name__)

UPLOAD_FOLDER = "static/student_images"
DATABASE_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Rup@212004",
    "database": "classroom_ai"
}

def get_connection():
    return mysql.connector.connect(**DATABASE_CONFIG)

def load_student_encodings():

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT roll, image_url FROM students")
    rows = cursor.fetchall()
    conn.close()

    encodings = {}

    for roll, img_path in rows:
        if os.path.exists(img_path):
            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)
            if encoding:
                encodings[roll] = encoding[0]

    return encodings

def save_results(results):

    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now()

    for roll in results:

        frames = results[roll]["frames"]
        bio_sum = results[roll]["bio_score_sum"]

        if frames == 0:
            continue

        attention = round((bio_sum / frames) * 100, 2)

        cursor.execute(
            "UPDATE students SET final_attention = %s WHERE roll = %s",
            (attention, roll)
        )

        cursor.execute(
            """
            INSERT INTO sessions (roll, attendance, attention, session_time)
            VALUES (%s, %s, %s, %s)
            """,
            (roll, 1, attention, now)
        )

    conn.commit()
    conn.close()

@app.route("/", methods=["GET", "POST"])
def dashboard():

    if request.method == "POST":
        file = request.files["video"]
        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(video_path)

        encodings = load_student_encodings()
        results = process_video(video_path, encodings)
        save_results(results)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT roll, name, age, final_attention FROM students")
    students = cursor.fetchall()
    conn.close()

    return render_template("dashboard.html", students=students)

if __name__ == "__main__":
    app.run(debug=True)