import cv2
import numpy as np
import face_recognition
import mediapipe as mp
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial import distance as dist

# Load YOLO
model = YOLO("yolov8m.pt")

# Tracker
tracker = DeepSort(max_age=30)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Store known tracker → student mapping
tracker_identity = {}

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def biological_attention_score(frame, landmarks):

    h, w, _ = frame.shape

    nose = landmarks[1]
    nose_x = int(nose.x * w)

    if 0.35*w < nose_x < 0.65*w:
        head_score = 1
    else:
        head_score = 0.5

    left_eye_idx = [33,160,158,133,153,144]
    right_eye_idx = [362,385,387,263,373,380]

    left_eye = [(landmarks[i].x*w, landmarks[i].y*h) for i in left_eye_idx]
    right_eye = [(landmarks[i].x*w, landmarks[i].y*h) for i in right_eye_idx]

    ear_left = eye_aspect_ratio(left_eye)
    ear_right = eye_aspect_ratio(right_eye)

    ear = (ear_left + ear_right) / 2

    if ear > 0.25:
        eye_score = 1
    else:
        eye_score = 0.4

    final_score = (0.6 * head_score + 0.4 * eye_score)

    return final_score


def process_video(video_path, student_encodings):

    cap = cv2.VideoCapture(video_path)

    results = {}

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

    # process only every 5th frame
        if frame_count % 5 != 0:
            continue

        detections = []

        yolo = model(frame, conf=0.5, verbose=False)[0]

        for box in yolo.boxes:

            if int(box.cls[0]) != 0:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            w = x2-x1
            h = y2-y1

            detections.append(([x1,y1,w,h],1.0,'person'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            l,t,w,h = map(int, track.to_ltrb())

            crop = frame[t:t+h, l:l+w]

            if crop.size == 0:
                continue

            if track_id not in tracker_identity:

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                faces = face_recognition.face_locations(rgb)
                encodings = face_recognition.face_encodings(rgb, faces)

                for encoding in encodings:

                    for roll,db_encoding in student_encodings.items():

                        match = face_recognition.compare_faces([db_encoding],encoding,tolerance=0.5)

                        if match[0]:

                            tracker_identity[track_id] = roll

                            if roll not in results:
                                results[roll] = {
                                    "frames":0,
                                    "bio_score_sum":0
                                }

            if track_id in tracker_identity:

                roll = tracker_identity[track_id]

                results[roll]["frames"] += 1

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                mp_results = face_mesh.process(rgb)

                if mp_results.multi_face_landmarks:

                    landmarks = mp_results.multi_face_landmarks[0].landmark

                    score = biological_attention_score(crop,landmarks)

                    results[roll]["bio_score_sum"] += score

    cap.release()

    return results