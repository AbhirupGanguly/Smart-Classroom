import cv2
import face_recognition
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
VIDEO_PATH = "student3.mp4"
OUTPUT_FOLDER = "static\student_images"
FRAME_SKIP = 50   # Process every 5th frame (increase for speed)

# -----------------------------
# CREATE OUTPUT DIRECTORY
# -----------------------------
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# -----------------------------
# LOAD VIDEO
# -----------------------------
video = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0
face_count = 0

print("Processing video...")

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames for speed
    if frame_count % FRAME_SKIP != 0:
        continue

    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    # model="cnn" for GPU (slower but more accurate)

    for (top, right, bottom, left) in face_locations:
        face_image = frame[top:bottom, left:right]

        face_filename = os.path.join(
            OUTPUT_FOLDER,
            f"face_{face_count}.jpg"
        )

        cv2.imwrite(face_filename, face_image)
        face_count += 1

    print(f"Frame {frame_count} processed | Faces found: {len(face_locations)}")

video.release()
print(f"\nDone. Total faces saved: {face_count}")