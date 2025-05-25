from flask import Flask, render_template, Response, request, jsonify
import cv2
import face_recognition
import numpy as np
import os
from pymongo import MongoClient
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["attendance_db"]
collection = db["attendance"]

# Load known faces
known_face_encodings = []
known_face_names = []

def load_known_faces():
    for filename in os.listdir("known_faces"):
        if filename.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join("known_faces", filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])

load_known_faces()

# Initialize camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    # Check if attendance already marked today
    existing = collection.find_one({"name": name, "date": date})
    if not existing:
        collection.insert_one({
            "name": name,
            "date": date,
            "time": time
        })
        print(f"[✓] Attendance saved for {name} at {time}")
        return "success"
    else:
        print(f"[!] Attendance already marked for {name} today.")
        return "duplicate"

def gen_frames():
    recognized_today = set()
    while True:
        success, frame = camera.read()
        if not success:
            break
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                index = matches.index(True)
                name = known_face_names[index]
                if name not in recognized_today:
                    mark_attendance(name)
                    recognized_today.add(name)

            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance_records')
def attendance_records():
    records = list(collection.find({}, {'_id': 0}))
    return jsonify(records)

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance_api():
    data = request.get_json()
    image_data = data.get('image')

    if not image_data:
        return jsonify({'status': 'error', 'message': 'No image data provided'})

    try:
        # Strip base64 header
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)

        # Convert to OpenCV image
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        frame = np.array(image)

        # Resize and convert color for face_recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR)  # Fix color conversion

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                result = mark_attendance(name)
                return jsonify({'status': result, 'name': name})

        return jsonify({'status': 'error', 'message': 'Face not recognized'})

    except Exception as e:
        print(f"Error in /mark_attendance: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'})

if __name__ == '__main__':
    app.run(debug=True)
