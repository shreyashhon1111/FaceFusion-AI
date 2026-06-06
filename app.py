import cv2
import dlib
import numpy as np
import os
import time
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CAPTURE_FOLDER = "captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

camera = cv2.VideoCapture(0)   # 🔥 FIX: removed CAP_AVFOUNDATION (causes freeze)

ref_img = None
ref_points = None
triangles = None
last_frame = None


# ---------------- LANDMARKS ----------------

def get_landmarks(gray, face):
    landmarks = predictor(gray, face)
    return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)], np.int32)


# ---------------- DELAUNAY ----------------

def delaunay(rect, points):
    subdiv = cv2.Subdiv2D(rect)

    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))

    triangleList = subdiv.getTriangleList()
    delaunayTri = []

    for t in triangleList:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        ind = []

        for j in range(3):
            for k in range(len(points)):
                if abs(pts[j][0] - points[k][0]) < 1 and abs(pts[j][1] - points[k][1]) < 1:
                    ind.append(k)

        if len(ind) == 3:
            delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri


# ---------------- WARP ----------------

def warp_triangle(img1, img2, t1, t2):

    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1Rect = []
    t2Rect = []

    for i in range(3):
        t1Rect.append((t1[i][0] - r1[0], t1[i][1] - r1[1]))
        t2Rect.append((t2[i][0] - r2[0], t2[i][1] - r2[1]))

    img1Rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]

    size = (r2[2], r2[3])

    warpMat = cv2.getAffineTransform(np.float32(t1Rect), np.float32(t2Rect))

    warpImage = cv2.warpAffine(
        img1Rect,
        warpMat,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1,1,1))

    img2_section = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    img2_section = img2_section * (1 - mask) + warpImage * mask

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_section


# ---------------- VIDEO STREAM ----------------

def generate_frames():

    global ref_img, ref_points, triangles, last_frame

    while True:

        success, frame = camera.read()

        if not success:
            print("❌ Camera read failed")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # 🔥 SAFE SWAP
        if ref_img is not None and triangles is not None and len(faces) > 0:

            try:
                face = faces[0]
                points = get_landmarks(gray, face)

                warped = frame.copy()

                for tri in triangles:
                    t1 = [ref_points[tri[0]], ref_points[tri[1]], ref_points[tri[2]]]
                    t2 = [points[tri[0]], points[tri[1]], points[tri[2]]]
                    warp_triangle(ref_img, warped, t1, t2)

                hull = cv2.convexHull(points)

                mask = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillConvexPoly(mask, hull, 255)

                r = cv2.boundingRect(hull)
                center = (r[0] + r[2]//2, r[1] + r[3]//2)

                frame = cv2.seamlessClone(
                    np.uint8(warped),
                    frame,
                    mask,
                    center,
                    cv2.MIXED_CLONE
                )

            except Exception as e:
                print("❌ Swap error:", e)

        last_frame = frame.copy()

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------- ROUTES ----------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():

    global ref_img, ref_points, triangles

    file = request.files['file']

    path = os.path.join(UPLOAD_FOLDER, "uploaded.jpg")
    file.save(path)

    print("📁 Uploaded")

    ref_img = cv2.imread(path)

    if ref_img is None:
        return jsonify({"status": "❌ Image load fail"})

    gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) == 0:
        return jsonify({"status": "❌ No face detected"})

    ref_points = get_landmarks(gray, faces[0])

    h, w = gray.shape
    triangles = delaunay((0, 0, w, h), ref_points)

    return jsonify({"status": "✅ Face Loaded"})


@app.route('/capture')
def capture():

    global last_frame

    filename = f"capture_{int(time.time())}.jpg"
    path = os.path.join(CAPTURE_FOLDER, filename)

    cv2.imwrite(path, last_frame)

    return jsonify({"file": filename})


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)