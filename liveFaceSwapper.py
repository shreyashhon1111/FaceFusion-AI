import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_landmarks(gray, face):

    landmarks = predictor(gray, face)

    points = []

    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        points.append((x,y))

    return np.array(points, np.int32)


def delaunay_triangulation(rect, points):

    subdiv = cv2.Subdiv2D(rect)

    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))

    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    for t in triangleList:

        pt = [(t[0],t[1]),(t[2],t[3]),(t[4],t[5])]

        ind = []

        for j in range(3):
            for k in range(len(points)):

                if abs(pt[j][0]-points[k][0]) < 1 and abs(pt[j][1]-points[k][1]) < 1:
                    ind.append(k)

        if len(ind) == 3:
            delaunayTri.append((ind[0],ind[1],ind[2]))

    return delaunayTri


def warp_triangle(img1, img2, t1, t2):

    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(3):

        t1Rect.append(((t1[i][0]-r1[0]),(t1[i][1]-r1[1])))
        t2Rect.append(((t2[i][0]-r2[0]),(t2[i][1]-r2[1])))
        t2RectInt.append(((t2[i][0]-r2[0]),(t2[i][1]-r2[1])))

    mask = np.zeros((r2[3], r2[2],3), dtype=np.float32)

    cv2.fillConvexPoly(mask, np.int32(t2RectInt),(1.0,1.0,1.0),16,0)

    img1Rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]

    size = (r2[2], r2[3])

    mat = cv2.getAffineTransform(np.float32(t1Rect), np.float32(t2Rect))

    warpImage = cv2.warpAffine(
        img1Rect,
        mat,
        size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101
    )

    img2Rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

    img2Rect = img2Rect*(1-mask) + warpImage*mask

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2Rect


ref_img = cv2.imread("Example 1.png")

ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

faces = detector(ref_gray)

if len(faces) == 0:
    print("No face in reference image")
    exit()

ref_points = get_landmarks(ref_gray, faces[0])

h, w = ref_gray.shape

rect = (0,0,w,h)

triangles = delaunay_triangulation(rect, ref_points)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame,None,fx=0.6,fy=0.6)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) == 0:

        cv2.imshow("Face Swap",frame)

        if cv2.waitKey(1)==27:
            break

        continue

    face = faces[0]

    points = get_landmarks(gray,face)

    warped = np.copy(frame)

    for tri in triangles:

        t1 = [ref_points[tri[0]],ref_points[tri[1]],ref_points[tri[2]]]
        t2 = [points[tri[0]],points[tri[1]],points[tri[2]]]

        warp_triangle(ref_img, warped, t1, t2)

    warped = cv2.GaussianBlur(warped,(5,5),0)

    warped = cv2.normalize(warped,None,0,255,cv2.NORM_MINMAX)

    hull = cv2.convexHull(points)

    top = np.min(points[:,1])
    left = np.min(points[:,0])
    right = np.max(points[:,0])

    extra = np.array([
        [[left, top-70]],
        [[right, top-70]]
    ], dtype=np.int32)

    hull = np.concatenate((hull, extra), axis=0)

    mask = np.zeros(frame.shape[:2], np.uint8)

    cv2.fillConvexPoly(mask, hull, 255)

    mask = cv2.GaussianBlur(mask,(61,61),0)

    r = cv2.boundingRect(hull)

    # ---- FIX TO PREVENT SEAMLESSCLONE CRASH ----
    h_frame, w_frame = frame.shape[:2]

    cx = r[0] + int(r[2]/2)
    cy = r[1] + int(r[3]/2)

    cx = max(0, min(cx, w_frame-1))
    cy = max(0, min(cy, h_frame-1))

    center = (cx, cy)
    # --------------------------------------------

    output = cv2.seamlessClone(
        np.uint8(warped),
        frame,
        mask,
        center,
        cv2.MIXED_CLONE
    )

    cv2.imshow("Face Swap",output)

    if cv2.waitKey(1)==27:
        break


cap.release()
cv2.destroyAllWindows()