import cv2
import numpy as np
import mediapipe as mp
import os

mp_face = mp.solutions.face_mesh

def extract_landmarks(image):
    if image is None:
        print("❌ Image is empty")
        return None
    with mp_face.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            print("❌ No face detected in image")
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        return [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) for lm in landmarks]

def animate_motion(source_path, driving_path):
    result_path = os.path.join('static', 'results', 'motion_transfer.mp4')

    source_img = cv2.imread(source_path)
    source_lm = extract_landmarks(source_img)
    if not source_lm:
        print("❌ No face detected in source image")
        return None

    cap = cv2.VideoCapture(driving_path)
    if not cap.isOpened():
        print("❌ Error opening video file")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    with mp_face.FaceMesh(static_image_mode=False) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                print("⚠️ No face detected in frame")
                out.write(frame)
                continue

            driving_lm = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in results.multi_face_landmarks[0].landmark]

            try:
                src_pts = np.float32([source_lm[33], source_lm[263], source_lm[1]])  # Left eye, right eye, nose tip
                dst_pts = np.float32([driving_lm[33], driving_lm[263], driving_lm[1]])
            except IndexError:
                print("⚠️ Landmark index out of bounds")
                out.write(frame)
                continue

            matrix = cv2.getAffineTransform(src_pts, dst_pts)
            warped = cv2.warpAffine(source_img, matrix, (frame.shape[1], frame.shape[0]))

            hull_indices = cv2.convexHull(np.array(driving_lm), returnPoints=False)
            hull_points = np.array([driving_lm[i[0]] for i in hull_indices])

            if len(hull_points) < 3:
                print("⚠️ Not enough points for convex hull")
                out.write(frame)
                continue

            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull_points, 255)

            warped_face = cv2.bitwise_and(warped, warped, mask=mask)
            background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            blended = cv2.add(background, warped_face)

            out.write(blended)

    cap.release()
    out.release()
    print("✅ Motion transfer complete:", result_path)
    return result_path
