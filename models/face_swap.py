import cv2
import numpy as np
import mediapipe as mp
import os

mp_face_mesh = mp.solutions.face_mesh

def get_landmarks(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_image)
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0]
            h, w, _ = image.shape
            return [(int(pt.x * w), int(pt.y * h)) for pt in landmarks.landmark]
    return None

def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def get_triangles(landmarks, size):
    subdiv = cv2.Subdiv2D((0, 0, size[1], size[0]))
    for p in landmarks:
        subdiv.insert(p)
    triangle_list = subdiv.getTriangleList()
    indexes = []
    for t in triangle_list:
        pts = [(int(t[i]), int(t[i + 1])) for i in range(0, 6, 2)]
        idx = []
        for pt in pts:
            if pt in landmarks:
                idx.append(landmarks.index(pt))
        if len(idx) == 3:
            indexes.append(tuple(idx))
    return indexes

def warp_triangle(src, dst, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    t1_rect = []
    t2_rect = []

    for i in range(3):
        t1_rect.append(((t_src[i][0] - r1[0]), (t_src[i][1] - r1[1])))
        t2_rect.append(((t_dst[i][0] - r2[0]), (t_dst[i][1] - r2[1])))

    src_crop = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    if src_crop.size == 0:
        return

    warp_img = apply_affine_transform(src_crop, t1_rect, t2_rect, (r2[2], r2[3]))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)

    dst_slice = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    if dst_slice.shape == warp_img.shape:
        dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dst_slice * (1 - mask) + warp_img * mask

def swap_faces(source_path, target_path):
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)

    if source_img is None or target_img is None:
        raise ValueError("One or both images could not be loaded.")

    source_points = get_landmarks(source_img)
    target_points = get_landmarks(target_img)

    if source_points is None or target_points is None:
        print("[ERROR] Could not detect faces.")
        return target_path

    target_face = np.zeros_like(target_img)

    triangles = get_triangles(target_points, target_img.shape)

    for tri in triangles:
        x, y, z = tri
        t1 = [source_points[x], source_points[y], source_points[z]]
        t2 = [target_points[x], target_points[y], target_points[z]]
        warp_triangle(source_img, target_face, t1, t2)

    mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(np.array(target_points)).astype(np.int32)
    cv2.fillConvexPoly(mask, hull, 255)

    # Use moments to find center for seamlessClone
    M = cv2.moments(hull)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        center = (cx, cy)
    else:
        center = (target_img.shape[1] // 2, target_img.shape[0] // 2)

    result = cv2.seamlessClone(target_face, target_img, mask, center, cv2.NORMAL_CLONE)

    result_path = os.path.join("static/results", "face_swap.jpg")
    cv2.imwrite(result_path, result)
    return result_path
