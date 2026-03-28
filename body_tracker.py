"""
Realistic Face & Body Swap with Expression Mirroring
The swapped face mirrors your expressions - blinks, smiles, etc.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import numpy as np
import sys


def download_models():
    """Download required models if not present."""
    models = {
        "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    }

    for filename, url in models.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)

    print("Models ready.")
    return "pose_landmarker_lite.task", "face_landmarker.task"


# Face landmark indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Eye landmarks for blink detection
LEFT_EYE_TOP = [159, 145]
LEFT_EYE_BOTTOM = [145, 153]
RIGHT_EYE_TOP = [386, 374]
RIGHT_EYE_BOTTOM = [374, 380]

# Mouth landmarks
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]


class ExpressionMirrorSwapper:
    """Face swap with expression mirroring."""

    def __init__(self, face_landmarker):
        self.face_landmarker = face_landmarker
        self.source_image = None
        self.source_landmarks = None
        self.source_blendshapes = None

    def load_source(self, image_path: str) -> bool:
        """Load source face image."""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image: {image_path}")
            return False

        self.source_image = img.copy()
        self.source_image_original = img.copy()

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            print("Error: No face detected in source image")
            return False

        h, w = img.shape[:2]
        landmarks = result.face_landmarks[0]

        self.source_landmarks = []
        for lm in landmarks:
            self.source_landmarks.append([lm.x * w, lm.y * h])
        self.source_landmarks = np.array(self.source_landmarks, dtype=np.float32)

        self.source_h, self.source_w = h, w

        print(f"Source face loaded: {image_path}")
        return True

    def get_eye_aspect_ratio(self, landmarks, eye_top_idx, eye_bottom_idx, h):
        """Calculate eye openness."""
        top = landmarks[eye_top_idx[0]].y * h
        bottom = landmarks[eye_bottom_idx[1]].y * h
        return abs(bottom - top)

    def apply_expression(self, target_landmarks, target_blendshapes, frame_shape):
        """Modify source face based on target expressions."""
        if self.source_landmarks is None:
            return self.source_image

        modified = self.source_image_original.copy()
        h, w = modified.shape[:2]

        if target_blendshapes is None:
            return modified

        # Get blendshape values
        scores = {}
        for bs in target_blendshapes:
            scores[bs.category_name] = bs.score

        # Eye blink - darken/close eyes when blinking
        left_blink = scores.get('eyeBlinkLeft', 0)
        right_blink = scores.get('eyeBlinkRight', 0)

        if left_blink > 0.3 or right_blink > 0.3:
            # Draw closed eyes over source
            for eye_indices in [[33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]]:
                pts = []
                for idx in eye_indices:
                    x = int(self.source_landmarks[idx][0])
                    y = int(self.source_landmarks[idx][1])
                    pts.append([x, y])
                pts = np.array(pts, dtype=np.int32)

                # Get skin color near eye
                eye_center = np.mean(pts, axis=0).astype(int)
                skin_color = modified[max(0, eye_center[1]-5):eye_center[1]+5,
                                      max(0, eye_center[0]-5):eye_center[0]+5].mean(axis=(0,1))

                # Draw eyelid closed
                cv2.fillPoly(modified, [pts], skin_color.astype(int).tolist())

                # Draw eyelid line
                top_pts = pts[:3]
                cv2.polylines(modified, [top_pts], False, (int(skin_color[0]*0.7), int(skin_color[1]*0.7), int(skin_color[2]*0.7)), 2)

        # Mouth open
        jaw_open = scores.get('jawOpen', 0)
        if jaw_open > 0.2:
            # Darken mouth interior
            mouth_pts = []
            for idx in [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]:
                if idx < len(self.source_landmarks):
                    x = int(self.source_landmarks[idx][0])
                    y = int(self.source_landmarks[idx][1])
                    mouth_pts.append([x, y])

            if len(mouth_pts) > 3:
                mouth_pts = np.array(mouth_pts, dtype=np.int32)
                # Expand mouth opening based on jaw_open
                center = np.mean(mouth_pts, axis=0)
                expanded = ((mouth_pts - center) * (1 + jaw_open * 0.3) + center).astype(np.int32)
                cv2.fillPoly(modified, [expanded], (20, 20, 30))  # Dark mouth interior

        return modified

    def warp_face(self, source_img, source_pts, target_pts, output_shape):
        """Warp source face to target position."""
        h, w = output_shape[:2]
        output = np.zeros((h, w, 3), dtype=np.uint8)

        # Add corner points
        corners = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1],
                           [w//2, 0], [0, h//2], [w-1, h//2], [w//2, h-1]], dtype=np.float32)

        src_pts_full = np.vstack([source_pts, corners])
        tgt_pts_full = np.vstack([target_pts, corners])

        # Delaunay triangulation
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)

        pt_to_idx = {}
        for i, pt in enumerate(tgt_pts_full):
            key = (int(pt[0]), int(pt[1]))
            pt_to_idx[key] = i
            try:
                subdiv.insert((float(pt[0]), float(pt[1])))
            except:
                pass

        triangles = subdiv.getTriangleList()

        for t in triangles:
            pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]

            indices = []
            for pt in pts:
                # Find closest point
                min_dist = float('inf')
                min_idx = 0
                for i, ref_pt in enumerate(tgt_pts_full):
                    dist = (pt[0] - ref_pt[0])**2 + (pt[1] - ref_pt[1])**2
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = i
                if min_dist < 100:
                    indices.append(min_idx)

            if len(indices) != 3 or len(set(indices)) != 3:
                continue

            src_tri = np.array([src_pts_full[i] for i in indices], dtype=np.float32)
            tgt_tri = np.array([tgt_pts_full[i] for i in indices], dtype=np.float32)

            self._warp_triangle(source_img, output, src_tri, tgt_tri)

        return output

    def _warp_triangle(self, src_img, dst_img, src_tri, dst_tri):
        """Warp single triangle."""
        src_rect = cv2.boundingRect(src_tri)
        dst_rect = cv2.boundingRect(dst_tri)

        src_h, src_w = src_img.shape[:2]
        dst_h, dst_w = dst_img.shape[:2]

        # Bounds check
        if (src_rect[0] < 0 or src_rect[1] < 0 or
            src_rect[0] + src_rect[2] > src_w or src_rect[1] + src_rect[3] > src_h or
            dst_rect[0] < 0 or dst_rect[1] < 0 or
            dst_rect[0] + dst_rect[2] > dst_w or dst_rect[1] + dst_rect[3] > dst_h or
            src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0):
            return

        src_tri_offset = src_tri - np.array([src_rect[0], src_rect[1]], dtype=np.float32)
        dst_tri_offset = dst_tri - np.array([dst_rect[0], dst_rect[1]], dtype=np.float32)

        src_crop = src_img[src_rect[1]:src_rect[1]+src_rect[3], src_rect[0]:src_rect[0]+src_rect[2]]
        if src_crop.size == 0:
            return

        try:
            warp_mat = cv2.getAffineTransform(src_tri_offset, dst_tri_offset)
            warped = cv2.warpAffine(src_crop, warp_mat, (dst_rect[2], dst_rect[3]),
                                    borderMode=cv2.BORDER_REFLECT_101)
        except:
            return

        mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_tri_offset), 255)

        dst_roi = dst_img[dst_rect[1]:dst_rect[1]+dst_rect[3], dst_rect[0]:dst_rect[0]+dst_rect[2]]
        mask_3ch = cv2.merge([mask, mask, mask])
        dst_roi[:] = np.where(mask_3ch > 0, warped, dst_roi)

    def color_correct(self, source, target, mask):
        """Match colors."""
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        mask_bool = mask > 128
        if not np.any(mask_bool):
            return source

        for i in range(3):
            src_mean = np.mean(source_lab[:,:,i][mask_bool])
            src_std = np.std(source_lab[:,:,i][mask_bool]) + 1e-6
            tgt_mean = np.mean(target_lab[:,:,i][mask_bool])
            tgt_std = np.std(target_lab[:,:,i][mask_bool]) + 1e-6

            source_lab[:,:,i] = (source_lab[:,:,i] - src_mean) * (tgt_std / src_std) + tgt_mean

        source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)

    def swap_face(self, frame, face_landmarks, blendshapes=None):
        """Full face swap with expression mirroring."""
        if self.source_landmarks is None:
            return frame

        h, w = frame.shape[:2]

        # Apply expression to source face
        modified_source = self.apply_expression(face_landmarks, blendshapes, frame.shape)

        # Get target landmarks
        target_landmarks = []
        for lm in face_landmarks:
            target_landmarks.append([lm.x * w, lm.y * h])
        target_landmarks = np.array(target_landmarks, dtype=np.float32)

        # Scale source landmarks to source image size
        src_pts = self.source_landmarks.copy()

        # Use key points for warping
        key_indices = list(set([33, 133, 362, 263, 1, 4, 61, 291, 199, 10, 152] + FACE_OVAL))

        src_key = np.array([src_pts[i] for i in key_indices], dtype=np.float32)
        tgt_key = np.array([target_landmarks[i] for i in key_indices], dtype=np.float32)

        # Warp face
        warped = self.warp_face(modified_source, src_key, tgt_key, frame.shape)

        # Create mask
        hull_pts = np.array([target_landmarks[i] for i in FACE_OVAL], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(hull_pts)
        cv2.fillConvexPoly(mask, hull, 255)

        # Erode and blur mask
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Color correct
        warped = self.color_correct(warped, frame, mask)

        # Find center for seamless clone
        moments = cv2.moments(hull)
        cx = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else w // 2
        cy = int(moments["m01"] / moments["m00"]) if moments["m00"] != 0 else h // 2
        cx = max(1, min(cx, w - 2))
        cy = max(1, min(cy, h - 2))

        # Seamless clone
        try:
            output = cv2.seamlessClone(warped, frame, mask, (cx, cy), cv2.NORMAL_CLONE)
        except:
            mask_f = mask.astype(np.float32) / 255.0
            mask_3 = cv2.merge([mask_f, mask_f, mask_f])
            output = (warped * mask_3 + frame * (1 - mask_3)).astype(np.uint8)

        return output


def draw_pose(frame, detection_result):
    """Draw pose skeleton."""
    if not detection_result.pose_landmarks:
        return frame

    for pose_landmarks in detection_result.pose_landmarks:
        h, w = frame.shape[:2]
        points = []
        for lm in pose_landmarks:
            points.append((int(lm.x * w), int(lm.y * h)))
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

        connections = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                       (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                       (24, 26), (26, 28)]

        for start, end in connections:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (0, 0, 255), 2)

    return frame


def main():
    print("=" * 50)
    print("REALISTIC FACE SWAP WITH EXPRESSIONS")
    print("=" * 50)
    print("\nYour expressions will be mirrored!")
    print("- Blink and the face blinks")
    print("- Open mouth and it opens")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Toggle skeleton")
    print("  'w' - Toggle face swap")
    print("  'l' - Load new face")
    print("-" * 50)

    pose_model, face_model = download_models()

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=pose_model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    face_options_video = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=face_model),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,  # Enable for expression detection
    )

    face_options_image = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=face_model),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )

    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    face_landmarker = vision.FaceLandmarker.create_from_options(face_options_video)
    face_landmarker_image = vision.FaceLandmarker.create_from_options(face_options_image)

    swapper = ExpressionMirrorSwapper(face_landmarker_image)

    if len(sys.argv) > 1:
        swapper.load_source(sys.argv[1])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    frame_timestamp_ms = 0
    show_skeleton = False
    enable_swap = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            pose_result = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            face_result = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 33

            # Get blendshapes for expression mirroring
            blendshapes = None
            if face_result.face_blendshapes:
                blendshapes = face_result.face_blendshapes[0]

            # Face swap with expressions
            if enable_swap and swapper.source_landmarks is not None and face_result.face_landmarks:
                frame = swapper.swap_face(frame, face_result.face_landmarks[0], blendshapes)

            if show_skeleton:
                frame = draw_pose(frame, pose_result)

            # UI
            cv2.rectangle(frame, (5, 5), (280, 90), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, 5), (280, 90), (80, 80, 80), 1)

            swap_on = enable_swap and swapper.source_landmarks is not None
            cv2.putText(frame, f"FACE SWAP: {'ON' if swap_on else 'OFF'}", (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if swap_on else (100, 100, 100), 2)

            if swapper.source_landmarks is None:
                cv2.putText(frame, "Press 'l' to load face", (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            else:
                # Show detected expression
                if blendshapes:
                    scores = {bs.category_name: bs.score for bs in blendshapes}
                    expr = []
                    if scores.get('eyeBlinkLeft', 0) > 0.4 or scores.get('eyeBlinkRight', 0) > 0.4:
                        expr.append("BLINK")
                    if scores.get('mouthSmileLeft', 0) > 0.4 or scores.get('mouthSmileRight', 0) > 0.4:
                        expr.append("SMILE")
                    if scores.get('jawOpen', 0) > 0.3:
                        expr.append("MOUTH OPEN")

                    expr_text = ", ".join(expr) if expr else "Neutral"
                    cv2.putText(frame, f"Expression: {expr_text}", (10, 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

                face_status = "Face: TRACKING" if face_result.face_landmarks else "Face: SEARCHING"
                cv2.putText(frame, face_status, (10, 78),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 255, 0) if face_result.face_landmarks else (0, 100, 255), 1)

            cv2.imshow('Face Swap', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_skeleton = not show_skeleton
            elif key == ord('w'):
                enable_swap = not enable_swap
            elif key == ord('l'):
                print("\nEnter path to face image: ", end="", flush=True)
                path = input().strip().strip('"').strip("'")
                if path:
                    swapper.load_source(path)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_landmarker.close()
        face_landmarker.close()
        face_landmarker_image.close()


if __name__ == "__main__":
    main()
