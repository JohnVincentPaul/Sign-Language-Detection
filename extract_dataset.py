import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# ---------------- CONFIG ----------------
DATASET_FOLDER = "original_images"     # contains subfolders: 0-9, A-Z
OUTPUT_CSV = "dataset_features.csv"

HANDS_MIN_DET_CONF = 0.1
POSE_MIN_DET_CONF = 0.2

MAX_DIM_HANDS = 720
MAX_DIM_POSE = 640

SAVE_EVERY_ROWS = 500   # periodically flush to CSV so you don't lose progress
USE_POSE_CROPS = True   # set False to disable pose step (faster, fewer detections)
# ----------------------------------------


def resize_down_keep_aspect(bgr, max_dim):
    h, w = bgr.shape[:2]
    s = max_dim / max(h, w)
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return bgr


def preprocess_rgb(bgr, max_dim, contrast_norm=False):
    bgr2 = resize_down_keep_aspect(bgr, max_dim=max_dim)
    rgb = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    if contrast_norm:
        rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
    return rgb


def extract_features_v3_from_hands(global_hands):
    """
    global_hands: list of hands, each is list[(x_norm, y_norm)] length 21
    Uses your V3 math, and assigns left/right by wrist x-position.
    """
    if not global_hands:
        return None

    wrists = [(hand[0][0], hand) for hand in global_hands]  # (wrist_x, hand_pts)
    wrists.sort(key=lambda t: t[0])

    left_raw = wrists[0][1]
    right_raw = wrists[-1][1] if len(wrists) > 1 else None

    has_left = left_raw is not None
    has_right = right_raw is not None
    if not has_left and not has_right:
        return None

    anchor = left_raw[0] if has_left else right_raw[0]
    ax, ay = anchor

    ref_hand = left_raw if has_left else right_raw
    dx = ref_hand[9][0] - ax
    dy = ref_hand[9][1] - ay
    scale = float(np.sqrt(dx * dx + dy * dy) + 1e-6)

    features = {}

    for i in range(21):
        if has_left:
            features[f"left_x_{i}"] = (left_raw[i][0] - ax) / scale
            features[f"left_y_{i}"] = (left_raw[i][1] - ay) / scale
        else:
            features[f"left_x_{i}"] = 0.0
            features[f"left_y_{i}"] = 0.0

    for i in range(21):
        if has_right:
            features[f"right_x_{i}"] = (right_raw[i][0] - ax) / scale
            features[f"right_y_{i}"] = (right_raw[i][1] - ay) / scale
        else:
            features[f"right_x_{i}"] = 0.0
            features[f"right_y_{i}"] = 0.0

    return features


def hands_to_global_hands(results):
    if not results or not results.multi_hand_landmarks:
        return []
    out = []
    for hand_lms in results.multi_hand_landmarks:
        pts = [(float(lm.x), float(lm.y)) for lm in hand_lms.landmark]
        out.append(pts)
    return out


def try_hands_on_bgr(bgr, hands):
    # downscale + RGB; landmarks are normalized so resizing keeps them consistent
    rgb = preprocess_rgb(bgr, max_dim=MAX_DIM_HANDS, contrast_norm=True)
    results = hands.process(rgb)
    global_hands = hands_to_global_hands(results)
    return extract_features_v3_from_hands(global_hands)


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    return x1, y1, x2, y2


def crop_around_point(bgr, cx, cy, box_size):
    h, w = bgr.shape[:2]
    half = box_size / 2
    box = clamp_box(cx - half, cy - half, cx + half, cy + half, w, h)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop = bgr[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2 - x1, y2 - y1)  # ox, oy, cw, ch


def pose_wrist_crops(bgr, pose_results):
    if not pose_results or not pose_results.pose_landmarks:
        return []

    h, w = bgr.shape[:2]
    lm = pose_results.pose_landmarks.landmark
    PL = mp.solutions.pose.PoseLandmark

    L_WRIST = PL.LEFT_WRIST.value
    R_WRIST = PL.RIGHT_WRIST.value
    L_ELBOW = PL.LEFT_ELBOW.value
    R_ELBOW = PL.RIGHT_ELBOW.value

    def vis_ok(i):
        v = getattr(lm[i], "visibility", 1.0)
        return v is None or v > 0.3

    crops = []

    for wrist_i, elbow_i in [(L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW)]:
        if not vis_ok(wrist_i):
            continue

        wx, wy = lm[wrist_i].x * w, lm[wrist_i].y * h

        if vis_ok(elbow_i):
            ex, ey = lm[elbow_i].x * w, lm[elbow_i].y * h
            dist = float(np.sqrt((wx - ex) ** 2 + (wy - ey) ** 2))
            base = max(90.0, dist * 2.4)
        else:
            base = max(110.0, min(w, h) * 0.35)

        for mul in (1.0, 1.35, 1.7):
            out = crop_around_point(bgr, wx, wy, base * mul)
            if out is not None:
                crops.append(out)

    return crops


def try_hands_on_crop(crop_bgr, crop_rect, hands, full_w, full_h):
    ox, oy, cw, ch = crop_rect
    rgb = preprocess_rgb(crop_bgr, max_dim=MAX_DIM_HANDS, contrast_norm=True)
    results = hands.process(rgb)

    if not results or not results.multi_hand_landmarks:
        return None

    global_hands = []
    for hand_lms in results.multi_hand_landmarks:
        pts = []
        for lm in hand_lms.landmark:
            # lm.x/lm.y are normalized in the crop (resizing doesn't change normalization meaning)
            fx = (ox + lm.x * cw) / full_w
            fy = (oy + lm.y * ch) / full_h
            pts.append((float(fx), float(fy)))
        global_hands.append(pts)

    return extract_features_v3_from_hands(global_hands)


def flush_rows(buffer_rows, output_csv):
    if not buffer_rows:
        return 0
    df = pd.DataFrame(buffer_rows)
    write_header = not os.path.exists(output_csv)
    df.to_csv(output_csv, mode="a", header=write_header, index=False)
    return len(buffer_rows)


def main():
    if not os.path.isdir(DATASET_FOLDER):
        raise FileNotFoundError(
            f"Folder not found: {DATASET_FOLDER}\n"
            f"Set DATASET_FOLDER to the folder that contains 0-9 and A-Z subfolders."
        )

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=HANDS_MIN_DET_CONF,
    )

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,                 # FAST (this is the key speed fix)
        min_detection_confidence=POSE_MIN_DET_CONF,
        enable_segmentation=False,
        smooth_landmarks=False,
    )

    buffer_rows = []
    total_written = 0

    print(f"Starting extraction from {DATASET_FOLDER}...")

    try:
        for label in sorted(os.listdir(DATASET_FOLDER)):
            folder_path = os.path.join(DATASET_FOLDER, label)
            if not os.path.isdir(folder_path):
                continue

            print(f"\n--- Processing folder: {label} ---")
            success_count = 0
            total_files = 0
            fail_examples = 0

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if os.path.isdir(img_path) or img_name.startswith("."):
                    continue

                total_files += 1
                image = cv2.imread(img_path)
                if image is None:
                    continue

                features = None

                # Attempt 1: direct hands
                features = try_hands_on_bgr(image, hands)

                # Attempt 2: padded
                if features is None:
                    h, w = image.shape[:2]
                    pad_h, pad_w = int(h * 0.25), int(w * 0.25)
                    padded = cv2.copyMakeBorder(
                        image, pad_h, pad_h, pad_w, pad_w,
                        cv2.BORDER_CONSTANT, value=[0, 0, 0]
                    )
                    features = try_hands_on_bgr(padded, hands)

                # Attempt 3: pose-guided wrist crops (only if enabled)
                if features is None and USE_POSE_CROPS:
                    # Run pose ONCE on a resized RGB image (fast)
                    pose_rgb = preprocess_rgb(image, max_dim=MAX_DIM_POSE, contrast_norm=False)
                    pose_results = pose.process(pose_rgb)

                    crops = pose_wrist_crops(image, pose_results)
                    H, W = image.shape[:2]
                    for crop_bgr, crop_rect in crops:
                        features = try_hands_on_crop(crop_bgr, crop_rect, hands, W, H)
                        if features is not None:
                            break

                if features is not None:
                    features["label"] = label
                    buffer_rows.append(features)
                    success_count += 1

                    if len(buffer_rows) >= SAVE_EVERY_ROWS:
                        wrote = flush_rows(buffer_rows, OUTPUT_CSV)
                        total_written += wrote
                        buffer_rows.clear()
                        print(f"    (saved {total_written} rows so far...)")
                else:
                    if fail_examples < 5:
                        print(f" [WARN] No hands detected in: {img_name}")
                        fail_examples += 1

            print(f" -> {success_count} out of {total_files} valid images extracted for {label}.")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C). Saving progress...")

    finally:
        if buffer_rows:
            wrote = flush_rows(buffer_rows, OUTPUT_CSV)
            total_written += wrote
            buffer_rows.clear()

        hands.close()
        pose.close()

    if os.path.exists(OUTPUT_CSV):
        print(f"\n✅ Done. CSV is at: {OUTPUT_CSV} (rows written this run: {total_written})")
    else:
        print("\n❌ No CSV written (0 samples).")


if __name__ == "__main__":
    main()







