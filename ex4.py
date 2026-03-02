import os
import cv2
import numpy as np
from PIL import Image


def generate_panorama(input_frames_path, n_out_frames):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    frame_names = sorted([f for f in os.listdir(input_frames_path) if f.lower().endswith(valid_extensions)])

    raw_frames = []
    for name in frame_names:
        img = cv2.imread(os.path.join(input_frames_path, name))
        if img is not None:
            raw_frames.append(img)

    if len(raw_frames) < 2:
        return []
    h, w, _ = raw_frames[0].shape

    # 1. MOTION TRACKING
    orb = cv2.ORB_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    accum_tx, accum_ty = [0.0], [0.0]

    for i in range(len(raw_frames) - 1):
        kp1, des1 = orb.detectAndCompute(raw_frames[i], None)
        kp2, des2 = orb.detectAndCompute(raw_frames[i + 1], None)
        dx, dy = 0, 0
        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            if len(matches) > 10:
                diffs = np.array([(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0],
                                   kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) for m in matches])
                dx, dy = np.median(diffs, axis=0)
        accum_tx.append(accum_tx[-1] + dx)
        accum_ty.append(accum_ty[-1] + dy)

    # 2. CANVAS CALCULATION
    min_tx, max_tx = min(accum_tx), max(accum_tx)
    min_ty, max_ty = min(accum_ty), max(accum_ty)
    pano_w = int(round(max_tx - min_tx)) + w
    pano_h = int(round(max_ty - min_ty)) + h

    # 3. MULTI-VIEW GENERATION
    view_offsets = np.linspace(w // 4, 3 * w // 4, n_out_frames).astype(int)
    output_panos = []

    for center_x in view_offsets:
        # Create canvas and a 'mask' to track which pixels are filled
        canvas = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
        filled_mask = np.zeros((pano_h, pano_w), dtype=np.uint8)

        for i in range(len(raw_frames)):
            tx = int(round(accum_tx[i] - min_tx))
            ty = int(round(accum_ty[i] - min_ty))

            win_w = 100
            x1, x2 = max(0, center_x - win_w), min(w, center_x + win_w)

            strip = raw_frames[i][:, x1:x2]
            sx_start = tx + x1

            # Place the strip on the canvas
            # The 'filled_mask' ensures we don't leave gaps, and later frames
            # fill in where earlier ones couldn't reach.
            sh, sw = strip.shape[:2]
            if sx_start + sw <= pano_w and ty + sh <= pano_h:

                # Only update pixels that haven't been filled yet to maintain perspective
                # or overwrite to ensure zero black gaps.
                canvas[ty:ty + sh, sx_start:sx_start + sw] = strip
                filled_mask[ty:ty + sh, sx_start:sx_start + sw] = 255

        # Final Crop to remove unused canvas edges
        coords = cv2.findNonZero(filled_mask)
        if coords is not None:
            rx, ry, rw, rh = cv2.boundingRect(coords)
            canvas = canvas[ry:ry + rh, rx:rx + rw]

        output_panos.append(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))

    return output_panos


def video_to_panorama_boomerang(video_path, output_path, n_out_frames=30, fps=24):
    # 1. קריאת הסרטון
    cap = cv2.VideoCapture(video_path)
    raw_frames = []

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    if len(raw_frames) < 2:
        print("Error: Not enough frames to process.")
        return

    h, w, _ = raw_frames[0].shape

    # 2. חישוב תנועה (ORB)
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    accum_tx, accum_ty = [0.0], [0.0]

    for i in range(len(raw_frames) - 1):
        kp1, des1 = orb.detectAndCompute(raw_frames[i], None)
        kp2, des2 = orb.detectAndCompute(raw_frames[i + 1], None)
        dx, dy = 0, 0
        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            if len(matches) > 10:
                diffs = np.array([(kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0],
                                   kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) for m in matches])
                dx, dy = np.median(diffs, axis=0)
        accum_tx.append(accum_tx[-1] + dx)
        accum_ty.append(accum_ty[-1] + dy)

    min_tx, max_tx = min(accum_tx), max(accum_tx)
    min_ty, max_ty = min(accum_ty), max(accum_ty)
    pano_w = int(round(max_tx - min_tx)) + w
    pano_h = int(round(max_ty - min_ty)) + h

    # 3. יצירת פריימים פנורמיים
    view_offsets = np.linspace(w // 4, 3 * w // 4, n_out_frames).astype(int)
    panorama_sequence = []

    for center_x in view_offsets:
        canvas = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
        filled_mask = np.zeros((pano_h, pano_w), dtype=np.uint8)

        for i in range(len(raw_frames)):
            tx = int(round(accum_tx[i] - min_tx))
            ty = int(round(accum_ty[i] - min_ty))
            win_w = 100
            x1, x2 = max(0, center_x - win_w), min(w, center_x + win_w)
            strip = raw_frames[i][:, x1:x2]
            sx_start = tx + x1
            sh, sw = strip.shape[:2]

            if sx_start + sw <= pano_w and ty + sh <= pano_h:
                canvas[ty:ty + sh, sx_start:sx_start + sw] = strip
                filled_mask[ty:ty + sh, sx_start:sx_start + sw] = 255

        coords = cv2.findNonZero(filled_mask)
        if coords is not None:
            rx, ry, rw, rh = cv2.boundingRect(coords)
            cropped = canvas[ry:ry + rh, rx:rx + rw]
            panorama_sequence.append(cropped)

    if not panorama_sequence:
        return

    # 4. הכנת רצף הלוך-חזור (Boomerang)
    # מוסיפים את ההיפוך של הרצף כדי ליצור לופ חלק
    full_sequence = panorama_sequence + panorama_sequence[-2:0:-1]

    # 5. שמירה לקובץ וידאו - קריטי להשתמש בגודל אחיד
    # נקבע גודל סופי לפי הפריים הראשון ברצף
    target_h, target_w = full_sequence[0].shape[:2]

    # שימוש במקודד נפוץ יותר (XVID עבור .avi או mp4v עבור .mp4)
    # אם mp4v לא עובד, נסה 'XVID' וסיומת .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    for frame in full_sequence:
        # חובה לבצע resize כי כל פריים פנורמי עשוי לצאת בגודל מעט שונה
        res_frame = cv2.resize(frame, (target_w, target_h))
        out.write(res_frame)

    out.release()  # סגירה מסודרת של הקובץ
    print(f"Done! Saved to {output_path}")



if __name__ == "__main__":
     video_file = "bad_input.mp4"
     output_dir = "bad_panorama"
     os.makedirs(output_dir, exist_ok=True)
     temp_frames_dir = "temp"

     # 1. PRE-PROCESS: Extract Video to Directory
     print(f"Extracting frames from {video_file}...")
     cap = cv2.VideoCapture(video_file)
     count = 0
     while True:
         ret, frame = cap.read()
         if not ret:
             break
         # Save in the format specified in the API: frame_i:05d.jpg
         cv2.imwrite(os.path.join(temp_frames_dir, f"frame_{count:05d}.jpg"), frame)
         count += 1
     cap.release()
     print(f"Extracted {count} frames to {temp_frames_dir}.")
     panos = generate_panorama(temp_frames_dir, 10)
     for i, p in enumerate(panos):
         p.save(os.path.join(output_dir, f"view_{i:02d}.jpg"))
