def run_tracking(video_path):

    import cv2
    import numpy as np
    import csv

    if not video_path:
        raise RuntimeError("No video provided.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Video failed to open.")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame.")

    # -------------------------
    # CALIBRATION: 1 INCH SCALE (draw a line = 1 inch)
    # -------------------------
    calibration_points = []
    drawing_img = frame.copy()

    def click_event(event, x, y, flags, param):
        nonlocal drawing_img
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(calibration_points) < 2:
                calibration_points.append((x, y))
            else:
                calibration_points.clear()
                calibration_points.append((x, y))
            drawing_img = frame.copy()
            for p in calibration_points:
                cv2.circle(drawing_img, p, 5, (0, 0, 255), -1)
            if len(calibration_points) == 2:
                cv2.line(drawing_img, calibration_points[0], calibration_points[1], (0, 255, 0), 2)

    cv2.namedWindow("Calibration (draw 1 inch line)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration (draw 1 inch line)", click_event)

    px_per_inch = None
    print("Click TWO points that represent 1 inch in the real world. Press 'c' or Enter to confirm, 'r' to reset.")

    while True:
        temp = drawing_img.copy()
        if len(calibration_points) == 2:
            (x1, y1), (x2, y2) = calibration_points
            length_px = np.hypot(x2 - x1, y2 - y1)
            cv2.putText(
                temp,
                f"Pixels: {length_px:.2f}  (press 'c' or Enter to confirm, 'r' to reset)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        else:
            cv2.putText(
                temp,
                "Click 2 points = 1 inch reference",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.imshow("Calibration (draw 1 inch line)", temp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('c'), 13):  # 'c' or Enter to confirm
            if len(calibration_points) == 2:
                (x1, y1), (x2, y2) = calibration_points
                px_per_inch = np.hypot(x2 - x1, y2 - y1)
                break
        elif key == ord('r'):  # reset
            calibration_points.clear()
            drawing_img = frame.copy()
        elif key == 27:  # Esc to cancel
            cv2.destroyWindow("Calibration (draw 1 inch line)")
            raise RuntimeError("Calibration canceled by user.")

    cv2.destroyWindow("Calibration (draw 1 inch line)")
    cv2.setMouseCallback("Calibration (draw 1 inch line)", lambda *args: None)

    if px_per_inch is None or px_per_inch <= 0:
        raise RuntimeError("Invalid calibration measurement.")

    print(f"Calibration complete: {px_per_inch:.2f} pixels per inch")


    # -------------------------
    # TRACKER
    # -------------------------
    def create_tracker():
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        elif hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        else:
            return cv2.TrackerKCF_create()

    # -------------------------
    # ROI SELECTION
    # -------------------------
    scale = 0.5
    frame_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    cv2.namedWindow("Select Object", cv2.WINDOW_NORMAL)
    bbox_small = cv2.selectROI("Select Object", frame_small, False)
    cv2.destroyWindow("Select Object")
    for _ in range(5):
        cv2.waitKey(1)

    x, y, w, h = bbox_small
    bbox = (
        int(x / scale),
        int(y / scale),
        int(w / scale),
        int(h / scale),
    )

    tracker = create_tracker()
    tracker.init(frame, bbox)

    # -------------------------
    # STATE
    # -------------------------
    centers = []
    trail = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    smooth_cx, smooth_cy = None, None
    vx, vy = 0.0, 0.0

    alpha = 0.35
    beta = 0.75
    display_scale = 0.6

    cv2.namedWindow("Tracking View", cv2.WINDOW_NORMAL)

    # -------------------------
    # MAIN LOOP
    # -------------------------
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = map(int, bbox)

                roi = frame[y:y + h, x:x + w]
                if roi.size > 0:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, mask = cv2.threshold(
                        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    M = cv2.moments(mask)
                    dx = M["m10"] / M["m00"] if M["m00"] != 0 else w / 2
                    dy = M["m01"] / M["m00"] if M["m00"] != 0 else h / 2
                else:
                    dx, dy = w / 2, h / 2

                cx = x + dx
                cy = y + dy

                valid = True
                if len(centers) > 2:
                    px, py = centers[-1]
                    if abs(cx - px) > 150 or abs(cy - py) > 150:
                        valid = False

                if valid:
                    if smooth_cx is None:
                        smooth_cx, smooth_cy = cx, cy
                    else:
                        vx = beta * vx + (1 - beta) * (cx - smooth_cx)
                        vy = beta * vy + (1 - beta) * (cy - smooth_cy)
                        smooth_cx += vx
                        smooth_cy += vy
                        smooth_cx = alpha * cx + (1 - alpha) * smooth_cx
                        smooth_cy = alpha * cy + (1 - alpha) * smooth_cy

                    centers.append((smooth_cx, smooth_cy))
                    trail.append((int(smooth_cx), int(smooth_cy)))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if smooth_cx is not None:
                    cv2.circle(frame, (int(smooth_cx), int(smooth_cy)), 5, (0, 0, 255), -1)
                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i - 1], trail[i], (255, 0, 0), 2)

            cv2.putText(
                frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )

            frame_display = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
            cv2.imshow("Tracking View", frame_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty("Tracking View", cv2.WND_PROP_AUTOSIZE) < 0:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

    # -------------------------
    # SAVE CSV
    # -------------------------
    if centers:
        with open("centers.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "x", "y"])
            for i, c in enumerate(centers):
                writer.writerow([i / fps, c[0], c[1]])

    return centers, fps, px_per_inch