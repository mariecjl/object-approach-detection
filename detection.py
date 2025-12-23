import cv2
import numpy as np

def detect_approach(depth, depth_buffer, frame_shape, config):
    h, w = frame_shape

    prev_depth = np.mean(list(depth_buffer)[:-1], axis=0)
    delta = depth - prev_depth

    # adaptive threshold
    delta_med = np.median(delta)
    delta_mad = np.median(np.abs(delta - delta_med)) + 1e-6
    adaptive_thresh = delta_med + 2.0 * delta_mad

    delta_smooth = cv2.GaussianBlur(delta, (15, 15), 0)

    # center mask
    cx, cy = w // 2, h // 2
    rx, ry = int(w * config.CENTER_WEIGHT_RADIUS), int(h * config.CENTER_WEIGHT_RADIUS)

    center_mask = np.zeros((h, w), dtype=np.uint8)
    center_mask[cy - ry:cy + ry, cx - rx:cx + rx] = 1

    depth_near = depth > np.percentile(depth, 70)

    approach_mask = (
        (delta > adaptive_thresh) &
        (delta_smooth > adaptive_thresh * 0.8) &
        depth_near &
        center_mask
    ).astype(np.uint8) * 255

    kernel = np.ones((7, 7), np.uint8)
    approach_mask = cv2.morphologyEx(approach_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        approach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = config.MIN_AREA_RATIO * (h * w)
    main_cnt = max(
        (c for c in contours if cv2.contourArea(c) > min_area),
        key=cv2.contourArea,
        default=None
    )

    return main_cnt, delta
