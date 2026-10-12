import math
from itertools import combinations


def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def calculate_distance(point_a, point_b):
    return math.hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1]
    )


def calculate_speeds(trajectory):
    frames = sorted(
        (int(frame), data["center"])
        for frame, data in trajectory.items()
    )

    speeds = {}

    for i in range(1, len(frames)):
        previous_frame, previous_center = frames[i - 1]
        current_frame, current_center = frames[i]

        frame_delta = current_frame - previous_frame

        if frame_delta <= 0:
            continue

        speed = calculate_distance(
            previous_center,
            current_center
        ) / frame_delta

        speeds[current_frame] = speed

    return speeds


def detect_abrupt_deceleration(speeds, threshold=0.5):
    frames = sorted(speeds)

    for i in range(1, len(frames)):
        previous_speed = speeds[frames[i - 1]]
        current_speed = speeds[frames[i]]

        if previous_speed <= 0:
            continue

        relative_drop = (
            previous_speed - current_speed
        ) / previous_speed

        if relative_drop >= threshold:
            return {
                "detected": True,
                "frame": frames[i],
                "relative_drop": relative_drop
            }

    return {
        "detected": False,
        "frame": None,
        "relative_drop": 0.0
    }


def analyze_relative_motion(
    trajectory_a,
    trajectory_b,
    lookback=5
):
    common_frames = sorted(
        set(trajectory_a) & set(trajectory_b),
        key=int
    )

    if len(common_frames) < lookback + 1:
        return {
            "detected": False,
            "frame": None,
            "distance_before": None,
            "distance_after": None
        }

    for i in range(lookback, len(common_frames)):
        previous_frame = common_frames[i - lookback]
        current_frame = common_frames[i]

        distance_before = calculate_distance(
            trajectory_a[previous_frame]["center"],
            trajectory_b[previous_frame]["center"]
        )

        distance_after = calculate_distance(
            trajectory_a[current_frame]["center"],
            trajectory_b[current_frame]["center"]
        )

        if distance_before <= 0:
            continue

        relative_closing = (
            distance_before - distance_after
        ) / distance_before

        if relative_closing >= 0.15:
            return {
                "detected": True,
                "frame": int(current_frame),
                "distance_before": distance_before,
                "distance_after": distance_after
            }

    return {
        "detected": False,
        "frame": None,
        "distance_before": None,
        "distance_after": None
    }


def identify_accident_candidates(
    trajectories,
    min_iou=0.1,
    deceleration_threshold=0.5,
    lookback=5
):
    candidates = []

    vehicle_ids = list(trajectories.keys())

    for id_a, id_b in combinations(vehicle_ids, 2):
        vehicle_a = trajectories[id_a]
        vehicle_b = trajectories[id_b]

        trajectory_a = vehicle_a["trajectory"]
        trajectory_b = vehicle_b["trajectory"]

        common_frames = sorted(
            set(trajectory_a) & set(trajectory_b),
            key=int
        )

        if not common_frames:
            continue

        max_iou = 0.0
        overlap_frame = None

        for frame in common_frames:
            iou = calculate_iou(
                trajectory_a[frame]["box"],
                trajectory_b[frame]["box"]
            )

            if iou > max_iou:
                max_iou = iou
                overlap_frame = int(frame)

        speeds_a = calculate_speeds(trajectory_a)
        speeds_b = calculate_speeds(trajectory_b)

        deceleration_a = detect_abrupt_deceleration(
            speeds_a,
            deceleration_threshold
        )

        deceleration_b = detect_abrupt_deceleration(
            speeds_b,
            deceleration_threshold
        )

        relative_motion = analyze_relative_motion(
            trajectory_a,
            trajectory_b,
            lookback
        )

        overlap_detected = max_iou >= min_iou
        deceleration_detected = (
            deceleration_a["detected"]
            or deceleration_b["detected"]
        )

        motion_detected = relative_motion["detected"]

        score = (
            0.4 * overlap_detected
            + 0.3 * motion_detected
            + 0.3 * deceleration_detected
        )

        if score < 0.3:
            continue

        evidence = []

        if overlap_detected:
            evidence.append("bounding_box_overlap")

        if motion_detected:
            evidence.append("relative_closing_motion")

        if deceleration_a["detected"]:
            evidence.append(
                f"vehicle_{id_a}_sudden_deceleration"
            )

        if deceleration_b["detected"]:
            evidence.append(
                f"vehicle_{id_b}_sudden_deceleration"
            )

        candidates.append({
            "vehicle_ids": [str(id_a), str(id_b)],
            "classes": [
                vehicle_a["class"],
                vehicle_b["class"]
            ],
            "suspicion_score": round(score, 3),
            "max_iou": round(max_iou, 3),
            "overlap_frame": overlap_frame,
            "relative_motion": relative_motion,
            "deceleration": {
                str(id_a): deceleration_a,
                str(id_b): deceleration_b
            },
            "evidence": evidence
        })

    candidates.sort(
        key=lambda item: item["suspicion_score"],
        reverse=True
    )

    return candidates


def format_tracking_summary(trajectories, fps=30.0, candidates=None):
    """Format vehicle trajectories and collision candidate alerts into a concise

    ground truth summary for prompting Vision-Language Models (VLMs).
    """
    if not trajectories:
        return "No vehicles were detected or tracked by the object detector."

    if candidates is None:
        candidates = identify_accident_candidates(trajectories)

    lines = []
    lines.append(f"DETECTED ROAD USERS & TRACKING TELEMETRY (Total: {len(trajectories)} tracked):")

    for vid, data in sorted(
        trajectories.items(),
        key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0])
    ):
        cls_name = data.get("class", "vehicle")
        frames = sorted(int(f) for f in data.get("trajectory", {}).keys())
        if not frames:
            continue
        start_t = frames[0] / fps if fps > 0 else 0.0
        end_t = frames[-1] / fps if fps > 0 else 0.0
        start_center = data["trajectory"][str(frames[0])]["center"]
        end_center = data["trajectory"][str(frames[-1])]["center"]

        # Check deceleration
        speeds = calculate_speeds(data["trajectory"])
        decel = detect_abrupt_deceleration(speeds)
        decel_str = ""
        if decel["detected"] and decel["frame"] is not None:
            decel_t = decel["frame"] / fps if fps > 0 else 0.0
            decel_str = f" | Sudden deceleration at {decel_t:.2f}s (-{decel['relative_drop']*100:.0f}%)"

        lines.append(
            f"- Vehicle ID {vid} ({cls_name}): visible from {start_t:.2f}s to {end_t:.2f}s "
            f"(frames {frames[0]}-{frames[-1]}), moved from ({start_center[0]:.0f}, {start_center[1]:.0f}) "
            f"to ({end_center[0]:.0f}, {end_center[1]:.0f}){decel_str}"
        )

    lines.append("\nTRACKING COLLISION ALERTS & CANDIDATE INTERACTIONS:")
    if candidates:
        for idx, cand in enumerate(candidates[:3], 1):
            v_pair = " and ".join(f"Vehicle ID {v}" for v in cand["vehicle_ids"])
            overlap_t = f"{cand['overlap_frame'] / fps:.2f}s" if (cand['overlap_frame'] is not None and fps > 0) else "N/A"
            evidence_str = ", ".join(cand["evidence"]) if cand.get("evidence") else "trajectory proximity"
            lines.append(
                f"  Alert {idx}: {v_pair} | Suspicion Score: {cand['suspicion_score']} | "
                f"Peak Box Overlap at {overlap_t} (IoU: {cand['max_iou']}) | Evidence: {evidence_str}"
            )
    else:
        lines.append("  No high-probability collisions or bounding-box overlaps detected by the tracker.")

    return "\n".join(lines)

