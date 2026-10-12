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
