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


def filter_spurious_tracks(trajectories, min_frames=6):
    """Filter out short-lived ghost tracklets (fewer than min_frames)."""
    return {
        vid: data
        for vid, data in trajectories.items()
        if len(data.get("trajectory", {})) >= min_frames
    }


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


def smooth_speeds(speeds, window=3):
    """Apply moving average smoothing over speeds to eliminate single-frame detection jitter."""
    if not speeds or len(speeds) < window:
        return speeds
    frames = sorted(speeds.keys())
    smoothed = {}
    for i in range(len(frames)):
        start_idx = max(0, i - window + 1)
        pts = [speeds[frames[j]] for j in range(start_idx, i + 1)]
        smoothed[frames[i]] = sum(pts) / len(pts)
    return smoothed


def detect_abrupt_deceleration(speeds, threshold=0.5, min_moving_speed=2.0, window=3):
    """Detect significant, abrupt deceleration from a moving vehicle.

    Ignores stationary/parked vehicles with trivial speed (< min_moving_speed pixels/frame).
    Finds the peak deceleration event rather than stopping at the first small dip.
    """
    if len(speeds) < 3:
        return {
            "detected": False,
            "frame": None,
            "relative_drop": 0.0,
            "peak_speed": 0.0,
        }

    smoothed = smooth_speeds(speeds, window=window)
    frames = sorted(smoothed.keys())

    best_drop = 0.0
    best_frame = None
    best_peak_speed = 0.0

    for i in range(len(frames) - 2):
        prev_f = frames[i]
        curr_f = frames[min(i + 2, len(frames) - 1)]

        speed_before = smoothed[prev_f]
        speed_after = smoothed[curr_f]

        if speed_before < min_moving_speed:
            continue

        drop = (speed_before - speed_after) / speed_before
        if drop >= threshold and drop > best_drop:
            best_drop = drop
            best_frame = curr_f
            best_peak_speed = speed_before

    if best_drop >= threshold:
        return {
            "detected": True,
            "frame": best_frame,
            "relative_drop": round(best_drop, 2),
            "peak_speed": round(best_peak_speed, 1),
        }

    return {
        "detected": False,
        "frame": None,
        "relative_drop": 0.0,
        "peak_speed": 0.0,
    }


def analyze_relative_motion(
    trajectory_a,
    trajectory_b,
    lookback=5,
    proximity_threshold=200.0,
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
            "distance_after": None,
            "min_distance": None,
            "close_proximity": False,
        }

    min_distance = float("inf")
    closest_frame = None
    for f in common_frames:
        d = calculate_distance(
            trajectory_a[f]["center"],
            trajectory_b[f]["center"]
        )
        if d < min_distance:
            min_distance = d
            closest_frame = int(f)

    close_proximity = min_distance <= proximity_threshold

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

        # Relative closing is significant only when vehicles actually approach close proximity
        if relative_closing >= 0.15 and distance_after <= proximity_threshold:
            return {
                "detected": True,
                "frame": int(current_frame),
                "distance_before": round(distance_before, 1),
                "distance_after": round(distance_after, 1),
                "min_distance": round(min_distance, 1),
                "close_proximity": close_proximity,
            }

    return {
        "detected": False,
        "frame": closest_frame,
        "distance_before": None,
        "distance_after": None,
        "min_distance": round(min_distance, 1) if min_distance != float("inf") else None,
        "close_proximity": close_proximity,
    }


def identify_duplicate_tracks(trajectories):
    """Identify duplicate tracklets / tracker fragmentation (e.g. from vehicle flipping/deforming)."""
    vehicle_ids = list(trajectories.keys())
    duplicates = {}  # dup_id -> original_id

    for id_a, id_b in combinations(vehicle_ids, 2):
        t_a = trajectories[id_a]["trajectory"]
        t_b = trajectories[id_b]["trajectory"]
        common = sorted(set(t_a) & set(t_b), key=int)
        if not common:
            continue

        max_iou = max(calculate_iou(t_a[f]["box"], t_b[f]["box"]) for f in common)
        f0 = str(common[0])
        init_iou = calculate_iou(t_a[f0]["box"], t_b[f0]["box"])

        # If a tracklet starts directly on top of an existing one with near-identical IoU
        if init_iou >= 0.70 and max_iou >= 0.85:
            if len(t_a) <= len(t_b):
                duplicates[id_a] = id_b
            else:
                duplicates[id_b] = id_a

    return duplicates


def identify_accident_candidates(
    trajectories,
    min_iou=0.05,
    deceleration_threshold=0.5,
    lookback=5,
    min_track_frames=6,
):
    cleaned = filter_spurious_tracks(trajectories, min_frames=min_track_frames)
    duplicates = identify_duplicate_tracks(cleaned)
    active_ids = [vid for vid in cleaned.keys() if vid not in duplicates]

    candidates = []

    for id_a, id_b in combinations(active_ids, 2):
        vehicle_a = cleaned[id_a]
        vehicle_b = cleaned[id_b]

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
        decel_a = deceleration_a["detected"]
        decel_b = deceleration_b["detected"]
        mutual_decel = decel_a and decel_b
        any_decel = decel_a or decel_b
        motion_detected = relative_motion["detected"]
        close_proximity = relative_motion.get("close_proximity", False)

        score = 0.0
        if overlap_detected:
            score += 0.35 + min(0.20, max_iou * 0.3)
        elif close_proximity:
            score += 0.25

        if mutual_decel:
            score += 0.40
        elif any_decel:
            score += 0.20

        if motion_detected:
            score += 0.25

        score = min(1.0, score)

        if score < 0.3:
            continue

        evidence = []
        if overlap_detected:
            evidence.append(f"bounding_box_overlap (IoU: {max_iou:.2f})")
        elif close_proximity:
            evidence.append(f"close_proximity ({relative_motion['min_distance']}px)")

        if motion_detected:
            evidence.append("relative_closing_motion")

        if decel_a:
            evidence.append(
                f"vehicle_{id_a}_sudden_deceleration (-{deceleration_a['relative_drop']*100:.0f}%)"
            )

        if decel_b:
            evidence.append(
                f"vehicle_{id_b}_sudden_deceleration (-{deceleration_b['relative_drop']*100:.0f}%)"
            )

        if mutual_decel:
            evidence.append("mutual_emergency_braking")

        candidates.append({
            "vehicle_ids": [str(id_a), str(id_b)],
            "classes": [
                vehicle_a["class"],
                vehicle_b["class"]
            ],
            "suspicion_score": round(score, 3),
            "max_iou": round(max_iou, 3),
            "min_distance_px": relative_motion.get("min_distance"),
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


def identify_single_vehicle_incidents(
    trajectories,
    fps=30.0,
    decel_threshold=0.65,
    min_track_frames=6,
):
    """Detect single-vehicle incidents (severe deceleration, rollovers, or barrier impacts)."""
    cleaned = filter_spurious_tracks(trajectories, min_frames=min_track_frames)
    duplicates = identify_duplicate_tracks(cleaned)
    incidents = []

    for vid, data in cleaned.items():
        if vid in duplicates:
            continue
        traj = data.get("trajectory", {})
        if len(traj) < 8:
            continue

        speeds = calculate_speeds(traj)
        decel = detect_abrupt_deceleration(speeds, threshold=decel_threshold)
        if decel["detected"] and decel["frame"] is not None:
            frames = sorted(int(f) for f in traj.keys())
            decel_f = decel["frame"]

            post_decel_frames = [f for f in frames if f >= decel_f]
            post_movement = 0.0
            if len(post_decel_frames) >= 2:
                p_start = traj[str(post_decel_frames[0])]["center"]
                p_end = traj[str(post_decel_frames[-1])]["center"]
                post_movement = calculate_distance(p_start, p_end)

            incidents.append({
                "vehicle_id": str(vid),
                "class": data.get("class", "vehicle"),
                "deceleration_frame": decel_f,
                "deceleration_time": round(decel_f / fps, 2) if fps > 0 else 0.0,
                "relative_drop": decel["relative_drop"],
                "peak_speed": decel["peak_speed"],
                "stopped_after": post_movement < 30.0,
            })

    return incidents


def format_tracking_summary(
    trajectories,
    fps=30.0,
    candidates=None,
    min_track_frames=6,
):
    """Format vehicle trajectories and collision candidate alerts into a concise

    ground truth summary for prompting Vision-Language Models (VLMs).
    """
    cleaned = filter_spurious_tracks(trajectories, min_frames=min_track_frames)
    if not cleaned:
        return "No persistent vehicles were detected or tracked by the object detector."

    duplicates = identify_duplicate_tracks(cleaned)

    if candidates is None:
        candidates = identify_accident_candidates(cleaned, min_track_frames=min_track_frames)

    single_incidents = identify_single_vehicle_incidents(
        cleaned, fps=fps, min_track_frames=min_track_frames
    )

    lines = []
    lines.append(
        f"DETECTED ROAD USERS & TRACKING TELEMETRY (Total: {len(cleaned)} persistent tracked users):"
    )

    for vid, data in sorted(
        cleaned.items(),
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
        total_disp = calculate_distance(start_center, end_center)

        dup_tag = f" [Tracker fragment of Vehicle ID {duplicates[vid]}]" if vid in duplicates else ""

        # Stationary check: if total movement is negligible, label as stationary to prevent false decel
        if total_disp < 15.0 and len(frames) >= 10:
            status_str = f" | [Stationary / Parked vehicle at ({start_center[0]:.0f}, {start_center[1]:.0f})]"
        else:
            speeds = calculate_speeds(data["trajectory"])
            decel = detect_abrupt_deceleration(speeds)
            status_str = ""
            if decel["detected"] and decel["frame"] is not None:
                decel_t = decel["frame"] / fps if fps > 0 else 0.0
                status_str = f" | Sudden deceleration at {decel_t:.2f}s (-{decel['relative_drop']*100:.0f}%)"

        lines.append(
            f"- Vehicle ID {vid} ({cls_name}){dup_tag}: visible from {start_t:.2f}s to {end_t:.2f}s "
            f"(frames {frames[0]}-{frames[-1]}), moved from ({start_center[0]:.0f}, {start_center[1]:.0f}) "
            f"to ({end_center[0]:.0f}, {end_center[1]:.0f}){status_str}"
        )

    lines.append("\nTRACKING COLLISION ALERTS & CANDIDATE INTERACTIONS:")
    alert_count = 0
    if candidates:
        for idx, cand in enumerate(candidates[:3], 1):
            v_pair = " and ".join(f"Vehicle ID {v}" for v in cand["vehicle_ids"])
            overlap_t = (
                f"{cand['overlap_frame'] / fps:.2f}s"
                if (cand.get("overlap_frame") is not None and fps > 0)
                else "N/A"
            )
            evidence_str = ", ".join(cand["evidence"]) if cand.get("evidence") else "trajectory proximity"
            dist_str = f" | Min Distance: {cand['min_distance_px']:.0f}px" if cand.get("min_distance_px") is not None else ""
            lines.append(
                f"  Alert {idx}: {v_pair} | Suspicion Score: {cand['suspicion_score']} | "
                f"Peak Box Overlap at {overlap_t} (IoU: {cand['max_iou']}){dist_str} | Evidence: {evidence_str}"
            )
            alert_count += 1

    # Single-vehicle alerts
    for inc in single_incidents:
        if inc["stopped_after"]:
            alert_count += 1
            lines.append(
                f"  Alert {alert_count} (Single-Vehicle Event): Vehicle ID {inc['vehicle_id']} ({inc['class']}) "
                f"experienced severe deceleration (-{inc['relative_drop']*100:.0f}%) at {inc['deceleration_time']:.2f}s "
                f"and stopped in abnormal position."
            )

    if duplicates:
        dup_notes = [f"ID {k} (duplicate of ID {v})" for k, v in duplicates.items()]
        lines.append(
            f"  Note on Tracking: High box overlap detected for {', '.join(dup_notes)}, "
            "consistent with vehicle deformation or rollover."
        )

    if alert_count == 0:
        lines.append("  No high-probability collisions or bounding-box overlaps detected by the tracker.")

    return "\n".join(lines)
