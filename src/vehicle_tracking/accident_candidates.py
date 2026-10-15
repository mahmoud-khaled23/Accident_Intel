"""Heuristic accident-candidate detection from vehicle trajectory data.

Analyzes per-vehicle trajectories produced by the tracking pipeline and identifies
pairs of vehicles that show signs of a potential collision.  Three independent
signals are combined into a single suspicion score:

- Bounding-box IoU overlap (weight 0.4)
- Relative closing motion between vehicle centers (weight 0.3)
- Abrupt per-vehicle deceleration (weight 0.3)

Key function: :func:`identify_accident_candidates`.
"""

import math
from itertools import combinations


def calculate_iou(box_a, box_b):
    """Compute the Intersection-over-Union (IoU) of two axis-aligned bounding boxes.

    Args:
        box_a: Bounding box in ``[x1, y1, x2, y2]`` pixel coordinates.
        box_b: Bounding box in ``[x1, y1, x2, y2]`` pixel coordinates.

    Returns:
        float: IoU value in the range ``[0.0, 1.0]``.  Returns ``0.0`` when
        both boxes have zero area (degenerate case) so callers never see a
        division-by-zero error.

    Example:
        >>> calculate_iou([0, 0, 10, 10], [5, 5, 15, 15])
        0.14285714285714285
    """
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
    """Compute the Euclidean distance between two 2-D points.

    Args:
        point_a: A ``(x, y)`` coordinate pair.
        point_b: A ``(x, y)`` coordinate pair.

    Returns:
        float: Straight-line pixel distance between the two points.

    Example:
        >>> calculate_distance((0, 0), (3, 4))
        5.0
    """
    return math.hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1]
    )


def calculate_speeds(trajectory):
    """Derive per-frame pixel speeds from a vehicle's tracked trajectory.

    Speed is defined as the Euclidean distance (in pixels) between consecutive
    center positions divided by the frame index gap.  Frames where the gap is
    zero are skipped to avoid division-by-zero errors.

    Args:
        trajectory: A ``dict`` mapping string frame indices to frame data.
            Each value must contain a ``"center"`` key holding ``[cx, cy]``
            pixel coordinates.

    Returns:
        dict: Maps integer frame index to the pixel-per-frame speed leading
        into that frame.  The very first frame has no entry because there is
        no preceding position.

    Example:
        >>> traj = {"0": {"center": [0, 0]}, "1": {"center": [3, 4]}}
        >>> calculate_speeds(traj)
        {1: 5.0}
    """
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
    """Check whether a vehicle experiences a sudden drop in speed.

    Scans the per-frame speed series for the first instance where speed falls
    by at least ``threshold`` (as a fraction of the prior speed).  A 50 %
    drop, for example, suggests emergency braking.

    Args:
        speeds: A ``dict`` mapping integer frame index to pixel-per-frame
            speed, as returned by :func:`calculate_speeds`.
        threshold: Minimum relative speed drop (0 – 1) that counts as abrupt.
            Defaults to ``0.5`` (50 % drop).

    Returns:
        dict: Contains:
            - ``"detected"`` (bool): ``True`` if an abrupt deceleration was found.
            - ``"frame"`` (int | None): Frame where the deceleration occurred,
              or ``None`` if not detected.
            - ``"relative_drop"`` (float): The relative speed drop at that frame,
              or ``0.0`` if not detected.

    Example:
        >>> detect_abrupt_deceleration({1: 10.0, 2: 4.0})
        {'detected': True, 'frame': 2, 'relative_drop': 0.6}
    """
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
    """Detect whether two vehicles are significantly closing in on each other.

    For every frame that has at least ``lookback`` earlier shared frames, the
    function computes how much the distance between the two vehicle centers
    shrunk relative to the distance ``lookback`` frames ago.  A relative
    closing ratio of ≥ 15 % is flagged as converging motion.

    Args:
        trajectory_a: Trajectory dict for vehicle A, keyed by string frame
            index.  Each value must contain a ``"center"`` key.
        trajectory_b: Trajectory dict for vehicle B in the same format.
        lookback: Number of frames to look back when measuring distance change.
            Defaults to ``5``.

    Returns:
        dict: Contains:
            - ``"detected"`` (bool): ``True`` if significant closing motion was found.
            - ``"frame"`` (int | None): Frame at which the condition was met.
            - ``"distance_before"`` (float | None): Distance between centers
              ``lookback`` frames before the detected frame.
            - ``"distance_after"`` (float | None): Distance at the detected frame.

    Example:
        >>> traj_a = {str(i): {"center": [i*10, 0]} for i in range(10)}
        >>> traj_b = {str(i): {"center": [100 - i*10, 0]} for i in range(10)}
        >>> result = analyze_relative_motion(traj_a, traj_b, lookback=5)
        >>> result["detected"]
        True
    """
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
    """Score all vehicle pairs and return those that show signs of a potential accident.

    For every unique pair of tracked vehicles the function computes three
    evidence signals:

    1. **Bounding-box overlap** — maximum IoU across shared frames (weight 0.4).
    2. **Relative closing motion** — whether the vehicles converged by ≥ 15 %
       within a ``lookback``-frame window (weight 0.3).
    3. **Abrupt deceleration** — whether either vehicle braked sharply above
       ``deceleration_threshold`` (weight 0.3).

    Pairs with a combined score below 0.3 are discarded.  The survivors are
    returned sorted by ``suspicion_score`` (highest first).

    Args:
        trajectories: Nested dict produced by :func:`~vehicle_tracking.tracker.process_video`.
            Keys are integer track IDs; each value has ``"class"`` (str) and
            ``"trajectory"`` (dict mapping string frame index to a frame record
            with ``"center"``, ``"box"``, and ``"confidence"``).
        min_iou: Minimum IoU to count bounding-box overlap as evidence.
            Defaults to ``0.1``.
        deceleration_threshold: Minimum relative speed drop (0 – 1) to flag
            as abrupt deceleration.  Defaults to ``0.5``.
        lookback: Frame window used by :func:`analyze_relative_motion`.
            Defaults to ``5``.

    Returns:
        list[dict]: Each element describes one candidate vehicle pair and
        contains:

        - ``"vehicle_ids"`` (list[str]): The two track IDs as strings.
        - ``"classes"`` (list[str]): COCO class names of each vehicle.
        - ``"suspicion_score"`` (float): Combined score in ``[0.0, 1.0]``.
        - ``"max_iou"`` (float): Highest IoU observed for the pair.
        - ``"overlap_frame"`` (int | None): Frame at which max IoU occurred.
        - ``"relative_motion"`` (dict): Output of :func:`analyze_relative_motion`.
        - ``"deceleration"`` (dict): Maps each vehicle ID to its
          :func:`detect_abrupt_deceleration` result.
        - ``"evidence"`` (list[str]): Human-readable evidence tags.

    Example:
        >>> candidates = identify_accident_candidates(trajectories)
        >>> candidates[0]["suspicion_score"]
        0.7
    """
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
