from pathlib import Path
from collections import defaultdict, deque
import csv
import json
import math

import cv2
import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}

CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

SEVERITY = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "DANGER": 3,
}


def resolve_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def bottom_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def orange_ratio_in_box(frame, box):
    x1, y1, x2, y2 = box

    h, w = frame.shape[:2]
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([0, 60, 80])
    upper_orange = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    return float(np.count_nonzero(mask)) / float(mask.size)


def is_valid_person_detection(frame, box, conf, point, crosswalk_rois, waiting_zone):
    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1
    area = box_w * box_h

    if box_w <= 0 or box_h <= 0:
        return False

    aspect_ratio = box_h / max(box_w, 1)

    in_crosswalk = any(
        point_in_polygon(point, roi_item["polygon"])
        for roi_item in crosswalk_rois
    )

    in_waiting = point_in_polygon(point, waiting_zone)
    orange_ratio = orange_ratio_in_box(frame, box)

    if in_crosswalk or in_waiting:
        if conf < 0.20:
            return False
        if box_h < 20 or area < 180:
            return False
        if orange_ratio > 0.40:
            return False
        return True

    if conf < 0.42:
        return False
    if box_h < 35 or area < 500:
        return False
    if aspect_ratio < 1.1 or aspect_ratio > 6.0:
        return False
    if orange_ratio > 0.22 and conf < 0.65:
        return False

    return True


def is_valid_vehicle_detection(box, conf):
    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1
    area = box_w * box_h

    if conf < 0.25:
        return False
    if area < 800:
        return False

    return True


def draw_polygon(frame, points, color, label):
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, 3)

    x, y = pts[0]
    cv2.putText(
        frame,
        label,
        (int(x), int(y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def risk_color(risk_level):
    if risk_level == "LOW":
        return (0, 255, 0)
    if risk_level == "MEDIUM":
        return (0, 255, 255)
    if risk_level == "HIGH":
        return (0, 165, 255)
    if risk_level == "DANGER":
        return (0, 0, 255)
    return (255, 255, 255)


def setup_homography(roi_config):
    homography_cfg = roi_config.get("homography", {})
    enabled = homography_cfg.get("enabled", False)

    if not enabled:
        return None, None, None

    bev_w = int(homography_cfg.get("bev_width", 320))
    bev_h = int(homography_cfg.get("bev_height", 220))

    src = np.array(homography_cfg["src_points"], dtype=np.float32)
    dst = np.array(
        [
            [0, 0],
            [bev_w - 1, 0],
            [bev_w - 1, bev_h - 1],
            [0, bev_h - 1],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(src, dst)
    return H, bev_w, bev_h


def transform_point_homography(point, H):
    if H is None:
        return None

    pts = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pts, H)[0][0]

    return int(transformed[0]), int(transformed[1])


def transform_polygon_homography(points, H):
    if H is None:
        return None

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    return transformed.astype(np.int32)


def draw_bird_eye_panel(frame, roi_config, persons, vehicles, dangerous_pair, H, bev_w, bev_h, risk_level):
    if H is None:
        return frame

    bev = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    bev[:] = (30, 30, 30)

    def draw_bev_poly(key, color, label):
        if key not in roi_config:
            return

        poly = transform_polygon_homography(roi_config[key], H)
        if poly is None:
            return

        cv2.polylines(bev, [poly], True, color, 2)
        x, y = poly[0]

        cv2.putText(
            bev,
            label,
            (int(x), max(15, int(y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    draw_bev_poly("crosswalk_roi", (0, 255, 255), "main")
    draw_bev_poly("secondary_crosswalk_roi", (0, 255, 0), "secondary")
    draw_bev_poly("vehicle_approach_zone", (0, 0, 255), "main app")
    draw_bev_poly("secondary_vehicle_approach_zone", (255, 0, 255), "sec app")

    for p in persons:
        bev_pt = p.get("bev_point")
        if bev_pt is None:
            continue

        x, y = bev_pt
        if 0 <= x < bev_w and 0 <= y < bev_h:
            cv2.circle(bev, (x, y), 5, (0, 255, 255), -1)

    for v in vehicles:
        bev_pt = v.get("bev_point")
        if bev_pt is None:
            continue

        x, y = bev_pt
        if 0 <= x < bev_w and 0 <= y < bev_h:
            cv2.circle(bev, (x, y), 5, (255, 180, 0), -1)

    if dangerous_pair is not None:
        p, v = dangerous_pair
        p_bev = p.get("bev_point")
        v_bev = v.get("bev_point")

        if p_bev is not None and v_bev is not None:
            cv2.line(bev, p_bev, v_bev, risk_color(risk_level), 2)

            d_bev = distance(p_bev, v_bev)
            mid_x = int((p_bev[0] + v_bev[0]) / 2)
            mid_y = int((p_bev[1] + v_bev[1]) / 2)

            cv2.putText(
                bev,
                f"BEV d={d_bev:.1f}",
                (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                risk_color(risk_level),
                1,
            )

    cv2.putText(
        bev,
        "Bird's-eye view",
        (10, bev_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    x0 = frame.shape[1] - bev_w - 20
    y0 = frame.shape[0] - bev_h - 20

    x0 = max(0, x0)
    y0 = max(90, y0)

    frame[y0:y0 + bev_h, x0:x0 + bev_w] = bev
    cv2.rectangle(frame, (x0, y0), (x0 + bev_w, y0 + bev_h), (255, 255, 255), 2)

    return frame


class CrosswalkRiskPipeline:
    def __init__(self, config, project_root: Path):
        self.config = config
        self.project_root = Path(project_root)

        paths = config["paths"]
        model_cfg = config["model"]
        risk_cfg = config["risk"]

        self.input_video = resolve_path(self.project_root, paths["input_video"])
        self.roi_config_path = resolve_path(self.project_root, paths["roi_config"])
        self.output_video = resolve_path(self.project_root, paths["output_video"])
        self.event_log = resolve_path(self.project_root, paths["event_log"])
        self.frame_log = resolve_path(
            self.project_root,
            paths.get("frame_log", "outputs/logs/frame_log_crosswalk_yolo11s_60s.csv")
        )
        self.screenshot_dir = resolve_path(self.project_root, paths["screenshot_dir"])

        self.output_video.parent.mkdir(parents=True, exist_ok=True)
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        self.frame_log.parent.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_cfg["name"]
        self.conf_threshold = float(model_cfg["confidence_threshold"])
        self.img_size = int(model_cfg["image_size"])
        self.tracker = model_cfg["tracker"]

        self.high_distance_px = float(risk_cfg["high_distance_px"])
        self.danger_distance_px = float(risk_cfg["danger_distance_px"])
        self.critical_distance_px = float(risk_cfg["critical_distance_px"])
        self.distance_decrease_margin = float(risk_cfg["distance_decrease_margin"])
        self.history_length = int(risk_cfg["history_length"])
        self.screenshot_cooldown_frames = int(risk_cfg["screenshot_cooldown_frames"])
        self.display_hold_frames = int(risk_cfg["display_hold_frames"])

        score_cfg = config.get("risk_score", {})
        self.proximity_far_px = float(score_cfg.get("proximity_far_px", 180))
        self.proximity_high_px = float(score_cfg.get("proximity_high_px", self.high_distance_px))
        self.proximity_danger_px = float(score_cfg.get("proximity_danger_px", self.danger_distance_px))
        self.proximity_critical_px = float(score_cfg.get("proximity_critical_px", self.critical_distance_px))
        self.ttc_medium_sec = float(score_cfg.get("ttc_medium_sec", 4.0))
        self.ttc_high_sec = float(score_cfg.get("ttc_high_sec", 2.5))
        self.ttc_danger_sec = float(score_cfg.get("ttc_danger_sec", 1.5))
        self.slow_closing_px_per_frame = float(score_cfg.get("slow_closing_px_per_frame", 5))
        self.fast_closing_px_per_frame = float(score_cfg.get("fast_closing_px_per_frame", 12))

    def load_roi(self):
        with open(self.roi_config_path, "r", encoding="utf-8") as f:
            self.roi_config = json.load(f)

        self.main_approach_zone = np.array(self.roi_config["vehicle_approach_zone"], dtype=np.int32)
        self.secondary_approach_zone = np.array(
            self.roi_config.get("secondary_vehicle_approach_zone", self.roi_config["vehicle_approach_zone"]),
            dtype=np.int32,
        )

        self.crosswalk_rois = [
            {
                "name": "main_crosswalk",
                "polygon": np.array(self.roi_config["crosswalk_roi"], dtype=np.int32),
                "approach_zone": self.main_approach_zone,
            }
        ]

        if "secondary_crosswalk_roi" in self.roi_config:
            self.crosswalk_rois.append(
                {
                    "name": "secondary_crosswalk",
                    "polygon": np.array(self.roi_config["secondary_crosswalk_roi"], dtype=np.int32),
                    "approach_zone": self.secondary_approach_zone,
                }
            )

        self.pedestrian_waiting_zone = np.array(self.roi_config["pedestrian_waiting_zone"], dtype=np.int32)
        self.H_bev, self.bev_width, self.bev_height = setup_homography(self.roi_config)

    def score_to_level(self, score):
        if score >= 75:
            return "DANGER"
        if score >= 50:
            return "HIGH"
        if score >= 25:
            return "MEDIUM"
        return "LOW"

    def proximity_score(self, distance_px):
        if distance_px is None:
            return 0
        if distance_px < self.proximity_critical_px:
            return 40
        if distance_px < self.proximity_danger_px:
            return 30
        if distance_px < self.proximity_high_px:
            return 20
        if distance_px < self.proximity_far_px:
            return 10
        return 0

    def closing_speed_score(self, closing_speed_px_per_frame):
        if closing_speed_px_per_frame is None or closing_speed_px_per_frame <= 0:
            return 0
        if closing_speed_px_per_frame >= self.fast_closing_px_per_frame:
            return 20
        if closing_speed_px_per_frame >= self.slow_closing_px_per_frame:
            return 10
        return 0

    def ttc_like_score(self, distance_px, closing_speed_px_per_frame, fps):
        if closing_speed_px_per_frame is None or closing_speed_px_per_frame <= 0:
            return 0, None

        if fps <= 0:
            return 0, None

        ttc_frames = distance_px / closing_speed_px_per_frame
        ttc_sec = ttc_frames / fps

        if ttc_sec < self.ttc_danger_sec:
            return 30, ttc_sec
        if ttc_sec < self.ttc_high_sec:
            return 20, ttc_sec
        if ttc_sec < self.ttc_medium_sec:
            return 10, ttc_sec

        return 0, ttc_sec

    def analyze_risk(self, persons, vehicles, persons_in_crosswalk, persons_waiting, pair_distance_history, fps):
        """
        Image-space surrogate risk score.

        The score is inspired by surrogate safety measures:
        - pedestrian exposure
        - minimum image-space distance
        - closing speed
        - TTC-like image-space approximation
        - vehicle approach zone

        This is not metric-calibrated traffic safety scoring.
        """
        risk_score = 0
        risk_level = "LOW"
        reason = "No pedestrian-related risk detected"
        min_distance = None
        dangerous_pair = None
        best_ttc_like_sec = None
        best_closing_speed = 0.0

        if len(persons_waiting) == 0 and len(persons_in_crosswalk) == 0:
            return (
                "LOW",
                "No pedestrian in crosswalk or waiting zone",
                0,
                None,
                None,
                None,
                0.0,
            )

        if len(persons_waiting) > 0 and len(persons_in_crosswalk) == 0:
            risk_score = max(risk_score, 25)
            risk_level = self.score_to_level(risk_score)
            reason = "Pedestrian waiting near unsignalized crosswalk"

        if len(persons_in_crosswalk) > 0:
            risk_score = max(risk_score, 35)
            risk_level = self.score_to_level(risk_score)
            reason = "Pedestrian inside unsignalized crosswalk"

            for p in persons_in_crosswalk:
                for v in vehicles:
                    d = distance(p["point"], v["point"])

                    pair_key = (p["track_id"], v["track_id"])
                    pair_distance_history[pair_key].append(d)
                    hist = pair_distance_history[pair_key]

                    closing_speed = 0.0
                    distance_decreasing = False

                    if len(hist) >= 5:
                        previous_d = hist[-5]
                        current_d = hist[-1]
                        closing_speed = max(0.0, (previous_d - current_d) / 5.0)
                        distance_decreasing = current_d < previous_d - self.distance_decrease_margin

                    current_approach_zone = p.get("approach_zone", self.main_approach_zone)

                    vehicle_in_relevant_approach_zone = point_in_polygon(
                        v["point"],
                        current_approach_zone
                    )

                    pair_score = 35  # pedestrian exposure: pedestrian inside crosswalk

                    if vehicle_in_relevant_approach_zone:
                        pair_score += 10

                    pair_score += self.proximity_score(d)
                    pair_score += self.closing_speed_score(closing_speed)

                    ttc_score, ttc_like_sec = self.ttc_like_score(
                        distance_px=d,
                        closing_speed_px_per_frame=closing_speed,
                        fps=fps
                    )

                    pair_score += ttc_score
                    pair_score = min(pair_score, 100)

                    if min_distance is None or d < min_distance:
                        min_distance = d

                    if pair_score > risk_score:
                        risk_score = pair_score
                        risk_level = self.score_to_level(risk_score)
                        dangerous_pair = (p, v)
                        best_ttc_like_sec = ttc_like_sec
                        best_closing_speed = closing_speed

                        if risk_level == "DANGER":
                            if d < self.critical_distance_px:
                                reason = f'Pedestrian in {p.get("crosswalk_name", "crosswalk")} and vehicle critically close'
                            elif ttc_like_sec is not None and ttc_like_sec < self.ttc_danger_sec:
                                reason = f'Pedestrian in {p.get("crosswalk_name", "crosswalk")} and short TTC-like risk detected'
                            elif distance_decreasing:
                                reason = f'Pedestrian in {p.get("crosswalk_name", "crosswalk")}, vehicle approaching, and distance decreasing'
                            else:
                                reason = f'Pedestrian in {p.get("crosswalk_name", "crosswalk")} and high surrogate risk score detected'

                        elif risk_level == "HIGH":
                            if d < self.high_distance_px:
                                reason = f'Pedestrian in {p.get("crosswalk_name", "crosswalk")} and vehicle close'
                            elif vehicle_in_relevant_approach_zone:
                                reason = f'Pedestrian in {p.get("crosswalk_name", "crosswalk")} and vehicle in approach zone'
                            else:
                                reason = f'Pedestrian in {p.get("crosswalk_name", "crosswalk")} and elevated surrogate risk score'

                        elif risk_level == "MEDIUM":
                            reason = f'Pedestrian in {p.get("crosswalk_name", "crosswalk")} with moderate surrogate risk score'

        risk_score = int(min(max(risk_score, 0), 100))

        return (
            risk_level,
            reason,
            risk_score,
            min_distance,
            dangerous_pair,
            best_ttc_like_sec,
            best_closing_speed,
        )

    def run(self):
        self.load_roi()

        cap = cv2.VideoCapture(str(self.input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("Input video:", self.input_video)
        print("FPS:", fps)
        print("Resolution:", width, "x", height)
        print("Frames:", frame_count)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_video), fourcc, fps, (width, height))

        model = YOLO(self.model_name)

        track_history = defaultdict(lambda: deque(maxlen=self.history_length))
        pair_distance_history = defaultdict(lambda: deque(maxlen=self.history_length))
        last_screenshot_frame = -9999

        display_risk_level = "LOW"
        display_risk_score = 0
        display_reason = "No pedestrian-related risk detected"
        display_hold_until = 0

        with open(self.event_log, "w", newline="", encoding="utf-8") as f:
            log_writer = csv.writer(f)
            log_writer.writerow([
                "frame_id",
                "time_sec",
                "risk_level",
                "risk_score",
                "person_count",
                "vehicle_count",
                "min_distance_px",
                "ttc_like_sec",
                "closing_speed_px_per_frame",
                "reason",
                "screenshot_path",
            ])

        with open(self.frame_log, "w", newline="", encoding="utf-8") as f:
            frame_log_writer = csv.writer(f)
            frame_log_writer.writerow([
                "frame_id",
                "time_sec",
                "risk_level",
                "risk_score",
                "person_count",
                "vehicle_count",
                "min_distance_px",
                "ttc_like_sec",
                "closing_speed_px_per_frame",
                "reason",
            ])

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            time_sec = frame_id / fps

            results = model.track(
                frame,
                persist=True,
                tracker=self.tracker,
                conf=self.conf_threshold,
                imgsz=self.img_size,
                classes=[0, 1, 2, 3, 5, 7],
                verbose=False,
            )

            persons = []
            vehicles = []

            r = results[0]

            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)

                for box, cls_id, conf, track_id in zip(boxes, class_ids, confs, track_ids):
                    pt = bottom_center(box)
                    bev_pt = transform_point_homography(pt, self.H_bev)

                    track_history[track_id].append((frame_id, pt[0], pt[1]))

                    obj = {
                        "track_id": int(track_id),
                        "class_id": int(cls_id),
                        "class_name": CLASS_NAMES.get(int(cls_id), str(cls_id)),
                        "conf": float(conf),
                        "box": box,
                        "point": pt,
                        "bev_point": bev_pt,
                    }

                    if cls_id == PERSON_CLASS_ID:
                        if is_valid_person_detection(
                            frame=frame,
                            box=box,
                            conf=conf,
                            point=pt,
                            crosswalk_rois=self.crosswalk_rois,
                            waiting_zone=self.pedestrian_waiting_zone,
                        ):
                            persons.append(obj)

                    elif cls_id in VEHICLE_CLASS_IDS:
                        if is_valid_vehicle_detection(box, conf):
                            vehicles.append(obj)

            persons_in_crosswalk = []

            for p in persons:
                for roi_item in self.crosswalk_rois:
                    if point_in_polygon(p["point"], roi_item["polygon"]):
                        p_copy = p.copy()
                        p_copy["crosswalk_name"] = roi_item["name"]
                        p_copy["approach_zone"] = roi_item["approach_zone"]
                        persons_in_crosswalk.append(p_copy)
                        break

            persons_waiting = [
                p for p in persons
                if point_in_polygon(p["point"], self.pedestrian_waiting_zone)
            ]

            (
                risk_level,
                reason,
                risk_score,
                min_distance,
                dangerous_pair,
                ttc_like_sec,
                closing_speed_px_per_frame,
            ) = self.analyze_risk(
                persons=persons,
                vehicles=vehicles,
                persons_in_crosswalk=persons_in_crosswalk,
                persons_waiting=persons_waiting,
                pair_distance_history=pair_distance_history,
                fps=fps,
            )

            if (
                SEVERITY[risk_level] > SEVERITY[display_risk_level]
                or frame_id >= display_hold_until
            ):
                display_risk_level = risk_level
                display_risk_score = risk_score
                display_reason = reason

                if risk_level in ["HIGH", "DANGER"]:
                    display_hold_until = frame_id + self.display_hold_frames
                else:
                    display_hold_until = frame_id + 15

            draw_polygon(frame, self.roi_config["crosswalk_roi"], (0, 255, 255), "main crosswalk")
            draw_polygon(frame, self.roi_config["vehicle_approach_zone"], (0, 0, 255), "main approach")
            draw_polygon(frame, self.roi_config["pedestrian_waiting_zone"], (255, 0, 0), "waiting zone")

            if "secondary_crosswalk_roi" in self.roi_config:
                draw_polygon(frame, self.roi_config["secondary_crosswalk_roi"], (0, 255, 0), "secondary crosswalk")

            if "secondary_vehicle_approach_zone" in self.roi_config:
                draw_polygon(frame, self.roi_config["secondary_vehicle_approach_zone"], (255, 0, 255), "secondary approach")

            for obj in persons + vehicles:
                x1, y1, x2, y2 = obj["box"]

                color = (0, 255, 255) if obj["class_id"] == PERSON_CLASS_ID else (255, 180, 0)

                if dangerous_pair is not None:
                    if obj["track_id"] in [dangerous_pair[0]["track_id"], dangerous_pair[1]["track_id"]]:
                        color = risk_color(risk_level)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, obj["point"], 4, color, -1)

                label = f'{obj["class_name"]} ID:{obj["track_id"]} {obj["conf"]:.2f}'

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            if dangerous_pair is not None:
                p, v = dangerous_pair
                cv2.line(frame, p["point"], v["point"], risk_color(risk_level), 3)

                if min_distance is not None:
                    mid_x = int((p["point"][0] + v["point"][0]) / 2)
                    mid_y = int((p["point"][1] + v["point"][1]) / 2)

                    cv2.putText(
                        frame,
                        f"{min_distance:.1f}px",
                        (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        risk_color(risk_level),
                        2,
                    )

            frame = draw_bird_eye_panel(
                frame=frame,
                roi_config=self.roi_config,
                persons=persons,
                vehicles=vehicles,
                dangerous_pair=dangerous_pair,
                H=self.H_bev,
                bev_w=self.bev_width,
                bev_h=self.bev_height,
                risk_level=risk_level,
            )

            banner_color = risk_color(display_risk_level)
            cv2.rectangle(frame, (0, 0), (width, 85), banner_color, -1)

            cv2.putText(
                frame,
                f"Risk Level: {display_risk_level} | Score: {display_risk_score}/100",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                3,
            )

            cv2.putText(
                frame,
                f"Reason: {display_reason}",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

            if display_risk_level == "DANGER":
                cv2.putText(
                    frame,
                    "AUDIO WARNING: Pedestrian crossing risk detected!",
                    (30, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    3,
                )

            # Write frame-level log every 30 frames, about once per second at 30 FPS
            if frame_id % 30 == 0:
                with open(self.frame_log, "a", newline="", encoding="utf-8") as f:
                    frame_log_writer = csv.writer(f)
                    frame_log_writer.writerow([
                        frame_id,
                        round(time_sec, 2),
                        risk_level,
                        risk_score,
                        len(persons),
                        len(vehicles),
                        round(min_distance, 2) if min_distance is not None else "",
                        round(ttc_like_sec, 3) if ttc_like_sec is not None else "",
                        round(closing_speed_px_per_frame, 3) if closing_speed_px_per_frame is not None else "",
                        reason,
                    ])

            screenshot_path = ""
            should_log = risk_level in ["HIGH", "DANGER"]

            if should_log and frame_id - last_screenshot_frame >= self.screenshot_cooldown_frames:
                screenshot_path = str(self.screenshot_dir / f"{risk_level}_yolo11s_frame_{frame_id}.jpg")
                cv2.imwrite(screenshot_path, frame)
                last_screenshot_frame = frame_id

                with open(self.event_log, "a", newline="", encoding="utf-8") as f:
                    log_writer = csv.writer(f)
                    log_writer.writerow([
                        frame_id,
                        round(time_sec, 2),
                        risk_level,
                        risk_score,
                        len(persons),
                        len(vehicles),
                        round(min_distance, 2) if min_distance is not None else "",
                        round(ttc_like_sec, 3) if ttc_like_sec is not None else "",
                        round(closing_speed_px_per_frame, 3) if closing_speed_px_per_frame is not None else "",
                        reason,
                        screenshot_path,
                    ])

            writer.write(frame)

            frame_id += 1

            if frame_id % 100 == 0:
                print(f"Processed {frame_id}/{frame_count} frames")

        cap.release()
        writer.release()

        print("Done.")
        print("Output video:", self.output_video)
        print("Event log:", self.event_log)
        print("Screenshots:", self.screenshot_dir)

        return {
            "output_video": self.output_video,
            "event_log": self.event_log,
            "screenshot_dir": self.screenshot_dir,
        }
